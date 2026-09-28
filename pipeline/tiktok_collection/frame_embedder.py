from __future__ import annotations

"""CLIP image embedder adapter for TikTok video frames.

Reads JPEG frames from tiktok_frames (via FinCollectionStorage) and writes
L2-normalised 512-dim float32 embeddings into image_embeddings in the same
database.  Schema and CLIP inference are identical to pipeline.embeddings
(ImageEmbedder) — the only difference is the data source table.

image_id in image_embeddings = frame_id from tiktok_frames
(e.g. "7123456789_fr00002" or "7123456789_thumb")

This allows the future ClassificationPipeline extension to join:
    image_embeddings → tiktok_frames → tiktok_fin_posts
analogously to the existing:
    image_embeddings → news_images → news_articles
"""

import io
from datetime import datetime, timezone
from pathlib import Path

from pipeline.embeddings import decode_embedding, encode_embedding
from pipeline.logger import get_logger
from pipeline.tiktok_collection.storage import FinCollectionStorage

log = get_logger(__name__)

_OPENCLIP_MODEL = "ViT-B-32"
_OPENCLIP_PRETRAINED = "openai"
_HF_MODEL = "openai/clip-vit-base-patch32"
_EMBEDDING_DIM = 512


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TikTokFrameEmbedder:
    """Embed TikTok video frames into image_embeddings using CLIP ViT-B/32.

    Usage::

        embedder = TikTokFrameEmbedder(storage)
        embedder.init_model()
        n = embedder.run(batch_size=32)
    """

    def __init__(self, storage: FinCollectionStorage) -> None:
        self._storage = storage
        self._model = None
        self._processor = None
        self._backend: str = ""

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        device = torch.device("cpu")

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                _OPENCLIP_MODEL, pretrained=_OPENCLIP_PRETRAINED
            )
            model = model.to(device)
            model.eval()
            self._model = model
            self._processor = preprocess
            self._backend = "open_clip"
            log.info("frame_embedder_model_loaded", backend="open_clip", device=str(device))
            return
        except Exception as exc:
            log.warning("frame_embedder_open_clip_unavailable", exc=str(exc))

        try:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained(_HF_MODEL).to(device)
            self._processor = CLIPProcessor.from_pretrained(_HF_MODEL)
            self._model.eval()
            self._backend = "transformers"
            log.info("frame_embedder_model_loaded", backend="transformers", device=str(device))
        except Exception as exc:
            log.error("frame_embedder_model_load_failed", exc=str(exc))
            raise RuntimeError("CLIP model unavailable.") from exc

    def _embed_batch(self, pil_images: list) -> list:
        import torch

        if self._backend == "open_clip":
            tensors = torch.stack([self._processor(img) for img in pil_images])
            tensors = tensors.to(self._model.device)
            with torch.no_grad():
                features = self._model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            return [features[i].float().cpu().numpy() for i in range(len(pil_images))]

        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return [outputs[i].float().cpu().numpy() for i in range(len(pil_images))]

    @staticmethod
    def _pil_from_blob(blob: bytes):
        from PIL import Image
        return Image.open(io.BytesIO(blob)).convert("RGB")

    def run(self, batch_size: int = 32) -> int:
        """Embed all frames not yet in image_embeddings.  Returns count embedded."""
        pending = self._storage.unembedded_frames()
        total = len(pending)
        log.info("frame_embedding_start", total_pending=total, batch_size=batch_size)

        if total == 0:
            return 0

        self._load_model()

        model_tag = (
            f"{_OPENCLIP_MODEL}/{_OPENCLIP_PRETRAINED}"
            if self._backend == "open_clip"
            else _HF_MODEL
        )
        done = 0

        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start: batch_start + batch_size]

            pil_images = []
            valid_ids = []
            for frame_id, blob in batch:
                try:
                    pil_images.append(self._pil_from_blob(blob))
                    valid_ids.append(frame_id)
                except Exception as exc:
                    log.warning("frame_decode_failed", frame_id=frame_id, exc=str(exc))

            if not pil_images:
                continue

            try:
                embeddings = self._embed_batch(pil_images)
            except Exception as exc:
                log.error("frame_embed_batch_failed", batch_start=batch_start, exc=str(exc))
                continue

            now = _utc_now_iso()
            for fid, emb in zip(valid_ids, embeddings):
                self._storage.conn.execute(
                    """
                    INSERT OR IGNORE INTO image_embeddings
                        (image_id, embedding, model_name, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (fid, encode_embedding(emb), model_tag, int(emb.shape[0]), now),
                )
            self._storage.commit()
            done += len(valid_ids)
            log.info("frame_embed_batch_done", done=done, total=total)

        log.info("frame_embedding_complete", embedded=done, skipped=total - done)
        return done
