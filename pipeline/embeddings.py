from __future__ import annotations

"""CLIP image embedding extractor for news article images.

Loads news images from SQLite (news_images.image_binary BLOB), runs each
through CLIP ViT-B/32, and stores the resulting 512-dim float32 vectors in a
new image_embeddings table keyed by image_id.  Processing is idempotent:
images that already have an entry in image_embeddings are skipped.

Schema added to news.db
-----------------------
    image_embeddings(image_id PK, embedding BLOB, model_name TEXT,
                     embedding_dim INTEGER, created_at TEXT)

Backends tried in order
-----------------------
    1. open_clip  (open-clip-torch package)
    2. transformers  (HuggingFace)

Usage
-----
    from pipeline.embeddings import ImageEmbedder
    from pathlib import Path

    embedder = ImageEmbedder(Path("data/news.db"))
    embedder.init_schema()
    n = embedder.run(batch_size=32)
    embedder.close()
    print(f"Embedded {n} images")
"""

import io
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.logger import get_logger

log = get_logger(__name__)

_OPENCLIP_MODEL = "ViT-B-32"
_OPENCLIP_PRETRAINED = "openai"
_HF_MODEL = "openai/clip-vit-base-patch32"
_EMBEDDING_DIM = 512


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def encode_embedding(arr) -> bytes:
    """Serialize a 1-D float32 numpy array to raw bytes."""
    import numpy as np

    return np.asarray(arr, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes, dim: int):
    """Deserialize raw bytes back to a float32 numpy array."""
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()


class ImageEmbedder:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._model = None
        self._processor = None
        self._backend: str = ""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_id       TEXT PRIMARY KEY,
            embedding      BLOB NOT NULL,
            model_name     TEXT NOT NULL,
            embedding_dim  INTEGER NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_img_emb ON image_embeddings(image_id);
        """)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        device = torch.device("cpu")

        # Attempt 1: open_clip
        try:
            import open_clip

            log.info("loading_clip_model", backend="open_clip", model=_OPENCLIP_MODEL, device=str(device))
            model, _, preprocess = open_clip.create_model_and_transforms(
                _OPENCLIP_MODEL, pretrained=_OPENCLIP_PRETRAINED
            )
            model = model.to(device)
            model.eval()
            self._model = model
            self._processor = preprocess
            self._backend = "open_clip"
            log.info("clip_model_loaded", backend="open_clip", device=str(device))
            return
        except Exception as exc:
            log.warning("open_clip_unavailable", exc=str(exc))

        # Attempt 2: HuggingFace transformers
        try:
            from transformers import CLIPModel, CLIPProcessor

            log.info("loading_clip_model", backend="transformers", model=_HF_MODEL, device=str(device))
            self._model = CLIPModel.from_pretrained(_HF_MODEL)
            self._processor = CLIPProcessor.from_pretrained(_HF_MODEL)
            self._model = self._model.to(device)
            self._model.eval()
            self._backend = "transformers"
            log.info("clip_model_loaded", backend="transformers", device=str(device))
            return
        except Exception as exc:
            log.error("clip_model_load_failed", exc=str(exc))
            raise RuntimeError(
                "CLIP model unavailable. Install open-clip-torch or transformers."
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _embed_batch(self, pil_images: list) -> list:
        """Return list of L2-normalised float32 numpy arrays (one per image)."""
        import torch

        if self._backend == "open_clip":
            tensors = torch.stack([self._processor(img) for img in pil_images])
            tensors = tensors.to(self._model.device)
            with torch.no_grad():
                features = self._model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)
            return [features[i].float().cpu().numpy() for i in range(len(pil_images))]

        # transformers backend
        inputs = self._processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return [outputs[i].float().cpu().numpy() for i in range(len(pil_images))]

    @staticmethod
    def _pil_from_blob(blob: bytes):
        from PIL import Image

        return Image.open(io.BytesIO(blob)).convert("RGB")

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def _already_embedded(self, image_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM image_embeddings WHERE image_id=?", (image_id,)
        ).fetchone()
        return row is not None

    def run(self, batch_size: int = 32) -> int:
        """Embed all unprocessed images. Returns number of images embedded."""
        all_rows = self.conn.execute(
            "SELECT image_id, image_binary FROM news_images WHERE image_binary IS NOT NULL"
        ).fetchall()

        pending = [
            (r["image_id"], r["image_binary"])
            for r in all_rows
            if not self._already_embedded(r["image_id"])
        ]
        total = len(pending)
        log.info("image_embedding_start", total_pending=total, batch_size=batch_size)

        if total == 0:
            log.info("image_embedding_complete", embedded=0, skipped=len(all_rows))
            return 0

        self._load_model()

        done = 0
        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start : batch_start + batch_size]

            pil_images = []
            valid_ids = []
            for image_id, blob in batch:
                try:
                    pil_images.append(self._pil_from_blob(blob))
                    valid_ids.append(image_id)
                except Exception as exc:
                    log.warning("image_decode_failed", image_id=image_id, exc=str(exc))

            if not pil_images:
                continue

            try:
                embeddings = self._embed_batch(pil_images)
            except Exception as exc:
                log.error("embed_batch_failed", batch_start=batch_start, exc=str(exc))
                continue

            now = _utc_now_iso()
            model_tag = f"{_OPENCLIP_MODEL}/{_OPENCLIP_PRETRAINED}" if self._backend == "open_clip" else _HF_MODEL
            for image_id, emb in zip(valid_ids, embeddings):
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO image_embeddings
                        (image_id, embedding, model_name, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (image_id, encode_embedding(emb), model_tag, int(emb.shape[0]), now),
                )
            self.conn.commit()
            done += len(valid_ids)
            log.info("image_batch_done", done=done, total=total)

        log.info("image_embedding_complete", embedded=done, skipped=len(all_rows) - total)
        return done
