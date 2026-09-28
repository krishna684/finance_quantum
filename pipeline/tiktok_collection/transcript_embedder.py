from __future__ import annotations

"""FinBERT transcript embedder for TikTok posts.

Reads transcripts from tiktok_fin_posts and writes mean-pooled, L2-normalised
embeddings into text_embeddings in the same database.  Schema and inference
are identical to pipeline.text_embeddings (TextEmbedder) — the only
difference is the data source.

article_id in text_embeddings = "tiktok_{post_id}"

This allows the future ClassificationPipeline extension to join:
    text_embeddings (WHERE article_id LIKE 'tiktok_%') → tiktok_fin_posts
analogously to the existing:
    text_embeddings → news_articles

FinBERT tokenizer has a 512-token max sequence length.  Transcripts are
truncated via the tokenizer's built-in truncation (max_length=512).
"""

from datetime import datetime, timezone
from pathlib import Path

from pipeline.text_embeddings import decode_embedding, encode_embedding
from pipeline.logger import get_logger
from pipeline.tiktok_collection.storage import FinCollectionStorage

log = get_logger(__name__)

_FINBERT = "ProsusAI/finbert"
_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_TRANSCRIPT_CHARS = 4096   # rough guard before tokenisation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text_embedding_id(post_id: str) -> str:
    return f"tiktok_{post_id}"


class TikTokTranscriptEmbedder:
    """Embed TikTok transcripts using FinBERT (or MiniLM fallback).

    Usage::

        embedder = TikTokTranscriptEmbedder(storage)
        n = embedder.run(batch_size=64)
    """

    def __init__(self, storage: FinCollectionStorage, prefer_finbert: bool = True) -> None:
        self._storage = storage
        self._prefer_finbert = prefer_finbert
        self._model = None
        self._tokenizer = None
        self._model_name: str = ""

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        device = torch.device("cpu")
        candidates = [_FINBERT, _MINILM] if self._prefer_finbert else [_MINILM, _FINBERT]
        last_exc: Exception | None = None

        for model_id in candidates:
            try:
                from transformers import AutoModel, AutoTokenizer
                log.info("transcript_embedder_loading", model=model_id)
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModel.from_pretrained(model_id).to(device)
                self._model.eval()
                self._model_name = model_id
                log.info("transcript_embedder_loaded", model=model_id)
                return
            except Exception as exc:
                log.warning("transcript_embedder_load_failed", model=model_id, exc=str(exc))
                last_exc = exc

        raise RuntimeError("Could not load any text model.") from last_exc

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        import torch
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

    def _embed_batch(self, texts: list[str]) -> list:
        import torch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return [pooled[i].float().cpu().numpy() for i in range(len(texts))]

    def run(self, batch_size: int = 64) -> int:
        """Embed all transcripts not yet in text_embeddings. Returns count embedded."""
        pending = self._storage.unembedded_transcripts()
        total = len(pending)
        log.info("transcript_embedding_start", total_pending=total, batch_size=batch_size)

        if total == 0:
            return 0

        self._load_model()
        done = 0

        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start: batch_start + batch_size]

            non_empty = [
                (pid, t[:_MAX_TRANSCRIPT_CHARS])
                for pid, t in batch
                if (t or "").strip()
            ]
            if not non_empty:
                continue

            try:
                embeddings = self._embed_batch([t for _, t in non_empty])
            except Exception as exc:
                log.error("transcript_embed_batch_failed", batch_start=batch_start, exc=str(exc))
                continue

            now = _utc_now_iso()
            for (pid, _), emb in zip(non_empty, embeddings):
                article_id = _text_embedding_id(pid)
                self._storage.conn.execute(
                    """
                    INSERT OR IGNORE INTO text_embeddings
                        (article_id, embedding, model_name, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (article_id, encode_embedding(emb), self._model_name, int(emb.shape[0]), now),
                )
            self._storage.commit()
            done += len(non_empty)
            log.info("transcript_embed_batch_done", done=done, total=total)

        log.info("transcript_embedding_complete", embedded=done, skipped=total - done)
        return done
