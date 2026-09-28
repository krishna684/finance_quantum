from __future__ import annotations

"""Text embedding extractor for news articles.

Concatenates headline + body, encodes with a transformer model, and stores
mean-pooled L2-normalised embeddings in a text_embeddings table.

Model preference order
----------------------
    1. ProsusAI/finbert  (finance-domain BERT, 768-dim)
    2. sentence-transformers/all-MiniLM-L6-v2  (general, 384-dim)

Schema added to news.db
-----------------------
    text_embeddings(article_id PK, embedding BLOB, model_name TEXT,
                    embedding_dim INTEGER, created_at TEXT)

Usage
-----
    from pipeline.text_embeddings import TextEmbedder
    from pathlib import Path

    embedder = TextEmbedder(Path("data/news.db"))
    embedder.init_schema()
    n = embedder.run(batch_size=64)
    embedder.close()
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from pipeline.logger import get_logger

log = get_logger(__name__)

_FINBERT = "ProsusAI/finbert"
_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_BODY_CHARS = 4096  # guard against very long bodies before tokenisation


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


class TextEmbedder:
    def __init__(self, db_path: Path, prefer_finbert: bool = True) -> None:
        self.db_path = db_path
        self.prefer_finbert = prefer_finbert
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._model = None
        self._tokenizer = None
        self._model_name: str = ""

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS text_embeddings (
            article_id     TEXT PRIMARY KEY,
            embedding      BLOB NOT NULL,
            model_name     TEXT NOT NULL,
            embedding_dim  INTEGER NOT NULL,
            created_at     TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_txt_emb ON text_embeddings(article_id);
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

        candidates = [_FINBERT, _MINILM] if self.prefer_finbert else [_MINILM, _FINBERT]
        last_exc: Exception | None = None

        for model_id in candidates:
            try:
                from transformers import AutoModel, AutoTokenizer

                log.info("loading_text_model", model=model_id, device=str(device))
                self._tokenizer = AutoTokenizer.from_pretrained(model_id)
                self._model = AutoModel.from_pretrained(model_id)
                self._model = self._model.to(device)
                self._model.eval()
                self._model_name = model_id
                log.info("text_model_loaded", model=model_id, device=str(device))
                return
            except Exception as exc:
                log.warning("text_model_load_failed", model=model_id, exc=str(exc))
                last_exc = exc

        raise RuntimeError(
            f"Could not load any text model. Install transformers and ensure "
            f"internet access for model download."
        ) from last_exc

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(token_embeddings, attention_mask):
        """Attention-mask-weighted mean over token dimension."""
        import torch

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )

    def _embed_batch(self, texts: list[str]) -> list:
        """Return list of L2-normalised float32 numpy arrays."""
        import torch

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)

        pooled = self._mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return [pooled[i].float().cpu().numpy() for i in range(len(texts))]

    @staticmethod
    def _make_text(headline: str | None, body: str | None) -> str:
        parts: list[str] = []
        if headline:
            parts.append(headline.strip())
        if body:
            parts.append(body.strip()[:_MAX_BODY_CHARS])
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def _already_embedded(self, article_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM text_embeddings WHERE article_id=?", (article_id,)
        ).fetchone()
        return row is not None

    def run(self, batch_size: int = 64) -> int:
        """Embed all articles not yet in text_embeddings. Returns count embedded."""
        self._load_model()

        all_rows = self.conn.execute(
            "SELECT article_id, headline, full_body FROM news_articles"
        ).fetchall()

        pending = [
            (r["article_id"], r["headline"], r["full_body"])
            for r in all_rows
            if not self._already_embedded(r["article_id"])
        ]
        total = len(pending)
        log.info("text_embedding_start", total_pending=total, batch_size=batch_size)

        done = 0
        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start : batch_start + batch_size]

            non_empty = [
                (aid, self._make_text(h, b))
                for aid, h, b in batch
                if self._make_text(h, b).strip()
            ]
            if not non_empty:
                continue

            try:
                embeddings = self._embed_batch([t for _, t in non_empty])
            except Exception as exc:
                log.error("text_batch_failed", batch_start=batch_start, exc=str(exc))
                continue

            now = _utc_now_iso()
            for (article_id, _), emb in zip(non_empty, embeddings):
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO text_embeddings
                        (article_id, embedding, model_name, embedding_dim, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (article_id, encode_embedding(emb), self._model_name, int(emb.shape[0]), now),
                )
            self.conn.commit()
            done += len(non_empty)
            log.info("text_batch_done", done=done, total=total)

        log.info("text_embedding_complete", embedded=done, skipped=len(all_rows) - total)
        return done
