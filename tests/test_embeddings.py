"""Tests for pipeline.embeddings (ImageEmbedder)."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    """Create a minimal news.db with the news_images table."""
    db = tmp_path / "news.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE news_images (
            image_id TEXT PRIMARY KEY,
            article_id TEXT,
            image_binary BLOB,
            caption TEXT,
            placement TEXT,
            image_order INTEGER,
            width_px INTEGER,
            height_px INTEGER,
            sha256_hash TEXT UNIQUE,
            file_extension TEXT,
            collection_timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db


def _tiny_png_bytes() -> bytes:
    """Return a minimal 1×1 white PNG as bytes."""
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        # Return a hardcoded 1x1 PNG if Pillow not installed
        return (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00'
            b'\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )


def _insert_image(db: Path, image_id: str, blob: bytes | None = None) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO news_images (image_id, image_binary, sha256_hash) VALUES (?, ?, ?)",
        (image_id, blob or _tiny_png_bytes(), f"hash_{image_id}"),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# encode / decode round-trip
# ---------------------------------------------------------------------------

def test_encode_decode_round_trip():
    from pipeline.embeddings import decode_embedding, encode_embedding

    arr = np.random.rand(512).astype(np.float32)
    blob = encode_embedding(arr)
    recovered = decode_embedding(blob, 512)
    np.testing.assert_array_almost_equal(arr, recovered)


def test_encode_returns_bytes():
    from pipeline.embeddings import encode_embedding

    arr = np.ones(512, dtype=np.float32)
    assert isinstance(encode_embedding(arr), bytes)
    assert len(encode_embedding(arr)) == 512 * 4  # float32 = 4 bytes


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def test_init_schema_creates_table(tmp_path):
    from pipeline.embeddings import ImageEmbedder

    db = _make_db(tmp_path)
    embedder = ImageEmbedder(db)
    embedder.init_schema()
    embedder.close()

    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(image_embeddings)")}
    conn.close()
    assert {"image_id", "embedding", "model_name", "embedding_dim", "created_at"} <= cols


def test_init_schema_idempotent(tmp_path):
    from pipeline.embeddings import ImageEmbedder

    db = _make_db(tmp_path)
    embedder = ImageEmbedder(db)
    embedder.init_schema()
    embedder.init_schema()  # second call must not raise
    embedder.close()


# ---------------------------------------------------------------------------
# Already-embedded check
# ---------------------------------------------------------------------------

def test_already_embedded_false_when_empty(tmp_path):
    from pipeline.embeddings import ImageEmbedder

    db = _make_db(tmp_path)
    _insert_image(db, "img1")
    embedder = ImageEmbedder(db)
    embedder.init_schema()
    assert not embedder._already_embedded("img1")
    embedder.close()


def test_already_embedded_true_after_insert(tmp_path):
    from pipeline.embeddings import ImageEmbedder, encode_embedding

    db = _make_db(tmp_path)
    embedder = ImageEmbedder(db)
    embedder.init_schema()
    embedder.conn.execute(
        "INSERT INTO image_embeddings (image_id, embedding, model_name, embedding_dim, created_at)"
        " VALUES ('img1', ?, 'test', 512, '2024-01-01T00:00:00+00:00')",
        (encode_embedding(np.zeros(512, dtype=np.float32)),),
    )
    embedder.conn.commit()
    assert embedder._already_embedded("img1")
    embedder.close()


# ---------------------------------------------------------------------------
# run() with mocked model
# ---------------------------------------------------------------------------

def _make_mock_embedder_instance(db: Path):
    """Return an ImageEmbedder with a mocked CLIP model."""
    from pipeline.embeddings import ImageEmbedder

    embedder = ImageEmbedder(db)
    embedder.init_schema()

    # Simulate open_clip backend
    embedder._backend = "open_clip"
    embedder._model = MagicMock()
    embedder._processor = MagicMock(side_effect=lambda img: np.zeros((3, 224, 224)))

    def fake_embed_batch(pil_images):
        return [np.random.rand(512).astype(np.float32) for _ in pil_images]

    embedder._embed_batch = fake_embed_batch
    return embedder


def test_run_embeds_images(tmp_path):
    db = _make_db(tmp_path)
    _insert_image(db, "img1")
    _insert_image(db, "img2")

    embedder = _make_mock_embedder_instance(db)
    n = embedder.run(batch_size=8)
    embedder.close()

    assert n == 2
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0]
    conn.close()
    assert count == 2


def test_run_skips_already_embedded(tmp_path):
    from pipeline.embeddings import encode_embedding

    db = _make_db(tmp_path)
    _insert_image(db, "img1")
    _insert_image(db, "img2")

    embedder = _make_mock_embedder_instance(db)
    # Pre-insert img1
    embedder.conn.execute(
        "INSERT INTO image_embeddings (image_id, embedding, model_name, embedding_dim, created_at)"
        " VALUES ('img1', ?, 'test', 512, '2024-01-01T00:00:00+00:00')",
        (encode_embedding(np.zeros(512, dtype=np.float32)),),
    )
    embedder.conn.commit()

    n = embedder.run(batch_size=8)
    embedder.close()

    assert n == 1  # only img2 was embedded


def test_run_skips_null_blob(tmp_path):
    db = _make_db(tmp_path)
    _insert_image(db, "img_null", blob=None)

    # Insert row with NULL binary (violates NOT NULL but test via patched query)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE news_images SET image_binary = NULL WHERE image_id = 'img_null'"
    )
    conn.commit()
    conn.close()

    embedder = _make_mock_embedder_instance(db)
    n = embedder.run(batch_size=8)
    embedder.close()
    # NULL images are filtered out by WHERE clause
    assert n == 0


def test_run_handles_corrupt_blob(tmp_path):
    db = _make_db(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO news_images (image_id, image_binary, sha256_hash) VALUES (?, ?, ?)",
        ("img_bad", b"not_an_image", "hash_bad"),
    )
    conn.commit()
    conn.close()

    embedder = _make_mock_embedder_instance(db)
    # _pil_from_blob should log a warning and skip, not raise
    n = embedder.run(batch_size=8)
    embedder.close()
    assert n == 0


def test_run_idempotent(tmp_path):
    db = _make_db(tmp_path)
    _insert_image(db, "img1")

    embedder = _make_mock_embedder_instance(db)
    n1 = embedder.run(batch_size=8)
    n2 = embedder.run(batch_size=8)
    embedder.close()

    assert n1 == 1
    assert n2 == 0  # already embedded on second run

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM image_embeddings").fetchone()[0]
    conn.close()
    assert count == 1
