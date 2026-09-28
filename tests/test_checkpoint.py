import tempfile
from pathlib import Path

import pytest
from pipeline.checkpoint import CheckpointStore


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    s = CheckpointStore(db)
    yield s
    s.close()


def test_not_done_initially(store):
    assert not store.is_done("add to cart", "20240101", "20240131")


def test_mark_and_check(store):
    store.mark_done("add to cart", "20240101", "20240131", videos_saved=50)
    assert store.is_done("add to cart", "20240101", "20240131")


def test_different_window_not_done(store):
    store.mark_done("add to cart", "20240101", "20240131")
    assert not store.is_done("add to cart", "20240201", "20240229")


def test_different_query_not_done(store):
    store.mark_done("add to cart", "20240101", "20240131")
    assert not store.is_done("drop the link", "20240101", "20240131")


def test_mark_done_idempotent(store):
    store.mark_done("add to cart", "20240101", "20240131")
    store.mark_done("add to cart", "20240101", "20240131")  # should not raise
    assert store.is_done("add to cart", "20240101", "20240131")
