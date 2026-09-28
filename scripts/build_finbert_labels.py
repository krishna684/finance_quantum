#!/usr/bin/env python
"""Score N24News finance captions with ProsusAI/finbert and write finbert_labeled.parquet.

Sentiment label construction
-----------------------------
  finbert_label  — str: "positive" | "negative" | "neutral"
  finbert_score  — float: model confidence for the predicted class

Input:  data/labeled.parquet   (10,206 rows, already has mkt_ret_next / label_up)
Output: data/finbert_labeled.parquet  (same rows + finbert_label + finbert_score)

Empty captions (2 rows) are assigned finbert_label="neutral", finbert_score=1.0
to avoid inference on blank strings.

Model: ProsusAI/finbert (finance-domain BERT, 3-class: positive/negative/neutral).
Truncation is not needed: captions have max 94 words, well under the 512-token limit.

NOTE: This script does NOT apply any neutral-handling policy (drop / collapse).
It writes the raw 3-class distribution. The caller (Phase 4 sizing) decides the
policy based on the reported counts.

Usage
-----
    python scripts/build_finbert_labels.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = _PROJECT_ROOT / "data" / "labeled.parquet"
OUTPUT       = _PROJECT_ROOT / "data" / "finbert_labeled.parquet"

_MODEL_ID   = "ProsusAI/finbert"
_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# FinBERT inference
# ---------------------------------------------------------------------------

def _run_finbert(captions: list[str]) -> tuple[list[str], list[float]]:
    """Return (labels, scores) for each caption via ProsusAI/finbert."""
    from transformers import pipeline

    print(f"Loading {_MODEL_ID} ...")
    classifier = pipeline(
        "text-classification",
        model=_MODEL_ID,
        device=-1,          # CPU; set to 0 for GPU
        truncation=True,
        max_length=512,
    )
    print(f"  Model loaded. Running inference on {len(captions):,} captions "
          f"(batch_size={_BATCH_SIZE}) ...")

    labels: list[str]  = []
    scores: list[float] = []

    t0 = time.time()
    for i in range(0, len(captions), _BATCH_SIZE):
        batch = captions[i : i + _BATCH_SIZE]
        results = classifier(batch, batch_size=_BATCH_SIZE)
        for r in results:
            labels.append(r["label"].lower())   # "positive"/"negative"/"neutral"
            scores.append(float(r["score"]))
        if (i // _BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done:,}/{len(captions):,}  ({rate:.0f} captions/s)")

    print(f"  Inference done in {time.time() - t0:.1f}s")
    return labels, scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame) -> None:
    errors: list[str] = []

    if len(df) != 10_206:
        errors.append(f"Row count: expected 10,206, got {len(df):,}")

    null_lbl = int(df["finbert_label"].isna().sum())
    if null_lbl:
        errors.append(f"Null finbert_label: {null_lbl}")

    null_scr = int(df["finbert_score"].isna().sum())
    if null_scr:
        errors.append(f"Null finbert_score: {null_scr}")

    valid_labels = {"positive", "negative", "neutral"}
    bad = set(df["finbert_label"].unique()) - valid_labels
    if bad:
        errors.append(f"Unexpected label values: {bad}")

    if errors:
        print("\n[VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    vc = df["finbert_label"].value_counts()
    n  = len(df)
    print("\n[VALIDATION PASSED]")
    print(f"  + Rows:      {n:,}")
    print(f"  + Null labels: 0")
    print()
    print("  3-class distribution (raw, no neutral policy applied):")
    for lbl in ["positive", "negative", "neutral"]:
        cnt = int(vc.get(lbl, 0))
        print(f"    {lbl:<10} {cnt:>6,}  ({cnt/n*100:.1f}%)")
    print()
    # Binary if neutral dropped
    pos = int(vc.get("positive", 0))
    neg = int(vc.get("negative", 0))
    non_neutral = pos + neg
    if non_neutral > 0:
        print(f"  Binary balance (neutral dropped, n={non_neutral:,}):")
        print(f"    positive  {pos:>6,}  ({pos/non_neutral*100:.1f}%)")
        print(f"    negative  {neg:>6,}  ({neg/non_neutral*100:.1f}%)")
    print()
    print(f"  finbert_score  mean={df.finbert_score.mean():.4f}  "
          f"min={df.finbert_score.min():.4f}  max={df.finbert_score.max():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_total = time.time()

    print(f"Loading {LABELED_PATH} ...")
    df = pd.read_parquet(LABELED_PATH)
    print(f"  {len(df):,} rows loaded")

    # Mark empty captions before inference
    empty_mask = df["caption"].isna() | (df["caption"].str.strip() == "")
    n_empty = int(empty_mask.sum())
    if n_empty:
        print(f"  {n_empty} empty captions -> pre-assigned neutral/1.0")

    captions = df["caption"].fillna("").tolist()

    # Replace empty with placeholder that won't trip the model
    captions_for_model = [
        c if c.strip() else "neutral"   # single-word placeholder, maps to neutral anyway
        for c in captions
    ]

    labels, scores = _run_finbert(captions_for_model)

    # Override empty-caption rows
    for i in np.where(empty_mask)[0]:
        labels[i] = "neutral"
        scores[i] = 1.0

    df = df.copy()
    df["finbert_label"] = labels
    df["finbert_score"] = scores

    _validate(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"  -> {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.1f} MB)")
    print(f"Total runtime: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
