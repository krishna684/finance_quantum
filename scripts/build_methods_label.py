#!/usr/bin/env python
"""Build data/methods_label.parquet: binary FinBERT label for the Phase 4 methods leg.

Source text field
-----------------
FinBERT (ProsusAI/finbert) was scored on the CAPTION field of each N24News
finance article. Provenance is recoverable from scripts/build_finbert_labels.py:

    captions = df["caption"].fillna("").tolist()   # line ~85

The caption, headline, and abstract are all present in labeled.parquet, but
only captions were fed to the model. This is load-bearing for the paper:
captions are the image-associated text; the model is predicting caption
sentiment from image features, which is the cross-modal task.

Label construction
------------------
  finbert_bin   — int: 1=positive, 0=negative  (neutral rows DROPPED)
  finbert_score — float: model confidence for the predicted class

Neutral handling: DROP. Collapsing neutral to either pole would be arbitrary
and would worsen the 77% negative skew. Dropping yields a clean binary label
on 3,612 rows (pos=827, neg=2,785).

Input:  data/finbert_labeled.parquet   (10,206 rows, 3-class)
Output: data/methods_label.parquet     (3,612 rows, binary)

Schema
------
  image_id       str    — joins to n24news_finance.npz
  trading_day    date   — next NYSE session after publication (= next_trading_day)
  finbert_bin    int    — 1=positive caption sentiment, 0=negative
  finbert_score  float  — FinBERT confidence for the predicted class

Usage
-----
    python scripts/build_methods_label.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINBERT_PATH  = _PROJECT_ROOT / "data" / "finbert_labeled.parquet"
NPZ_PATH      = _PROJECT_ROOT / "data" / "embeddings" / "n24news_finance.npz"
OUTPUT        = _PROJECT_ROOT / "data" / "methods_label.parquet"

_EXPECTED_ROWS = 3_612
_EXPECTED_POS  = 827
_EXPECTED_NEG  = 2_785
_TOLERANCE     = 0.01   # 1 %


def main() -> None:
    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"Loading {FINBERT_PATH} ...")
    fb = pd.read_parquet(FINBERT_PATH)
    print(f"  Full rows: {len(fb):,}")
    raw_counts = dict(fb["finbert_label"].value_counts())
    print(f"  Raw label counts: {raw_counts}")

    # ------------------------------------------------------------------
    # Provenance report (load-bearing — do not remove)
    # ------------------------------------------------------------------
    print()
    print("=== FinBERT source field ===")
    print("  Field scored : CAPTION")
    print("  Script       : scripts/build_finbert_labels.py")
    print("  Evidence     : captions = df['caption'].fillna('').tolist()")
    print("                 (the 'caption' column of labeled.parquet)")
    print("  NOT scored   : headline, abstract (both present in parquet but unused)")
    print()

    # ------------------------------------------------------------------
    # Drop neutral, encode
    # ------------------------------------------------------------------
    non_neutral = fb[fb["finbert_label"] != "neutral"].copy()
    pos_count   = int((non_neutral["finbert_label"] == "positive").sum())
    neg_count   = int((non_neutral["finbert_label"] == "negative").sum())
    print(f"Non-neutral rows: {len(non_neutral):,}  "
          f"(positive={pos_count:,}  negative={neg_count:,})")

    errors: list[str] = []
    if abs(len(non_neutral) - _EXPECTED_ROWS) / _EXPECTED_ROWS > _TOLERANCE:
        errors.append(
            f"Row count {len(non_neutral):,} deviates from expected "
            f"{_EXPECTED_ROWS:,} by >{_TOLERANCE*100:.0f}%"
        )
    if abs(pos_count - _EXPECTED_POS) / _EXPECTED_POS > _TOLERANCE:
        errors.append(
            f"Positive count {pos_count:,} deviates from expected "
            f"{_EXPECTED_POS:,} by >{_TOLERANCE*100:.0f}%"
        )
    if abs(neg_count - _EXPECTED_NEG) / _EXPECTED_NEG > _TOLERANCE:
        errors.append(
            f"Negative count {neg_count:,} deviates from expected "
            f"{_EXPECTED_NEG:,} by >{_TOLERANCE*100:.0f}%"
        )
    if errors:
        print("[VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)
    print("  Count validation PASSED")

    non_neutral["finbert_bin"] = (
        non_neutral["finbert_label"].map({"positive": 1, "negative": 0}).astype(int)
    )

    # ------------------------------------------------------------------
    # Build output frame
    # ------------------------------------------------------------------
    out = non_neutral[["image_id", "next_trading_day", "finbert_bin", "finbert_score"]].copy()
    out = out.rename(columns={"next_trading_day": "trading_day"})

    # ------------------------------------------------------------------
    # Embedding coverage check
    # ------------------------------------------------------------------
    print(f"\nVerifying embedding coverage against {NPZ_PATH} ...")
    npz      = np.load(NPZ_PATH, allow_pickle=True)
    emb_ids  = set(npz["image_ids"].astype(str))
    needed   = set(out["image_id"].astype(str))
    missing  = needed - emb_ids
    if missing:
        print(f"[VALIDATION FAILED] {len(missing):,} records have no embedding in NPZ.")
        print(f"  First 5 missing IDs: {sorted(missing)[:5]}")
        sys.exit(1)
    print(f"  Embedding coverage: {len(needed):,} / {len(needed):,} -- 100% OK")

    # ------------------------------------------------------------------
    # Null / integrity checks
    # ------------------------------------------------------------------
    null_checks = {
        "image_id":      int(out["image_id"].isna().sum()),
        "trading_day":   int(out["trading_day"].isna().sum()),
        "finbert_bin":   int(out["finbert_bin"].isna().sum()),
        "finbert_score": int(out["finbert_score"].isna().sum()),
    }
    null_errors = [f"Null {col}: {cnt}" for col, cnt in null_checks.items() if cnt]
    if null_errors:
        print("[VALIDATION FAILED]")
        for e in null_errors:
            print(f"  x {e}")
        sys.exit(1)

    unique_days = out["trading_day"].nunique()
    n_pos = int(out["finbert_bin"].sum())
    n_neg = len(out) - n_pos

    print()
    print("[VALIDATION PASSED]")
    print(f"  + Rows:              {len(out):,}")
    print(f"  + finbert_bin=1 (+): {n_pos:,}  ({n_pos / len(out) * 100:.1f}%)")
    print(f"  + finbert_bin=0 (-): {n_neg:,}  ({n_neg / len(out) * 100:.1f}%)")
    print(f"  + Nulls:             all 0")
    print(f"  + Date range:        {out['trading_day'].min()} -> {out['trading_day'].max()}")
    print(f"  + Unique trading_days: {unique_days:,}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"\n  -> {OUTPUT}  ({OUTPUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
