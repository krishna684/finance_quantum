#!/usr/bin/env python
"""Embed N24News finance-subset images with CLIP ViT-B/32.

Uses the same model + preprocessing as pipeline.embeddings and
pipeline.tiktok_collection.frame_embedder:
  - Primary:  open_clip  ViT-B-32 / openai
  - Fallback: transformers  openai/clip-vit-base-patch32
  - Output:   L2-normalised float32, 512 dimensions

Outputs
-------
data/embeddings/n24news_finance.npz
    image_ids   — 1-D str array, length 10206
    embeddings  — float32 array, shape (10206, 512)

data/embeddings/n24news_finance.json
    Sidecar: model name, preprocessing config, creation timestamp.

data/embeddings/.n24news_finance_ckpt.npz
    Rolling checkpoint written every CKPT_EVERY images.
    Deleted automatically on successful completion.
    Rerun the script to resume from checkpoint.

Usage
-----
    python scripts/embed_images.py
    python scripts/embed_images.py --batch-size 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _PROJECT_ROOT / "data" / "index.parquet"
OUT_DIR = _PROJECT_ROOT / "data" / "embeddings"
OUT_NPZ = OUT_DIR / "n24news_finance.npz"
OUT_JSON = OUT_DIR / "n24news_finance.json"
CKPT_PATH = OUT_DIR / ".n24news_finance_ckpt.npz"

_OPENCLIP_MODEL = "ViT-B-32"
_OPENCLIP_PRETRAINED = "openai"
_HF_MODEL = "openai/clip-vit-base-patch32"
_DIM = 512
_DEFAULT_BATCH = 32
_CKPT_EVERY = 500


# ---------------------------------------------------------------------------
# Model loading (mirrors pipeline.embeddings logic exactly)
# ---------------------------------------------------------------------------

def _load_model(device):
    """Try open_clip first, fall back to transformers. Returns (model, preprocess, backend)."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            _OPENCLIP_MODEL, pretrained=_OPENCLIP_PRETRAINED
        )
        model = model.to(device).eval()
        print(f"Backend: open_clip  ({_OPENCLIP_MODEL}/{_OPENCLIP_PRETRAINED})")
        return model, preprocess, "open_clip"
    except Exception as exc:
        print(f"open_clip unavailable ({exc}), falling back to transformers ...")

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(_HF_MODEL).to(device).eval()
    preprocess = CLIPProcessor.from_pretrained(_HF_MODEL)
    print(f"Backend: transformers  ({_HF_MODEL})")
    return model, preprocess, "transformers"


def _embed_batch(model, preprocess, backend, device, pil_images: list) -> np.ndarray:
    import torch
    if backend == "open_clip":
        tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
        with torch.no_grad():
            features = model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    # transformers — use vision_model + visual_projection explicitly to avoid
    # API differences between transformers 4.x (returns tensor) and 5.x
    # (may return BaseModelOutputWithPooling from get_image_features).
    inputs = preprocess(images=pil_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=pixel_values)
        feat = model.visual_projection(vision_out.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> tuple[list[str], list[np.ndarray]]:
    if not CKPT_PATH.exists():
        return [], []
    data = np.load(CKPT_PATH, allow_pickle=True)
    ids = list(data["image_ids"].astype(str))
    embs = list(data["embeddings"].astype(np.float32))
    print(f"Resumed checkpoint: {len(ids):,} images already embedded")
    return ids, embs


def _save_checkpoint(ids: list[str], embs: list[np.ndarray]) -> None:
    tmp = CKPT_PATH.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            tmp,
            image_ids=np.array(ids),
            embeddings=np.array(embs, dtype=np.float32),
        )
        tmp.replace(CKPT_PATH)
        tqdm.write(f"  [ckpt] saved {len(ids):,} embeddings")
    except Exception as exc:
        tqdm.write(f"  [ckpt] SAVE FAILED: {exc}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(all_ids: np.ndarray, all_embs: np.ndarray, expected_n: int) -> None:
    errors: list[str] = []

    if len(all_ids) != expected_n:
        errors.append(f"Row count: expected {expected_n:,}, got {len(all_ids):,}")

    nan_count = int(np.isnan(all_embs).sum())
    if nan_count:
        errors.append(f"NaN values in embeddings: {nan_count}")

    if all_embs.shape[1] != _DIM:
        errors.append(f"Embedding dim: expected {_DIM}, got {all_embs.shape[1]}")

    norms = np.linalg.norm(all_embs, axis=1)
    bad_norms = int((np.abs(norms - 1.0) > 1e-3).sum())
    if bad_norms:
        errors.append(f"L2 norms not ~1.0 for {bad_norms} embeddings")

    if errors:
        print("\n[VALIDATION FAILED]")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("\n[VALIDATION PASSED]")
    print(f"  + Shape:        {all_embs.shape}")
    print(f"  + NaN count:    0")
    print(f"  + L2 norm:      min={norms.min():.6f}  max={norms.max():.6f}  mean={norms.mean():.6f}")
    print(f"  + Value range:  min={all_embs.min():.4f}  max={all_embs.max():.4f}")
    print(f"  + Unique IDs:   {len(set(all_ids.tolist())):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(batch_size: int = _DEFAULT_BATCH) -> None:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load index, filter to finance subset
    print(f"Loading index from {INDEX_PATH} ...")
    df = pd.read_parquet(INDEX_PATH)
    finance_df = df[df["is_finance"]].reset_index(drop=True)
    print(f"Finance rows: {len(finance_df):,}")

    # Resume from checkpoint
    done_ids, done_embs = _load_checkpoint()
    done_set = set(done_ids)

    pending = finance_df[~finance_df["image_id"].isin(done_set)].reset_index(drop=True)
    print(f"To embed: {len(pending):,}  |  Already done: {len(done_ids):,}")

    model, preprocess, backend = None, None, None
    t0 = time.time()

    if len(pending) > 0:
        model, preprocess, backend = _load_model(device)
        records = list(pending[["image_id", "image_path"]].itertuples(index=False))
        since_ckpt = 0

        with tqdm(total=len(records), unit="img", desc="Embedding") as bar:
            for batch_start in range(0, len(records), batch_size):
                batch = records[batch_start: batch_start + batch_size]

                pil_images: list = []
                valid_ids: list[str] = []
                for rec in batch:
                    try:
                        img = Image.open(rec.image_path).convert("RGB")
                        pil_images.append(img)
                        valid_ids.append(rec.image_id)
                    except Exception as exc:
                        tqdm.write(f"WARN open {rec.image_path}: {exc}")

                if not pil_images:
                    bar.update(len(batch))
                    continue

                try:
                    emb_batch = _embed_batch(model, preprocess, backend, device, pil_images)
                except Exception as exc:
                    tqdm.write(f"ERROR batch @{batch_start}: {exc}")
                    bar.update(len(batch))
                    continue

                for iid, emb in zip(valid_ids, emb_batch):
                    done_ids.append(iid)
                    done_embs.append(emb)

                since_ckpt += len(valid_ids)
                bar.update(len(batch))

                if since_ckpt >= _CKPT_EVERY:
                    _save_checkpoint(done_ids, done_embs)
                    since_ckpt = 0

        elapsed = time.time() - t0
        print(f"\nEmbedding completed in {elapsed:.1f}s  ({elapsed / max(len(records), 1):.2f}s/img)")
    else:
        print("Nothing new to embed — writing final output from checkpoint.")

    # Assemble final arrays
    all_ids = np.array(done_ids)
    all_embs = np.array(done_embs, dtype=np.float32)

    _validate(all_ids, all_embs, expected_n=len(finance_df))

    # Write output
    np.savez_compressed(OUT_NPZ, image_ids=all_ids, embeddings=all_embs)
    size_mb = OUT_NPZ.stat().st_size / 1e6
    print(f"\n  -> {OUT_NPZ}  ({size_mb:.1f} MB)")

    # Determine model tag for sidecar
    if backend == "open_clip":
        model_tag = f"{_OPENCLIP_MODEL}/{_OPENCLIP_PRETRAINED}"
        preprocess_desc = (
            "open_clip standard: resize shortest edge to 224, center crop 224x224, "
            "normalize mean=(0.48145466,0.4578275,0.40821073) "
            "std=(0.26862954,0.26130258,0.27577711)"
        )
    else:
        model_tag = _HF_MODEL
        preprocess_desc = (
            "CLIPProcessor: resize to 224x224, center crop, "
            "normalize mean=(0.48145466,0.4578275,0.40821073) "
            "std=(0.26862954,0.26130258,0.27577711)"
        )

    sidecar = {
        "model": model_tag,
        "backend": backend or "checkpoint_resume",
        "preprocessing": preprocess_desc,
        "embedding_dim": _DIM,
        "l2_normalized": True,
        "n_images": int(len(all_ids)),
        "source_filter": "n24news  is_finance==True",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    OUT_JSON.write_text(json.dumps(sidecar, indent=2))
    print(f"  -> {OUT_JSON}")

    # Remove checkpoint on success
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
        print("Checkpoint removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed N24News finance images with CLIP")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH)
    args = parser.parse_args()
    main(batch_size=args.batch_size)
