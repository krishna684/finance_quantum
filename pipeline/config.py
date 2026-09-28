from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    client_key: str
    client_secret: str
    access_token: str
    db_path: Path
    query_videos_target: int
    random_low_engagement_target: int
    comments_per_video_target: int
    comment_scan_limit: int
    window_days: int
    lookback_months: int
    region_codes: list[str]
    log_level: str
    # Financial TikTok collection (Bright Data architecture) — all have defaults
    brightdata_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ticker_universe_path: Path | None = None
    frame_sample_rate: int = 2
    transcription_workers: int = 4
    fin_llm_model: str = "claude-haiku-4-5-20251001"
    fin_db_path: Path | None = None  # defaults to db_path.parent / "tiktok_fin.db"
    # Quantum pipeline — all have simulator-safe defaults; see pipeline/quantum/config.py
    quantum_n_qubits: int = 8
    quantum_feature_map: str = "angle"
    quantum_feature_map_reps: int = 2
    quantum_max_train_samples: int = 400
    quantum_vqc_layers: int = 3
    quantum_vqc_epochs: int = 60
    quantum_vqc_learning_rate: float = 0.05
    quantum_vqc_batch_size: int = 32
    quantum_svm_c: float = 1.0
    quantum_seed: int = 42
    quantum_shots: int | None = None  # None = analytic fast path; int = shot-noise simulation


def load_settings(env_file: str | None = None) -> Settings:
    load_dotenv(env_file)

    db_path = Path(os.getenv("DB_PATH", "data/tiktok_signals.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    regions = [
        code.strip().upper()
        for code in os.getenv("REGION_CODES", "US").split(",")
        if code.strip()
    ]

    ticker_universe_env = os.getenv("TICKER_UNIVERSE_PATH")
    fin_db_env = os.getenv("FIN_DB_PATH")

    return Settings(
        client_key=os.getenv("TIKTOK_CLIENT_KEY", ""),
        client_secret=os.getenv("TIKTOK_CLIENT_SECRET", ""),
        access_token=os.getenv("TIKTOK_ACCESS_TOKEN", ""),
        db_path=db_path,
        query_videos_target=int(os.getenv("QUERY_VIDEOS_TARGET", "75")),
        random_low_engagement_target=int(os.getenv("RANDOM_LOW_ENGAGEMENT_TARGET", "25")),
        comments_per_video_target=int(os.getenv("COMMENTS_PER_VIDEO_TARGET", "100")),
        comment_scan_limit=int(os.getenv("COMMENT_SCAN_LIMIT", "300")),
        window_days=int(os.getenv("WINDOW_DAYS", "30")),
        lookback_months=int(os.getenv("LOOKBACK_MONTHS", "24")),
        region_codes=regions,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        brightdata_api_key=os.getenv("BRIGHTDATA_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        ticker_universe_path=Path(ticker_universe_env) if ticker_universe_env else None,
        frame_sample_rate=int(os.getenv("FRAME_SAMPLE_RATE", "2")),
        transcription_workers=int(os.getenv("TRANSCRIPTION_WORKERS", "4")),
        fin_llm_model=os.getenv("FIN_LLM_MODEL", "claude-haiku-4-5-20251001"),
        fin_db_path=Path(fin_db_env) if fin_db_env else None,
        quantum_n_qubits=int(os.getenv("QUANTUM_N_QUBITS", "8")),
        quantum_feature_map=os.getenv("QUANTUM_FEATURE_MAP", "angle"),
        quantum_feature_map_reps=int(os.getenv("QUANTUM_FEATURE_MAP_REPS", "2")),
        quantum_max_train_samples=int(os.getenv("QUANTUM_MAX_TRAIN_SAMPLES", "400")),
        quantum_vqc_layers=int(os.getenv("QUANTUM_VQC_LAYERS", "3")),
        quantum_vqc_epochs=int(os.getenv("QUANTUM_VQC_EPOCHS", "60")),
        quantum_vqc_learning_rate=float(os.getenv("QUANTUM_VQC_LEARNING_RATE", "0.05")),
        quantum_vqc_batch_size=int(os.getenv("QUANTUM_VQC_BATCH_SIZE", "32")),
        quantum_svm_c=float(os.getenv("QUANTUM_SVM_C", "1.0")),
        quantum_seed=int(os.getenv("QUANTUM_SEED", "42")),
        quantum_shots=int(os.getenv("QUANTUM_SHOTS")) if os.getenv("QUANTUM_SHOTS") else None,
    )
