from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.collector import run_collection
from pipeline.config import load_settings
from pipeline.mapping import load_mapping_csv
from pipeline.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok demand signal collection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_cmd = sub.add_parser("collect", help="Run TikTok video/comment collection")
    collect_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")

    mapping_cmd = sub.add_parser("load-mapping", help="Load brand-to-firm mapping CSV into SQLite")
    mapping_cmd.add_argument("--csv", required=True, help="Path to mapping CSV")
    mapping_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")

    args = parser.parse_args()

    if args.command == "collect":
        settings = load_settings(args.env_file)
        if not settings.access_token and (not settings.client_key or not settings.client_secret):
            raise SystemExit("Set TIKTOK_ACCESS_TOKEN or both TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET before running collect.")
        run_collection(settings)
        return

    if args.command == "load-mapping":
        settings = load_settings(args.env_file)
        storage = Storage(settings.db_path)
        storage.init_schema()
        try:
            rows = load_mapping_csv(Path(args.csv), storage)
        finally:
            storage.close()
        print(f"Loaded {rows} mapping rows into {settings.db_path}")


if __name__ == "__main__":
    main()
