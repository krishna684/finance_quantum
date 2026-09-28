from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.collector import run_collection
from pipeline.config import load_settings
from pipeline.logger import configure_log_level
from pipeline.mapping import load_mapping_csv
from pipeline.news.ingester import ingest_directory, ingest_file
from pipeline.news.storage import NewsStorage
from pipeline.storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description="TikTok demand signal collection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # Existing commands
    # ------------------------------------------------------------------

    collect_cmd = sub.add_parser("collect", help="Run TikTok video/comment collection")
    collect_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")

    mapping_cmd = sub.add_parser("load-mapping", help="Load brand-to-firm mapping CSV into SQLite")
    mapping_cmd.add_argument("--csv", required=True, help="Path to mapping CSV")
    mapping_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")

    news_cmd = sub.add_parser("ingest-news", help="Ingest WSJ/NYT articles from academic DB exports")
    news_cmd.add_argument("--source", required=True, choices=["WSJ", "NYT"], help="Media source label")
    news_cmd.add_argument("--file", default=None, help="Path to a single export file")
    news_cmd.add_argument("--dir", default=None, help="Directory of export files to ingest recursively")
    news_cmd.add_argument("--format", default=None, choices=["proquest", "factiva", "csv"],
                          dest="fmt", help="Export format (auto-detected if omitted)")
    news_cmd.add_argument("--db", default="data/news.db", help="SQLite output path (default: data/news.db)")
    news_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")

    # ------------------------------------------------------------------
    # Phase 1: Image embeddings
    # ------------------------------------------------------------------

    emb_cmd = sub.add_parser(
        "extract-embeddings",
        help="Extract CLIP ViT-B/32 image embeddings from news images",
    )
    emb_cmd.add_argument("--db", default="data/news.db", help="news.db path (default: data/news.db)")
    emb_cmd.add_argument("--batch-size", type=int, default=32, help="Images per GPU batch (default: 32)")

    # ------------------------------------------------------------------
    # Phase 2: Text embeddings
    # ------------------------------------------------------------------

    txt_emb_cmd = sub.add_parser(
        "extract-text-embeddings",
        help="Extract FinBERT/MiniLM text embeddings from article headlines and bodies",
    )
    txt_emb_cmd.add_argument("--db", default="data/news.db", help="news.db path (default: data/news.db)")
    txt_emb_cmd.add_argument("--batch-size", type=int, default=64, help="Articles per batch (default: 64)")
    txt_emb_cmd.add_argument(
        "--no-finbert",
        action="store_true",
        help="Skip FinBERT and use all-MiniLM-L6-v2 directly",
    )

    # ------------------------------------------------------------------
    # Phase 3: Classification
    # ------------------------------------------------------------------

    clf_cmd = sub.add_parser(
        "classify",
        help="Run walk-forward classification on image/text/multimodal embeddings",
    )
    clf_cmd.add_argument("--db", default="data/news.db", help="news.db path (default: data/news.db)")
    clf_cmd.add_argument(
        "--crsp-returns", default="data/crsp_dsf.csv", help="CRSP DSF CSV (default: data/crsp_dsf.csv)"
    )
    clf_cmd.add_argument(
        "--crsp-ids", default="data/crsp_identifiers.csv",
        help="CRSP identifier CSV (default: data/crsp_identifiers.csv)",
    )
    clf_cmd.add_argument(
        "--embedding-type",
        default="image",
        choices=["image", "text", "multimodal"],
        help="Embedding type to use (default: image)",
    )
    clf_cmd.add_argument("--results-dir", default="results", help="Output directory (default: results)")
    clf_cmd.add_argument(
        "--min-train-days", type=int, default=252,
        help="Minimum unique trading days before first test fold (default: 252 ≈ 1 trading year)",
    )
    clf_cmd.add_argument(
        "--test-days", type=int, default=21,
        help="Trading days per test fold (default: 21 ≈ 1 month)",
    )
    clf_cmd.add_argument("--min-date", default="1800-01-01", help="Earliest article date (default: 1800-01-01)")
    clf_cmd.add_argument("--max-date", default="2099-12-31", help="Latest article date (default: 2099-12-31)")

    # ------------------------------------------------------------------
    # Phase 4: Portfolio construction
    # ------------------------------------------------------------------

    port_cmd = sub.add_parser(
        "build-portfolio",
        help="Build quintile long-short portfolios from classifier predictions",
    )
    port_cmd.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    port_cmd.add_argument(
        "--embedding-type",
        default="image",
        choices=["image", "text", "multimodal"],
        help="Embedding type to analyse (default: image)",
    )
    port_cmd.add_argument(
        "--ff-factors",
        default=None,
        help="Path to local Fama-French factors CSV (downloaded from Kenneth French library if omitted)",
    )

    # ------------------------------------------------------------------
    # Financial TikTok collection (Bright Data architecture)
    # ------------------------------------------------------------------

    fin_collect_cmd = sub.add_parser(
        "fin-collect",
        help="Run financial TikTok collection via Bright Data + LLM filtering",
    )
    fin_collect_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")
    fin_collect_cmd.add_argument(
        "--mock",
        action="store_true",
        help="Use MockBrightDataClient (no Bright Data credentials needed)",
    )
    fin_collect_cmd.add_argument(
        "--fixture",
        default=None,
        help="Path to fixture JSON file for mock client",
    )
    fin_collect_cmd.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip CLIP frame and FinBERT transcript embedding stages",
    )

    # ------------------------------------------------------------------
    # Quantum benchmark (demo_quantum.py logic, wired to Settings)
    # ------------------------------------------------------------------

    qbench_cmd = sub.add_parser(
        "quantum-bench",
        help="Benchmark quantum vs classical classifiers on synthetic 512-dim embeddings",
    )
    qbench_cmd.add_argument("--env-file", default=None, help="Optional path to .env file")
    qbench_cmd.add_argument(
        "--n-per-class", type=int, default=120,
        help="Synthetic samples per class (default: 120)",
    )
    qbench_cmd.add_argument(
        "--sep", type=float, default=3.0,
        help="Class separation in signal direction (default: 3.0; try 1.6 for weak-SNR analysis)",
    )
    qbench_cmd.add_argument(
        "--dim", type=int, default=512,
        help="Embedding dimension of synthetic data (default: 512, matching CLIP ViT-B/32)",
    )
    qbench_cmd.add_argument(
        "--seed", type=int, default=42,
        help="Data-generation RNG seed (default: 42)",
    )

    # ------------------------------------------------------------------
    # Phase 5: Report
    # ------------------------------------------------------------------

    report_cmd = sub.add_parser(
        "report",
        help="Generate Markdown summary report from all results",
    )
    report_cmd.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    report_cmd.add_argument("--output", default=None, help="Output path (default: results/report.md)")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    args = parser.parse_args()

    if args.command == "fin-collect":
        from pathlib import Path as _Path
        from pipeline.tiktok_collection.pipeline import run_fin_collection
        from pipeline.tiktok_collection.collector import MockBrightDataClient

        settings = load_settings(args.env_file)
        configure_log_level(settings.log_level)

        if args.mock or not settings.brightdata_api_key:
            # Monkey-patch the factory so pipeline uses mock client
            import pipeline.tiktok_collection.collector as _col
            fixture = _Path(args.fixture) if args.fixture else None
            _col.make_client = lambda key, fixture_path=None: MockBrightDataClient(fixture_path=fixture)

        run_fin_collection(settings)
        return

    if args.command == "collect":
        settings = load_settings(args.env_file)
        configure_log_level(settings.log_level)
        if not settings.access_token and (not settings.client_key or not settings.client_secret):
            raise SystemExit(
                "Set TIKTOK_ACCESS_TOKEN or both TIKTOK_CLIENT_KEY/TIKTOK_CLIENT_SECRET before running collect."
            )
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
        return

    if args.command == "ingest-news":
        db_path = Path(args.db)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        ns = NewsStorage(db_path)
        ns.init_schema()
        try:
            if args.file:
                count = ingest_file(Path(args.file), ns, source_hint=args.source, fmt=args.fmt)
            elif args.dir:
                count = ingest_directory(Path(args.dir), ns, source_hint=args.source, fmt=args.fmt)
            else:
                raise SystemExit("Provide --file or --dir")
        finally:
            ns.close()
        print(f"Ingested {count} articles into {db_path}")
        return

    if args.command == "extract-embeddings":
        from pipeline.embeddings import ImageEmbedder

        db_path = Path(args.db)
        embedder = ImageEmbedder(db_path)
        embedder.init_schema()
        try:
            n = embedder.run(batch_size=args.batch_size)
        finally:
            embedder.close()
        print(f"Embedded {n} images in {db_path}")
        return

    if args.command == "extract-text-embeddings":
        from pipeline.text_embeddings import TextEmbedder

        db_path = Path(args.db)
        embedder = TextEmbedder(db_path, prefer_finbert=not args.no_finbert)
        embedder.init_schema()
        try:
            n = embedder.run(batch_size=args.batch_size)
        finally:
            embedder.close()
        print(f"Embedded {n} articles in {db_path}")
        return

    if args.command == "classify":
        from pipeline.classifiers import ClassificationPipeline

        pipe = ClassificationPipeline(
            news_db=Path(args.db),
            crsp_csv=Path(args.crsp_returns),
            crsp_ids_csv=Path(args.crsp_ids),
            results_dir=Path(args.results_dir),
            min_train_days=args.min_train_days,
            test_days=args.test_days,
        )
        results = pipe.run(
            embedding_type=args.embedding_type,
            min_date=args.min_date,
            max_date=args.max_date,
        )
        print(results.summary())
        return

    if args.command == "build-portfolio":
        from pipeline.portfolio import PortfolioAnalyzer

        ff_path = Path(args.ff_factors) if args.ff_factors else None
        analyzer = PortfolioAnalyzer(
            results_dir=Path(args.results_dir),
            ff_factors_csv=ff_path,
        )
        out = analyzer.run(embedding_type=args.embedding_type)
        n_models = len(out.get("summary_stats", []))
        print(f"Portfolio analysis complete: {n_models} model(s) evaluated.")
        return

    if args.command == "quantum-bench":
        import time

        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.svm import SVC

        try:
            from pipeline.quantum import (
                EmbeddingReducer,
                QuantumKernel,
                QuantumKernelSVM,
                QuantumSettings,
                VariationalQuantumClassifier,
            )
        except ImportError as exc:
            raise SystemExit(
                f"PennyLane is required for quantum-bench. "
                f"Install it with: pip install 'pennylane>=0.38.0,<0.40'\n({exc})"
            ) from exc

        settings = load_settings(args.env_file)
        qs = QuantumSettings(
            n_qubits=settings.quantum_n_qubits,
            feature_map=settings.quantum_feature_map,
            feature_map_reps=settings.quantum_feature_map_reps,
            max_train_samples=settings.quantum_max_train_samples,
            vqc_layers=settings.quantum_vqc_layers,
            vqc_epochs=settings.quantum_vqc_epochs,
            vqc_learning_rate=settings.quantum_vqc_learning_rate,
            vqc_batch_size=settings.quantum_vqc_batch_size,
            svm_C=settings.quantum_svm_c,
            seed=settings.quantum_seed,
            shots=settings.quantum_shots,
        )

        def _make_data(n_per_class, dim, sep, seed):
            rng = np.random.default_rng(seed)
            direction = rng.normal(size=dim)
            direction /= np.linalg.norm(direction)
            X0 = rng.normal(0, 1, (n_per_class, dim))
            X1 = rng.normal(0, 1, (n_per_class, dim)) + sep * direction
            X = np.vstack([X0, X1])
            y = np.array([0] * n_per_class + [1] * n_per_class)
            perm = rng.permutation(len(y))
            return X[perm], y[perm]

        def _evaluate(name, y_true, y_pred, y_score, elapsed):
            print(
                f"{name:<42} acc={accuracy_score(y_true, y_pred):.3f}  "
                f"f1={f1_score(y_true, y_pred):.3f}  "
                f"auc={roc_auc_score(y_true, y_score):.3f}  "
                f"({elapsed:.1f}s)"
            )

        X, y = _make_data(args.n_per_class, args.dim, args.sep, args.seed)
        split = int(len(y) * 0.8)
        X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
        budget = QuantumKernel.evaluation_budget(
            min(len(y_tr), qs.max_train_samples), n_test=len(y_te)
        )
        print(f"\ntrain={len(y_tr)}  test={len(y_te)}  dim={X.shape[1]}  sep={args.sep}")
        print(f"quantum kernel circuit-evaluation budget: {budget:,}")
        print(f"quantum settings: {qs.n_qubits} qubits, {qs.feature_map} map, "
              f"reps={qs.feature_map_reps}, shots={qs.shots}\n")

        # 1. Classical SVM on full embeddings
        t0 = time.time()
        clf = SVC(kernel="rbf", probability=True, random_state=42).fit(X_tr, y_tr)
        _evaluate(f"SVM-RBF ({args.dim}-dim, classical)", y_te,
                  clf.predict(X_te), clf.predict_proba(X_te)[:, 1], time.time() - t0)

        # 2. Classical SVM on same reduced features
        t0 = time.time()
        red = EmbeddingReducer(n_components=qs.n_qubits, seed=qs.seed)
        Ztr, Zte = red.fit_transform(X_tr), red.transform(X_te)
        clf = SVC(kernel="rbf", probability=True, random_state=42).fit(Ztr, y_tr)
        _evaluate(f"SVM-RBF ({qs.n_qubits}-dim PCA, classical)", y_te,
                  clf.predict(Zte), clf.predict_proba(Zte)[:, 1], time.time() - t0)
        print(f"  PCA explained variance ({qs.n_qubits} comps): "
              f"{red.explained_variance_ratio.sum():.1%}")

        # 3. Quantum kernel SVM
        t0 = time.time()
        qsvm = QuantumKernelSVM(qs).fit(X_tr, y_tr)
        _evaluate(f"Quantum kernel SVM ({qs.n_qubits} qubits)", y_te,
                  qsvm.predict(X_te), qsvm.predict_proba(X_te)[:, 1], time.time() - t0)

        # 4. VQC
        t0 = time.time()
        vqc = VariationalQuantumClassifier(qs).fit(X_tr, y_tr)
        _evaluate(f"VQC ({qs.n_qubits} qubits, {qs.vqc_layers} layers)", y_te,
                  vqc.predict(X_te), vqc.predict_proba(X_te)[:, 1], time.time() - t0)
        print(f"  VQC loss: {vqc.loss_history_[0]:.3f} -> {vqc.loss_history_[-1]:.3f}")
        return

    if args.command == "report":
        from pipeline.report import ReportGenerator

        gen = ReportGenerator(results_dir=Path(args.results_dir))
        out_path = Path(args.output) if args.output else None
        written = gen.run(output_path=out_path)
        print(f"Report written to {written}")
        return


if __name__ == "__main__":
    main()
