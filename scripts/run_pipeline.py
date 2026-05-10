"""
End-to-end pipeline runner for ClimateGrid.

Runs: download → preprocess → train

Usage:
    uv run python scripts/run_pipeline.py              # uses config.yaml settings
    uv run python scripts/run_pipeline.py --synthetic  # force synthetic data (no HF download)
    uv run python scripts/run_pipeline.py --epochs 5   # override epoch count (quick smoke test)
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Ensure the project root is on sys.path so `src.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ClimateGrid pipeline runner")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of HuggingFace download")
    parser.add_argument("--epochs",   type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step (use already-downloaded data)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    args = parser.parse_args()

    config = load_config()

    if args.synthetic:
        config["dataset"]["use_synthetic"] = True
        logger.info("Synthetic data mode enabled.")

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info("Epochs overridden to %d.", args.epochs)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        logger.info("=" * 60)
        logger.info("STEP 1 / 3  —  Data download")
        logger.info("=" * 60)
        from src.data.download import download_climate_data
        download_climate_data(config)
    else:
        logger.info("Skipping download (--skip-download).")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        logger.info("=" * 60)
        logger.info("STEP 2 / 3  —  Preprocessing + label generation")
        logger.info("=" * 60)
        from src.data.preprocess import run_preprocessing
        run_preprocessing(config)
    else:
        logger.info("Skipping preprocessing (--skip-preprocess).")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 3  —  Training")
    logger.info("=" * 60)
    from src.training.train import run_training
    metrics = run_training(config)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("  best_val_loss  = %.4f", metrics["best_val_loss"])
    logger.info("  final_train_loss = %.4f", metrics["final_train_loss"])
    logger.info("  Checkpoint at: %s", config["paths"]["checkpoint"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
