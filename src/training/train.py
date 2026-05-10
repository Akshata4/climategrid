"""
Training loop for ClimateGridModel.

Only the risk head is trained; the encoder is frozen (see climategrid.py).
Checkpoints the best model (lowest val loss) to artifacts/best_model.pt.
"""

import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(config: dict) -> Dict[str, float]:
    """
    Train the ClimateGridModel.

    Args:
        config: full config dict loaded from config.yaml

    Returns:
        dict with best_val_loss and final_train_loss
    """
    from src.data.dataset import build_datasets
    from src.model.climategrid import ClimateGridModel
    from src.training.evaluate import evaluate

    _set_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logger.info("Train samples: %d  |  Val samples: %d", len(train_ds), len(val_ds))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ClimateGridModel(config).to(device)

    # Only optimise the risk head (encoder params have requires_grad=False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )
    criterion = nn.MSELoss()

    artifact_dir = Path(config["paths"]["artifacts"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(config["paths"]["checkpoint"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    final_train_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels   = labels.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        final_train_loss = train_loss

        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["mse"]

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                epoch, config["training"]["epochs"], train_loss, val_loss,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info("  ✓ New best model saved (val_loss=%.4f)", best_val_loss)

    logger.info(
        "Training complete. Best val_loss=%.4f  Checkpoint: %s",
        best_val_loss, ckpt_path,
    )
    return {"best_val_loss": best_val_loss, "final_train_loss": final_train_loss}
