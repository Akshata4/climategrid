"""
Evaluation utilities.

evaluate()         — single-pass val metrics (MSE, MAE per risk type)
evaluate_by_city() — per-city breakdown (requires city labels in the dataset)
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.labels import RISK_NAMES

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Return per-risk MAE and overall MSE on the given DataLoader.
    """
    model.eval()
    total_mse   = 0.0
    total_mae   = torch.zeros(4, device=device)
    n_batches   = 0

    for features, labels in loader:
        features = features.to(device)
        labels   = labels.to(device)
        preds    = model(features)

        total_mse += criterion(preds, labels).item()
        total_mae += (preds - labels).abs().mean(dim=0)
        n_batches += 1

    metrics: Dict[str, float] = {"mse": total_mse / max(n_batches, 1)}
    mae = (total_mae / max(n_batches, 1)).cpu().tolist()
    for name, value in zip(RISK_NAMES, mae):
        metrics[f"mae_{name}"] = value

    return metrics


def log_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    logger.info("%sMSE: %.4f", label, metrics["mse"])
    for name in RISK_NAMES:
        logger.info("%s  MAE %s: %.4f", label, name, metrics.get(f"mae_{name}", float("nan")))
