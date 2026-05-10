"""Tests for model components."""

import pytest
import torch
import yaml
from pathlib import Path


@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_config(config):
    """Reduced config so tests run quickly on CPU without ClimaX weights."""
    config["model"]["encoder"]   = "cnn"
    config["model"]["embed_dim"] = 64
    config["model"]["risk_head_hidden"] = [32]
    config["dataset"]["patch_size"] = 5
    return config


def test_cnn_encoder_shape(small_config):
    from src.model.cnn_encoder import CNNEncoder
    enc = CNNEncoder(
        variables=small_config["variables"],
        patch_size=small_config["dataset"]["patch_size"],
        embed_dim=small_config["model"]["embed_dim"],
    )
    B, V, H, W = 4, len(small_config["variables"]), 5, 5
    x = torch.randn(B, V, H, W)
    out = enc(x)
    assert out.shape == (B, small_config["model"]["embed_dim"])


def test_climax_encoder_shape(small_config):
    small_config["model"]["embed_dim"] = 64
    small_config["model"]["depth"]     = 2
    small_config["model"]["num_heads"] = 4
    from src.model.climax_encoder import ClimaXEncoder
    enc = ClimaXEncoder(
        variables=small_config["variables"],
        patch_size=small_config["dataset"]["patch_size"],
        embed_dim=small_config["model"]["embed_dim"],
        depth=small_config["model"]["depth"],
        num_heads=small_config["model"]["num_heads"],
        pretrained_model_id=None,  # no HF download in tests
    )
    B, V, H, W = 4, len(small_config["variables"]), 5, 5
    x = torch.randn(B, V, H, W)
    out = enc(x)
    assert out.shape == (B, small_config["model"]["embed_dim"])


def test_risk_head_shape(small_config):
    from src.model.risk_head import RiskHead
    head = RiskHead(
        embed_dim=small_config["model"]["embed_dim"],
        hidden_dims=small_config["model"]["risk_head_hidden"],
    )
    B = 8
    emb = torch.randn(B, small_config["model"]["embed_dim"])
    out = head(emb)
    assert out.shape == (B, 4)
    assert torch.all((out >= 0) & (out <= 1)), "Scores outside [0,1]"


def test_full_model_forward(small_config):
    from src.model.climategrid import ClimateGridModel
    model = ClimateGridModel(small_config)
    B, V = 4, len(small_config["variables"])
    H = W = small_config["dataset"]["patch_size"]
    x = torch.randn(B, V, H, W)
    out = model(x)
    assert out.shape == (B, 4)
    assert torch.all((out >= 0) & (out <= 1))


def test_encoder_frozen(small_config):
    from src.model.climategrid import ClimateGridModel
    model = ClimateGridModel(small_config)
    for name, param in model.encoder.named_parameters():
        assert not param.requires_grad, f"Encoder param '{name}' should be frozen"
    for name, param in model.risk_head.named_parameters():
        assert param.requires_grad, f"Risk head param '{name}' should be trainable"
