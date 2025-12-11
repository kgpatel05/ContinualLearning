# clx_mvp/features.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class FeatureExtractorConfig:
    """
    Configuration for extracting latent representations.

    Attributes:
        layer: optional dotted-path to a module whose output should be captured via a forward hook.
        flatten: whether to flatten spatial dimensions to [B, D].
        detach: if True, features are detached to avoid gradient tracking.
        normalize: if True, L2-normalize features (useful for prototypicality).
    """
    layer: Optional[str] = None
    flatten: bool = True
    detach: bool = True
    normalize: bool = False


class FeatureExtractor:
    """
    Utility to pull intermediate activations from a model without forcing a bespoke forward API.
    """
    def __init__(self, config: FeatureExtractorConfig = FeatureExtractorConfig()):
        self.config = config

    def __call__(self, model: nn.Module, x: Tensor) -> Tensor:
        return self.extract(model, x)

    def extract(self, model: nn.Module, x: Tensor) -> Tensor:
        cfg = self.config
        if cfg.layer:
            module = self._resolve_layer(model, cfg.layer)
            feats = self._capture_from_layer(module, model, x)
        elif hasattr(model, "forward_features") and callable(getattr(model, "forward_features")):
            feats = model.forward_features(x)  # type: ignore[attr-defined]
        else:
            feats = self._forward_without_head(model, x)

        if cfg.flatten:
            feats = torch.flatten(feats, 1)
        if cfg.normalize:
            feats = torch.nn.functional.normalize(feats, dim=1)
        if cfg.detach:
            feats = feats.detach()
        return feats

    def _resolve_layer(self, model: nn.Module, path: str) -> nn.Module:
        cur = model
        for attr in path.split("."):
            if not hasattr(cur, attr):
                raise AttributeError(f"Model has no attribute '{path}' (missing '{attr}')")
            cur = getattr(cur, attr)
        if not isinstance(cur, nn.Module):
            raise TypeError(f"Resolved path '{path}' is not a nn.Module")
        return cur

    def _capture_from_layer(self, layer: nn.Module, model: nn.Module, x: Tensor) -> Tensor:
        captured = {}

        def hook(_, __, output):
            captured["feat"] = output

        handle = layer.register_forward_hook(hook)
        try:
            _ = model(x)
        finally:
            handle.remove()
        if "feat" not in captured:
            raise RuntimeError("Failed to capture features from specified layer")
        return captured["feat"]

    def _forward_without_head(self, model: nn.Module, x: Tensor) -> Tensor:
        """
        Best-effort feature extraction when no explicit layer is provided:
        - If model has a `.fc` or `.classifier`, run everything except the last layer.
        - Otherwise, fall back to full forward.
        """
        if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
            modules = list(model.children())[:-1]
            body = nn.Sequential(*modules)
            feats = body(x)
        elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            modules = list(model.children())[:-1]
            body = nn.Sequential(*modules)
            feats = body(x)
        else:
            feats = model(x)
        return feats
