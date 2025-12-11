# clx_mvp/grasp.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import CLStrategy
from ..replay import RichERBuffer
from ..features import FeatureExtractor, FeatureExtractorConfig
from ..configs import LinearScheduleConfig, linear_schedule


@dataclass
class GraspConfig:
    """
    Configuration for GRASP rehearsal policy (curriculum over prototypicality).

    Attributes:
        replay_ratio: fraction of each batch to source from replay.
        distance_metric: "l2" or "cosine" to score prototypicality.
        prototype_momentum: EMA coefficient for class prototypes.
        difficulty_schedule: linear schedule of allowable percentile of samples (easy -> hard).
        min_samples_per_class: safety floor when sampling per class.
        recompute_features_every: if >0, recompute buffer feature distances every N steps.
        feature_extractor: configuration for feature capture.
        log_difficulty: keep last sampled difficulty scores for logging.
    """
    replay_ratio: float = 0.5
    distance_metric: str = "l2"
    prototype_momentum: float = 0.9
    difficulty_schedule: LinearScheduleConfig = field(
        default_factory=lambda: LinearScheduleConfig(start=0.3, end=1.0, total_steps=1000)
    )
    min_samples_per_class: int = 1
    recompute_features_every: int = 0
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    log_difficulty: bool = True


class GraspRehearsalPolicy:
    """
    Maintains class prototypes and selects replay examples from easy (prototypical)
    to harder samples as training progresses.

    Paper: "GRASP: A Rehearsal Policy for Efficient Online Continual Learning" (NeurIPS 2023).
    """
    def __init__(self, cfg: GraspConfig):
        self.cfg = cfg
        self.extractor = FeatureExtractor(cfg.feature_extractor)
        self.prototypes: Dict[int, Tensor] = {}
        self.global_step: int = 0
        self._last_sampled_scores: List[float] = []

    def process_batch(self, model, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute features and per-sample prototypicality scores for the incoming batch.
        Returns:
            scores: difficulty (distance) for each sample
            feats: detached features for storage
        """
        feats = self.extractor(model, x)
        scores = self._compute_scores(feats, y)
        self._update_prototypes(feats, y)
        return scores.detach(), feats.detach().cpu()

    def sample(self, buffer: RichERBuffer, k: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if len(buffer) == 0 or k <= 0:
            return None, None

        if self.cfg.recompute_features_every > 0 and self.global_step % self.cfg.recompute_features_every == 0:
            self._refresh_scores(buffer)

        allowed = self._allowed_fraction()
        entries = buffer._entries  # internal access by design for policy control
        by_cls: Dict[int, List[Tuple[int, float]]] = {}
        for idx, ent in enumerate(entries):
            by_cls.setdefault(ent["class"], []).append((idx, float(ent["score"])))

        candidate_indices: List[int] = []
        for cls, pairs in by_cls.items():
            # sort by difficulty (lower = easier/prototypical)
            pairs_sorted = sorted(pairs, key=lambda p: p[1])
            cutoff = max(self.cfg.min_samples_per_class, int(len(pairs_sorted) * allowed))
            cutoff = min(cutoff, len(pairs_sorted))
            candidate_indices.extend([idx for idx, _ in pairs_sorted[:cutoff]])

        if not candidate_indices:
            # fallback to uniform sampling
            candidate_indices = random.sample(range(len(buffer)), min(k, len(buffer)))

        chosen = random.sample(candidate_indices, min(k, len(candidate_indices)))
        xs = torch.stack([entries[i]["x"] for i in chosen], dim=0)
        ys = torch.stack([entries[i]["y"] for i in chosen], dim=0).long()

        if self.cfg.log_difficulty:
            self._last_sampled_scores = [float(entries[i]["score"]) for i in chosen]
        self.global_step += 1
        return xs, ys

    # ---- internal ----
    def _compute_scores(self, feats: Tensor, labels: Tensor) -> Tensor:
        scores = torch.zeros(feats.size(0), device=feats.device)
        for cls in labels.unique():
            mask = labels == cls
            proto = self.prototypes.get(int(cls.item()))
            cls_feats = feats[mask]
            if proto is None:
                scores[mask] = 0.0
                continue
            if self.cfg.distance_metric == "cosine":
                proto_norm = F.normalize(proto, dim=0)
                feat_norm = F.normalize(cls_feats, dim=1)
                # smaller distance = more prototypical
                dist = 1.0 - torch.matmul(feat_norm, proto_norm)
            else:
                dist = torch.linalg.norm(cls_feats - proto, dim=1)
            scores[mask] = dist
        return scores

    def _update_prototypes(self, feats: Tensor, labels: Tensor) -> None:
        momentum = self.cfg.prototype_momentum
        for cls in labels.unique():
            mask = labels == cls
            cls_feats = feats[mask]
            if cls_feats.numel() == 0:
                continue
            mean_feat = cls_feats.mean(dim=0)
            if int(cls.item()) not in self.prototypes:
                self.prototypes[int(cls.item())] = mean_feat.detach()
            else:
                prev = self.prototypes[int(cls.item())]
                self.prototypes[int(cls.item())] = momentum * prev + (1 - momentum) * mean_feat.detach()

    def _allowed_fraction(self) -> float:
        cfg = self.cfg.difficulty_schedule
        return linear_schedule(self.global_step, cfg)

    def _refresh_scores(self, buffer: RichERBuffer) -> None:
        """
        Optionally recompute difficulty scores for buffer entries using stored features.
        """
        entries = buffer._entries
        for ent in entries:
            feat = ent.get("feat")
            cls = ent.get("class")
            if feat is None or cls is None:
                continue
            proto = self.prototypes.get(int(cls))
            if proto is None:
                continue
            feat_t = feat.to(proto.device)
            if self.cfg.distance_metric == "cosine":
                feat_norm = F.normalize(feat_t, dim=0)
                proto_norm = F.normalize(proto, dim=0)
                dist = 1.0 - torch.dot(feat_norm, proto_norm)
            else:
                dist = torch.linalg.norm(feat_t - proto)
            ent["score"] = float(dist.detach().cpu().item())

    @property
    def last_sampled_scores(self) -> List[float]:
        return self._last_sampled_scores


class GraspStrategy(CLStrategy):
    """
    Strategy wrapper that plugs GRASP sampling into the standard replay hook.
    Requires the learner buffer to be a RichERBuffer so we can persist scores/features.
    """
    def __init__(self, cfg: GraspConfig, base_strategy: Optional[CLStrategy] = None):
        self.cfg = cfg
        self.policy = GraspRehearsalPolicy(cfg)
        self.base = base_strategy

    def _buffer(self, learner) -> RichERBuffer:
        if not isinstance(learner.buffer, RichERBuffer):
            raise ValueError("GraspStrategy requires learner.buffer to be a RichERBuffer.")
        return learner.buffer

    def before_experience(self, learner, exp) -> None:
        if self.base:
            self.base.before_experience(learner, exp)

    def before_batch(self, learner, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        buffer = self._buffer(learner)
        if self.base:
            x, y = self.base.before_batch(learner, x, y)

        scores, feats = self.policy.process_batch(learner.model, x, y)
        learner._current_batch_for_buffer = (x.detach(), y.detach(), scores.detach().cpu(), feats)

        r_k = int(self.cfg.replay_ratio * x.size(0))
        rx, ry = self.policy.sample(buffer, r_k)
        if rx is not None:
            rx = rx.to(x.device, non_blocking=True)
            ry = ry.to(y.device, non_blocking=True)
            x = torch.cat([x, rx], dim=0)
            y = torch.cat([y, ry], dim=0)
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        if self.base:
            return self.base.loss(learner, logits, y)
        return learner.criterion(logits, y)

    def after_experience(self, learner, exp) -> None:
        if self.base:
            self.base.after_experience(learner, exp)
