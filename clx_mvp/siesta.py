# clx_mvp/siesta.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterable
import time

import torch
import torch.nn as nn
from torch import Tensor

from .strategies import CLStrategy
from .replay import LatentReplayBuffer
from .compression import LatentCompressionConfig, build_compressor, LatentCompressor
from .features import FeatureExtractor, FeatureExtractorConfig
from .configs import OptimizerConfig


@dataclass
class SleepScheduleConfig:
    """
    Controls how often SIESTA transitions from wake to sleep.

    Attributes:
        sleep_every_batches: trigger a sleep phase after this many wake updates.
        sleep_on_task_end: if True, run a sleep phase at the end of every experience.
        min_buffer_fraction: minimum fraction of buffer filled before allowing sleep.
    """
    sleep_every_batches: int = 200
    sleep_on_task_end: bool = True
    min_buffer_fraction: float = 0.2


@dataclass
class SiestaConfig:
    """
    Config for SIESTA (Efficient Online Continual Learning with Sleep).

    Defaults prioritize lightweight wake updates and bounded sleep consolidation.
    """
    head_attr: str = "fc"
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    compression: LatentCompressionConfig = field(default_factory=lambda: LatentCompressionConfig(method="float16"))
    buffer_size: int = 4000
    buffer_replacement: str = "reservoir"

    # Wake phase
    wake_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=0.05, weight_decay=0.0, momentum=0.9))
    wake_steps: int = 1
    wake_freeze_backbone: bool = True
    wake_extra_unfrozen: Optional[List[str]] = None
    store_raw_in_er: bool = False  # keep raw samples in learner.buffer alongside latent buffer

    # Sleep phase
    sleep_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=0.03, weight_decay=5e-4, momentum=0.9))
    sleep_steps: int = 64
    sleep_batch_size: int = 128
    train_backbone_in_sleep: bool = False  # by default, only head/tail after extraction point

    schedule: SleepScheduleConfig = field(default_factory=SleepScheduleConfig)
    log_internal: bool = False


class SiestaStrategy(CLStrategy):
    """
    SIESTA strategy implementing wake/sleep alternation with latent replay.

    Paper: "SIESTA: Efficient Online Continual Learning with Sleep" (CVPR 2023).
    """

    handles_optimization = True  # Learner delegates backward/step to this strategy

    def __init__(
        self,
        config: SiestaConfig,
        *,
        latent_buffer: Optional[LatentReplayBuffer] = None,
        compressor: Optional[LatentCompressor] = None,
    ):
        self.cfg = config
        self.latent_buffer = latent_buffer or LatentReplayBuffer(
            capacity=config.buffer_size,
            replacement=config.buffer_replacement,
        )
        self.compressor = compressor or build_compressor(config.compression)
        self.extractor = FeatureExtractor(config.feature_extractor)

        self._wake_opt = None
        self._sleep_opt = None
        self._global_wake_steps = 0
        self._wake_since_sleep = 0
        self._current_exp_id: Optional[int] = None
        self._sleep_log: List[Dict[str, Any]] = []
        self._base_requires_grad: Dict[str, bool] = {}

    # ---- Hooks ----
    def before_experience(self, learner, exp) -> None:
        self._current_exp_id = getattr(exp, "exp_id", None)
        self._cache_requires_grad(learner)
        self._apply_wake_freeze(learner)
        self._wake_opt = self._build_optimizer(
            learner,
            params=self._trainable_params_for_wake(learner),
            opt_cfg=self.cfg.wake_optimizer,
        )
        self._sleep_opt = None  # lazily built on first sleep
        self._wake_since_sleep = 0

    def before_batch(self, learner, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        # Wake phase: store latent representations; skip raw buffer unless requested.
        learner._skip_buffer_addition = not self.cfg.store_raw_in_er

        with torch.no_grad():
            feats = self.extractor(learner.model, x)
            encoded, aux = self.compressor.compress(feats)
            meta = {
                "exp_id": self._current_exp_id,
                "step": self._global_wake_steps,
            }
            self.latent_buffer.add(encoded, y.detach().cpu(), aux=aux, metadata=meta)
        return x, y

    def loss(self, learner, logits: Tensor, y: Tensor) -> Tensor:
        # Wake loss defaults to CE on the fast head.
        return learner.criterion(logits, y)

    def optimize_batch(self, learner, loss: Tensor, logits: Tensor, y: Tensor) -> None:
        if self._wake_opt is None:
            raise RuntimeError("Wake optimizer not initialized; ensure before_experience was called.")

        # Wake: fast, lightweight updates
        for _ in range(max(1, int(self.cfg.wake_steps))):
            self._wake_opt.zero_grad(set_to_none=True)
            learner.fabric.backward(loss)
            self._wake_opt.step()

        self._global_wake_steps += 1
        self._wake_since_sleep += 1

        if self._should_sleep():
            self._run_sleep_phase(learner, reason="scheduled")

    def after_experience(self, learner, exp) -> None:
        if self.cfg.schedule.sleep_on_task_end:
            self._run_sleep_phase(learner, reason="task_end")
        # Restore wake-time freezing
        self._apply_wake_freeze(learner)

    def after_batch(self, learner, x: Tensor, y: Tensor, loss: Optional[Tensor] = None) -> None:
        # no-op; hook kept for symmetry/logging if needed in future
        return

    # ---- Sleep / wake helpers ----
    def _should_sleep(self) -> bool:
        sched = self.cfg.schedule
        if sched.sleep_every_batches <= 0:
            return False
        if self._wake_since_sleep < sched.sleep_every_batches:
            return False
        buffer_ready = len(self.latent_buffer) >= int(sched.min_buffer_fraction * self.latent_buffer.capacity)
        return buffer_ready

    def _run_sleep_phase(self, learner, reason: str = "scheduled") -> None:
        if len(self.latent_buffer) == 0:
            return

        self._apply_sleep_trainable(learner)
        if self._sleep_opt is None:
            params = self._trainable_params_for_sleep(learner)
            self._sleep_opt = self._build_optimizer(learner, params=params, opt_cfg=self.cfg.sleep_optimizer)

        sleep_steps = int(self.cfg.sleep_steps)
        batch_size = int(self.cfg.sleep_batch_size)
        t0 = time.perf_counter()
        sleep_updates = 0

        for _ in range(sleep_steps):
            latents, labels, aux = self.latent_buffer.sample(batch_size)
            if latents is None or labels is None:
                break
            device = next(learner.model.parameters()).device
            latents = self.compressor.decompress(latents, aux, device=device)
            labels = labels.to(device)

            logits = self._forward_from_latent(learner.model, latents)
            loss = learner.criterion(logits, labels)

            self._sleep_opt.zero_grad(set_to_none=True)
            learner.fabric.backward(loss)
            self._sleep_opt.step()
            sleep_updates += 1

        t1 = time.perf_counter()
        self._wake_since_sleep = 0
        if self.cfg.log_internal:
            self._sleep_log.append(
                {
                    "reason": reason,
                    "sleep_updates": sleep_updates,
                    "sleep_time_sec": t1 - t0,
                    "buffer_size": len(self.latent_buffer),
                }
            )
        # Re-freeze for wake
        self._apply_wake_freeze(learner)

    # ---- Parameter helpers ----
    def _cache_requires_grad(self, learner) -> None:
        if not self._base_requires_grad:
            self._base_requires_grad = {name: p.requires_grad for name, p in learner.model.named_parameters()}

    def _apply_wake_freeze(self, learner) -> None:
        if not self.cfg.wake_freeze_backbone:
            return
        allow_extra = self.cfg.wake_extra_unfrozen or []
        head_params = set(self._head_param_names(learner.model))
        for name, p in learner.model.named_parameters():
            allow = (name in head_params) or any(tag in name for tag in allow_extra)
            p.requires_grad = allow

    def _apply_sleep_trainable(self, learner) -> None:
        train_backbone = self.cfg.train_backbone_in_sleep
        if train_backbone:
            # restore defaults
            for name, p in learner.model.named_parameters():
                p.requires_grad = self._base_requires_grad.get(name, True)
        else:
            # head-only
            head_names = set(self._head_param_names(learner.model))
            for name, p in learner.model.named_parameters():
                p.requires_grad = name in head_names

    def _trainable_params_for_wake(self, learner) -> Iterable[Tensor]:
        if not self.cfg.wake_freeze_backbone:
            return [p for p in learner.model.parameters() if p.requires_grad]
        names = set(self._head_param_names(learner.model))
        extra = self.cfg.wake_extra_unfrozen or []
        params: List[Tensor] = []
        for name, p in learner.model.named_parameters():
            if name in names or any(tag in name for tag in extra):
                params.append(p)
        if not params:
            params = list(learner.model.parameters())
        return params

    def _trainable_params_for_sleep(self, learner) -> Iterable[Tensor]:
        if self.cfg.train_backbone_in_sleep:
            return [p for p in learner.model.parameters() if p.requires_grad]
        names = set(self._head_param_names(learner.model))
        params = [p for n, p in learner.model.named_parameters() if n in names]
        if not params:
            params = list(learner.model.parameters())
        return params

    def _head_param_names(self, model: nn.Module) -> List[str]:
        head_attr = self.cfg.head_attr
        names: List[str] = []
        if hasattr(model, head_attr):
            head = getattr(model, head_attr)
            for name, _ in head.named_parameters():
                names.append(f"{head_attr}.{name}")
            return names
        # fallback: last module in children
        children = list(model.named_children())
        if children:
            last_name, last_module = children[-1]
            for n, _ in last_module.named_parameters():
                names.append(f"{last_name}.{n}")
        return names

    def _build_optimizer(self, learner, params: Iterable[Tensor], opt_cfg: OptimizerConfig):
        opt = torch.optim.SGD(
            params,
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
        )
        return learner.fabric.setup_optimizers(opt)

    def _forward_from_latent(self, model: nn.Module, latents: Tensor) -> Tensor:
        head_attr = self.cfg.head_attr
        if hasattr(model, head_attr):
            head = getattr(model, head_attr)
            return head(latents)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
            return model.classifier(latents)
        # Fallback: assume the model itself can process latent vectors
        return model(latents)

    # ---- Introspection ----
    @property
    def sleep_log(self) -> List[Dict[str, Any]]:
        return self._sleep_log
