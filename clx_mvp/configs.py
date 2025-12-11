# clx_mvp/configs.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """
    Minimal optimizer hyperparameters used across strategies.
    """
    lr: float = 0.03
    weight_decay: float = 5e-4
    momentum: float = 0.9


@dataclass
class LinearScheduleConfig:
    """
    Linear schedule defined by start/end values over a fixed number of steps.

    Attributes:
        start: initial value at step 0.
        end: value at step >= total_steps.
        total_steps: number of steps over which to interpolate.
        clamp: if True, clamp to [min(start, end), max(start, end)] after interpolation.
    """
    start: float
    end: float
    total_steps: int
    clamp: bool = True


def linear_schedule(step: int, cfg: LinearScheduleConfig) -> float:
    """
    Compute the linearly-interpolated value for a given step.
    """
    if cfg.total_steps <= 0:
        return cfg.end
    t = min(max(step, 0), cfg.total_steps)
    frac = t / float(cfg.total_steps)
    val = cfg.start + frac * (cfg.end - cfg.start)
    if not cfg.clamp:
        return val
    lo, hi = sorted([cfg.start, cfg.end])
    return float(min(max(val, lo), hi))
