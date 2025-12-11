# Aggregated strategies package for continual learning hooks.
from .base import CLStrategy
from .er import ERStrategy
from .finetune import FinetuneStrategy
from .ewc import EWCStrategy
from .lwf import LwFStrategy
from .agem import AGEMStrategy
from .basr import BASRStrategy

__all__ = [
    "CLStrategy",
    "ERStrategy",
    "FinetuneStrategy",
    "EWCStrategy",
    "LwFStrategy",
    "AGEMStrategy",
    "BASRStrategy",
]
