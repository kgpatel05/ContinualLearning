# clx_mvp/__init__.py
"""
Minimal continual-learning MVP: Fabric-driven loop + ER replay + CIFAR10 class-IL stream.
"""

from .streams import (
    Experience,
    build_cifar10_cil_stream,
    build_cifar100_cil_stream,
    build_joint_stream_from_cil,
)
from .replay import ERBuffer, RichERBuffer
from .models import build_resnet18
from .learner import Learner, FitReport
from .metrics import accuracy, average_accuracy, compute_forgetting, ContinualEvaluator
from .strategies import (
    CLStrategy,
    ERStrategy,
    FinetuneStrategy,
    EWCStrategy,
    LwFStrategy,
    AGEMStrategy,
    BASRStrategy,
)

__all__ = [
    "Experience",
    "build_cifar10_cil_stream",
    "build_cifar100_cil_stream",
    "build_joint_stream_from_cil",
    "ERBuffer",
    "RichERBuffer",
    "build_resnet18",
    "Learner",
    "FitReport",
    "accuracy",
    "average_accuracy",
    "compute_forgetting",
    "ContinualEvaluator",
    "CLStrategy",
    "ERStrategy",
    "FinetuneStrategy",
    "EWCStrategy",
    "LwFStrategy",
    "AGEMStrategy",
    "BASRStrategy",
]
