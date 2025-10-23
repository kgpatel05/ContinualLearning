# clx_mvp/__init__.py
"""
Minimal continual-learning MVP: Fabric-driven loop + ER replay + CIFAR10 class-IL stream.
"""

from .streams import Experience, build_cifar10_cil_stream
from .replay import ERBuffer
from .models import build_resnet18
from .learner import Learner, FitReport
from .metrics import accuracy, average_accuracy, compute_forgetting

__all__ = [
    "Experience",
    "build_cifar10_cil_stream",
    "ERBuffer",
    "build_resnet18",
    "Learner",
    "FitReport",
    "accuracy",
    "average_accuracy",
    "compute_forgetting",
]
