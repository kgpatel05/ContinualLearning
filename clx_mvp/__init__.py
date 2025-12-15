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
from .replay import ERBuffer, RichERBuffer, LatentReplayBuffer
from .models import build_resnet18
from .learner import Learner, FitReport
from .metrics import (
    accuracy,
    average_accuracy,
    average_last_row,
    average_over_matrix,
    compute_forgetting,
    ContinualEvaluator,
    classwise_accuracy_and_confusion,
    classwise_accuracy_over_stream,
    classification_report_from_confusion,
    count_trainable_params,
    estimate_buffer_memory_bytes,
    estimate_model_memory_bytes,
    estimate_flops,
    summarize_efficiency,
)
from .compression import LatentCompressionConfig, LatentCompressor, build_compressor
from .features import FeatureExtractor, FeatureExtractorConfig
from .configs import OptimizerConfig, LinearScheduleConfig
from .lora import LoraConfig
from .strategies import (
    CLStrategy,
    ERStrategy,
    FinetuneStrategy,
    EWCStrategy,
    LwFStrategy,
    AGEMStrategy,
    BASRStrategy,
    SiestaConfig,
    SleepScheduleConfig,
    SiestaStrategy,
    GraspConfig,
    GraspRehearsalPolicy,
    GraspStrategy,
    SgmConfig,
    SgmStrategy,
)

__all__ = [
    "Experience",
    "build_cifar10_cil_stream",
    "build_cifar100_cil_stream",
    "build_joint_stream_from_cil",
    "ERBuffer",
    "RichERBuffer",
    "LatentReplayBuffer",
    "build_resnet18",
    "Learner",
    "FitReport",
    "accuracy",
    "average_accuracy",
    "average_last_row",
    "average_over_matrix",
    "compute_forgetting",
    "ContinualEvaluator",
    "classwise_accuracy_and_confusion",
    "classwise_accuracy_over_stream",
    "classification_report_from_confusion",
    "count_trainable_params",
    "estimate_buffer_memory_bytes",
    "estimate_model_memory_bytes",
    "estimate_flops",
    "summarize_efficiency",
    "LatentCompressionConfig",
    "LatentCompressor",
    "build_compressor",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "OptimizerConfig",
    "LinearScheduleConfig",
    "LoraConfig",
    "CLStrategy",
    "ERStrategy",
    "FinetuneStrategy",
    "EWCStrategy",
    "LwFStrategy",
    "AGEMStrategy",
    "BASRStrategy",
    "SiestaConfig",
    "SleepScheduleConfig",
    "SiestaStrategy",
    "GraspConfig",
    "GraspRehearsalPolicy",
    "GraspStrategy",
    "SgmConfig",
    "SgmStrategy",
]
