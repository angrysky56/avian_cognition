"""
Avian Cognitive Architecture

A neural architectural framework synthesizing Mamba-SSM selective state modeling with
BitNet extreme quantization, enhanced through avian-inspired cognitive circuits for
metacognition, Bayesian inference, planning, and numerical computation.
"""

from .core.mamba_integration import (
    AvianMambaModel,
    AvianMambaConfig,
    create_mini_model,
    create_small_model,
    create_medium_model
)

from .modules.metacognition import MetacognitionModule
from .modules.bayesian import BayesianInferenceModule
from .modules.planning import PlanningModule
from .modules.numerical import NumericalModule

__version__ = "0.1.0"
