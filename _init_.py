"""
Holographic Chemistry Package
=============================

A novel framework for understanding chemical properties through holographic duality.
"""

__version__ = "1.0.0"
__author__ = "Holographic Chemistry Research Group"
__email__ = "contact@holographic-chemistry.org"

from .core import (
    HolographicModel,
    NobleGasCorrection,
    PeriodicTableAnalyzer
)

from .models import (
    BenchmarkModel,
    EmergentModel,
    CompleteModel,
    TransitionMetalExtension
)

from .bayesian import (
    BayesianHierarchicalModel,
    MCMCSampler,
    PosteriorAnalyzer
)

from .validation import (
    ValidationMetrics,
    CrossValidator,
    PredictionValidator
)

from .visualization import (
    PeriodicTablePlotter,
    HolographicMapVisualizer,
    ContributionAnalyzer
)

__all__ = [
    "HolographicModel",
    "NobleGasCorrection",
    "PeriodicTableAnalyzer",
    "BenchmarkModel",
    "EmergentModel",
    "CompleteModel",
    "TransitionMetalExtension",
    "BayesianHierarchicalModel",
    "MCMCSampler",
    "PosteriorAnalyzer",
    "ValidationMetrics",
    "CrossValidator",
    "PredictionValidator",
    "PeriodicTablePlotter",
    "HolographicMapVisualizer",
    "ContributionAnalyzer"
]
