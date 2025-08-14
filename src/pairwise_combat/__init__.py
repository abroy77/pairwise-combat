"""
pairwise_combat package initialization.
"""

from .core import PairwiseComBAT
from .polars_interface import PairwiseComBATDataFrame, QuantileRegressorDataFrame
from .quantile_regressor import MultiQuantileRegressor

__version__ = "0.1.0"
__all__ = ["PairwiseComBAT", "PairwiseComBATDataFrame", "QuantileRegressorDataFrame", "MultiQuantileRegressor"]
