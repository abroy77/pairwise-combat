"""
Linear Quantile Regression Model with multiple percentiles.

This module provides a linear quantile regression model that fits models for all
integer percentiles between 1 and 99, with capabilities for saving/loading
coefficients and plotting quantile lines.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Optional, Tuple, List, Union
from sklearn.linear_model import QuantileRegressor
from datetime import datetime
from tqdm import tqdm
import warnings


class MultiQuantileRegressor:
    """
    Multi-percentile linear quantile regression model using scikit-learn.

    Fits linear quantile regression models for all integer percentiles from 1 to 99
    for multi-dimensional features. Provides functionality for prediction, plotting,
    and model persistence.
    """

    def __init__(
        self,
        metric_name: Optional[str] = None,
        alpha: float = 1.0,
        solver: str = "highs",
        fit_intercept: bool = True,
    ):
        """
        Initialize the multi-quantile regressor.

        Args:
            alpha: Regularization strength for quantile regression
            solver: Solver to use ('highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex')
            fit_intercept: Whether to fit an intercept term
        """
        self.metric_name = metric_name
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept

        # Will be populated after fitting
        self.coefficients_ = {}  # percentile -> coefficients
        self.intercepts_ = {}  # percentile -> intercept
        self.percentiles_ = list(range(1, 100))  # 1 to 99 inclusive
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiQuantileRegressor":
        """
        Fit quantile regression models for all percentiles.

        Args:
            X: Input features, shape (n_samples, n_features) - 2D array with features
               or (n_samples,) - 1D array will be reshaped to (n_samples, 1)
            y: Target values, shape (n_samples,)

        Returns:
            self: Fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Handle input dimensionality
        if X.ndim == 1:
            # 1D input, reshape to 2D
            X = X.reshape(-1, 1)
            self.n_features_ = 1
        elif X.ndim == 2:
            # 2D input, use as-is
            self.n_features_ = X.shape[1]
        else:
            raise ValueError(f"Input X must be 1D or 2D array. Got {X.ndim}D array.")

        print(
            f"Fitting quantile regression for {len(self.percentiles_)} percentiles with {self.n_features_} features..."
        )

        # Fit models for each percentile
        for percentile in tqdm(self.percentiles_):
            quantile = percentile / 100.0

            # Create and fit quantile regressor
            qr = QuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                solver=self.solver,
                fit_intercept=self.fit_intercept
            )

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            qr.fit(X, y)

            # Store coefficients and intercept
            self.coefficients_[percentile] = qr.coef_.copy()
            if self.fit_intercept:
                self.intercepts_[percentile] = qr.intercept_
            else:
                self.intercepts_[percentile] = 0.0

        self.is_fitted_ = True
        print("Quantile regression fitting completed!")
        return self

    def predict(self, X: np.ndarray, percentile: int) -> np.ndarray:
        """
        Predict values for a given percentile.

        Args:
            X: Input features, shape (n_samples, n_features) - 2D array with features
               or (n_samples,) - 1D array will be reshaped to (n_samples, 1)
            percentile: Percentile to predict (1-99)

        Returns:
            Predicted values, shape (n_samples,)
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if percentile not in self.percentiles_:
            raise ValueError(
                f"Percentile {percentile} not available. Must be in range 1-99."
            )

        X = np.asarray(X)

        # Handle input dimensionality
        if X.ndim == 1:
            # 1D input, reshape to 2D
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            # 2D input, check feature consistency
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"Input has {X.shape[1]} features, but model was fitted with {self.n_features_} features."
                )
        else:
            raise ValueError(f"Input X must be 1D or 2D array. Got {X.ndim}D array.")

        # Predict using stored coefficients
        predictions = X @ self.coefficients_[percentile] + self.intercepts_[percentile]
        return predictions.flatten()

    def predict_single(
        self, x_value: Union[float, np.ndarray], percentile: int
    ) -> float:
        """
        Predict a single value for a given percentile.

        Args:
            x_value: Single input value (float) for single feature, or array-like for multiple features
            percentile: Percentile to predict (1-99)

        Returns:
            Predicted value (float)
        """
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

        if isinstance(x_value, (int, float)):
            # Single feature case
            if self.n_features_ != 1:
                raise ValueError(
                    f"Model expects {self.n_features_} features, but got single value."
                )
            X = np.array([[x_value]])
        else:
            # Multiple features case
            x_value = np.asarray(x_value)
            if x_value.ndim == 1:
                X = x_value.reshape(1, -1)
            else:
                X = x_value
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"Model expects {self.n_features_} features, but got {X.shape[1]}."
                )

        return self.predict(X, percentile)[0]

    def find_closest_percentile(self, X: np.ndarray, y: float) -> int:
        """
        Given an input X and an output y, find the percentile whose prediction is closest to y.

        Args:
            X: Input features, single sample - shape (n_features,) or (1, n_features)
            y: Target value (float)

        Returns:
            Closest percentile (int)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before using this method.")

        X = np.asarray(X)

        # Handle input dimensionality for single sample
        if X.ndim == 1:
            # 1D input, reshape to 2D single sample
            if self.n_features_ == 1 and X.size == 1:
                X = X.reshape(1, 1)
            elif X.size == self.n_features_:
                X = X.reshape(1, -1)
            else:
                raise ValueError(
                    f"find_closest_percentile expects a single sample with {self.n_features_} features"
                )
        elif X.ndim == 2:
            if X.shape[0] != 1:
                raise ValueError("find_closest_percentile expects a single sample")
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"find_closest_percentile expects {self.n_features_} features, got {X.shape[1]}"
                )
        else:
            raise ValueError(f"Input X must be 1D or 2D array. Got {X.ndim}D array.")

        min_diff = float("inf")
        closest_percentile = self.percentiles_[0]

        # Simple linear search through all percentiles
        for percentile in self.percentiles_:
            pred = X @ self.coefficients_[percentile] + self.intercepts_[percentile]
            pred_val = pred.item()
            diff = abs(pred_val - y)

            if diff < min_diff:
                min_diff = diff
                closest_percentile = percentile

        return closest_percentile

    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to an HDF5 file.

        Args:
            filepath: Path where to save the model
        """
        return self.save(filepath)

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to an HDF5 file.

        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving. Call fit() first.")

        filepath = Path(filepath)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")

        with h5py.File(filepath, "w") as f:
            # Create groups
            params_group = f.create_group("parameters")
            config_group = f.create_group("config")
            metadata_group = f.create_group("metadata")

            # Save coefficients for each percentile
            coeffs_group = params_group.create_group("coefficients")
            intercepts_group = params_group.create_group("intercepts")

            for percentile in self.percentiles_:
                coeffs_group.create_dataset(
                    f"percentile_{percentile}", data=self.coefficients_[percentile]
                )
                intercepts_group.create_dataset(
                    f"percentile_{percentile}", data=self.intercepts_[percentile]
                )

            # Save configuration
            config_group.attrs["alpha"] = self.alpha
            config_group.attrs["solver"] = self.solver
            config_group.attrs["fit_intercept"] = self.fit_intercept
            config_group.attrs["n_features"] = self.n_features_

            # Save percentiles array
            params_group.create_dataset("percentiles", data=np.array(self.percentiles_))

            # Save metadata
            metadata_group.attrs["metric_name"] = self.metric_name
            metadata_group.attrs["model_type"] = "MultiQuantileRegressor"
            metadata_group.attrs["version"] = "1.0"
            metadata_group.attrs["creation_time"] = datetime.now().isoformat()
            metadata_group.attrs["fitted"] = True

        print(f"Model saved to {filepath}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")

    def load_model(self, filepath: str) -> None:
        """
        Load a model from an HDF5 file.

        Args:
            filepath: Path to the saved model
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "r") as f:
            # Load configuration
            config_group = f["config"]
            self.alpha = config_group.attrs["alpha"]
            self.solver = config_group.attrs["solver"]
            self.fit_intercept = config_group.attrs["fit_intercept"]
            self.n_features_ = config_group.attrs["n_features"]

            # Load percentiles
            params_group = f["parameters"]
            self.percentiles_ = params_group["percentiles"][:].tolist()

            # Load coefficients and intercepts
            coeffs_group = params_group["coefficients"]
            intercepts_group = params_group["intercepts"]

            self.coefficients_ = {}
            self.intercepts_ = {}

            for percentile in self.percentiles_:
                self.coefficients_[percentile] = coeffs_group[
                    f"percentile_{percentile}"
                ][:]
                self.intercepts_[percentile] = intercepts_group[
                    f"percentile_{percentile}"
                ][()]

            # Check metadata for fitted status
            if "metadata" in f:
                metadata_group = f["metadata"]
                self.is_fitted_ = metadata_group.attrs["fitted"]
                self.metric_name = metadata_group.attrs["metric_name"]
            else:
                self.is_fitted_ = True

    @staticmethod
    def load(filepath: str) -> "MultiQuantileRegressor":
        """
        Load a model from file (static method).

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded MultiQuantileRegressor instance
        """
        model = MultiQuantileRegressor()
        model.load_model(filepath)
        return model

    @staticmethod
    def load_from_file(filepath: str) -> "MultiQuantileRegressor":
        """
        Load a model from file (static method - alias for load).

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded MultiQuantileRegressor instance
        """
        return MultiQuantileRegressor.load(filepath)
