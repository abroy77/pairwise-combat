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
import warnings


class MultiQuantileRegressor:
    """
    Multi-percentile linear quantile regression model using scikit-learn.
    
    Fits linear quantile regression models for all integer percentiles from 1 to 99
    for a single feature (1D input). Provides functionality for prediction, plotting, 
    and model persistence.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        solver: str = "highs",
        fit_intercept: bool = True
    ):
        """
        Initialize the multi-quantile regressor.
        
        Args:
            alpha: Regularization strength for quantile regression
            solver: Solver to use ('highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex')
            fit_intercept: Whether to fit an intercept term
        """
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        
        # Will be populated after fitting
        self.coefficients_ = {}  # percentile -> coefficients
        self.intercepts_ = {}    # percentile -> intercept
        self.percentiles_ = list(range(1, 100))  # 1 to 99 inclusive
        self.n_features_ = None
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiQuantileRegressor":
        """
        Fit quantile regression models for all percentiles.
        
        Args:
            X: Input features, shape (n_samples,) - 1D array with single feature
               or (n_samples, 1) - 2D array with single feature (for backward compatibility)
            y: Target values, shape (n_samples,)
            
        Returns:
            self: Fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Handle both 1D and 2D single-feature input
        if X.ndim == 1:
            # Already 1D, reshape for sklearn
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            # Check if it's single feature
            if X.shape[1] == 1:
                # Already correct shape for sklearn
                pass
            else:
                raise ValueError(f"Only single feature (1D) input is supported. Got {X.shape[1]} features.")
        else:
            raise ValueError(f"Input X must be 1D or 2D array with single feature. Got {X.ndim}D array.")
        
        self.n_features_ = 1  # Always single feature
        
        print(f"Fitting quantile regression for {len(self.percentiles_)} percentiles...")
        
        # Fit models for each percentile
        for i, percentile in enumerate(self.percentiles_):
            quantile = percentile / 100.0
            
            # Create and fit quantile regressor
            qr = QuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                solver=self.solver,
                fit_intercept=self.fit_intercept
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qr.fit(X, y)
            
            # Store coefficients and intercept
            self.coefficients_[percentile] = qr.coef_.copy()
            if self.fit_intercept:
                self.intercepts_[percentile] = qr.intercept_
            else:
                self.intercepts_[percentile] = 0.0
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(self.percentiles_)} percentiles")
        
        self.is_fitted_ = True
        print("Quantile regression fitting completed!")
        return self
    
    def predict(
        self, 
        X: np.ndarray, 
        percentile: int
    ) -> np.ndarray:
        """
        Predict values for a given percentile.
        
        Args:
            X: Input features, shape (n_samples,) - 1D array with single feature
               or (n_samples, 1) - 2D array with single feature (for backward compatibility)
            percentile: Percentile to predict (1-99)
            
        Returns:
            Predicted values, shape (n_samples,)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        if percentile not in self.percentiles_:
            raise ValueError(f"Percentile {percentile} not available. Must be in range 1-99.")
        
        X = np.asarray(X)
        
        # Handle both 1D and 2D single-feature input
        if X.ndim == 1:
            # 1D input, reshape for sklearn
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            # Check if it's single feature
            if X.shape[1] == 1:
                # Already correct shape for sklearn
                pass
            else:
                raise ValueError(f"Only single feature (1D) input is supported. Got {X.shape[1]} features.")
        else:
            raise ValueError(f"Input X must be 1D or 2D array with single feature. Got {X.ndim}D array.")
        
        # Predict using stored coefficients
        predictions = X @ self.coefficients_[percentile] + self.intercepts_[percentile]
        return predictions.flatten()
    
    def predict_single(self, x_value: float, percentile: int) -> float:
        """
        Predict a single value for a given percentile.
        
        Args:
            x_value: Single input value (float)
            percentile: Percentile to predict (1-99)
            
        Returns:
            Predicted value (float)
        """
        X = np.array([x_value])
        return self.predict(X, percentile)[0]   
    
    def predict_all_percentiles(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all percentiles for given input data.
        
        Args:
            X: Input features, shape (n_samples,) or (n_samples, 1)
            
        Returns:
            Predictions array with shape (n_samples, n_percentiles)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.asarray(X)
        
        # Handle both 1D and 2D single-feature input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 2 and X.shape[1] == 1:
            pass  # Already correct shape
        else:
            raise ValueError("Only single feature (1D) input is supported")
            
        n_samples = X.shape[0]
        n_percentiles = len(self.percentiles_)
        predictions = np.zeros((n_samples, n_percentiles))
        
        for i, percentile in enumerate(self.percentiles_):
            # Predict using stored coefficients
            pred = X @ self.coefficients_[percentile] + self.intercepts_[percentile]
            predictions[:, i] = pred.flatten()
            
        return predictions   
    
    def find_closest_percentile(self, X: np.ndarray, y: float) -> int:
        """
        Given an input X and an output y, find the percentile whose prediction is closest to y.

        Args:
            X: Input features, single sample - shape (1,) or (1, 1)
            y: Target value (float)

        Returns:
            Closest percentile (int)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before using this method.")

        X = np.asarray(X)
        
        # Handle both 1D and 2D single-feature input
        if X.ndim == 1:
            if X.size != 1:
                raise ValueError("find_closest_percentile expects a single sample")
            X = X.reshape(1, 1)
        elif X.ndim == 2:
            if X.shape != (1, 1):
                raise ValueError("find_closest_percentile expects a single sample with single feature")
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
    
    def plot_quantile_lines(
        self, 
        X_train: Optional[np.ndarray] = None, 
        y_train: Optional[np.ndarray] = None,
        percentiles_to_plot: Optional[List[int]] = None,
        X_plot: Optional[np.ndarray] = None, 
        percentiles: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Quantile Regression Lines"
    ) -> plt.Figure:
        """
        Plot quantile regression lines.
        
        Args:
            X_train: Training data for plotting (alternative parameter name)
            y_train: Training targets for plotting (alternative parameter name)  
            percentiles_to_plot: List of percentiles to plot (alternative parameter name)
            X_plot: X values for plotting. If None, uses a default range
            percentiles: List of percentiles to plot. If None, uses [10, 25, 50, 75, 90]
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")
        
        # Handle alternative parameter names for backward compatibility
        if percentiles_to_plot is not None:
            percentiles = percentiles_to_plot
        if X_train is not None:
            X_plot = X_train
            
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90]
        
        if X_plot is None:
            X_plot = np.linspace(0, 10, 100)
        
        # Ensure X_plot is 1D for plotting
        if X_plot.ndim == 2 and X_plot.shape[1] == 1:
            X_plot = X_plot.flatten()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data if provided
        if y_train is not None and X_train is not None:
            X_train_flat = X_train.flatten() if X_train.ndim == 2 else X_train
            ax.scatter(X_train_flat, y_train, alpha=0.3, color='gray', s=10, label='Training data')
        
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        styles = ['--', '-.', '-', '-.', '--']
        
        for i, percentile in enumerate(percentiles):
            if percentile in self.percentiles_:
                y_pred = self.predict(X_plot, percentile)
                color = colors[i % len(colors)]
                style = styles[i % len(styles)]
                ax.plot(X_plot, y_pred, color=color, linestyle=style, 
                       linewidth=2, label=f'{percentile}th percentile')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Predicted Y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_all_quantiles(
        self, 
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_plot: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8),
        alpha: float = 0.3
    ) -> plt.Figure:
        """
        Plot all quantile regression lines.
        
        Args:
            X_train: Training data for plotting (alternative parameter name)
            y_train: Training targets for plotting (alternative parameter name)
            X_plot: X values for plotting. If None, uses a default range
            figsize: Figure size
            alpha: Transparency for the quantile lines
            
        Returns:
            matplotlib Figure object
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before plotting. Call fit() first.")
        
        # Handle alternative parameter names for backward compatibility
        if X_train is not None:
            X_plot = X_train
        
        if X_plot is None:
            X_plot = np.linspace(0, 10, 100)
        
        # Ensure X_plot is 1D for plotting  
        if X_plot.ndim == 2 and X_plot.shape[1] == 1:
            X_plot = X_plot.flatten()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data if provided
        if y_train is not None and X_train is not None:
            X_train_flat = X_train.flatten() if X_train.ndim == 2 else X_train
            ax.scatter(X_train_flat, y_train, alpha=0.3, color='gray', s=10, label='Training data')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all percentiles
        for percentile in self.percentiles_:
            y_pred = self.predict(X_plot, percentile)
            ax.plot(X_plot, y_pred, color='blue', alpha=alpha, linewidth=1)
        
        # Highlight key percentiles
        key_percentiles = [10, 25, 50, 75, 90]
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        
        for percentile, color in zip(key_percentiles, colors):
            if percentile in self.percentiles_:
                y_pred = self.predict(X_plot, percentile)
                ax.plot(X_plot, y_pred, color=color, linewidth=2, 
                       label=f'{percentile}th percentile')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Predicted Y')
        ax.set_title('All Quantile Regression Lines')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
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
                    f"percentile_{percentile}", 
                    data=self.coefficients_[percentile]
                )
                intercepts_group.create_dataset(
                    f"percentile_{percentile}",
                    data=self.intercepts_[percentile]
                )
            
            # Save configuration
            config_group.attrs["alpha"] = self.alpha
            config_group.attrs["solver"] = self.solver
            config_group.attrs["fit_intercept"] = self.fit_intercept
            config_group.attrs["n_features"] = self.n_features_
            
            # Save percentiles array
            params_group.create_dataset("percentiles", data=np.array(self.percentiles_))
            
            # Save metadata
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
                self.coefficients_[percentile] = coeffs_group[f"percentile_{percentile}"][:]
                self.intercepts_[percentile] = intercepts_group[f"percentile_{percentile}"][()]
            
            # Check metadata for fitted status
            if "metadata" in f:
                metadata_group = f["metadata"]
                self.is_fitted_ = metadata_group.attrs["fitted"]
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
    
    def get_model_info(self) -> dict:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first.")
        
        return {
            "model_type": "MultiQuantileRegressor",
            "n_percentiles": len(self.percentiles_),
            "percentile_range": f"{min(self.percentiles_)}-{max(self.percentiles_)}",
            "n_features": 1,  # Always single feature
            "alpha": self.alpha,
            "solver": self.solver,
            "fit_intercept": self.fit_intercept,
            "is_fitted": self.is_fitted_
        }


