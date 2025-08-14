"""
Polars DataFrame interface for PairwiseComBAT harmonization.

This module provides a user-friendly DataFrame interface to the PairwiseComBAT
harmonization algorithm, allowing users to work directly with Polars DataFrames
instead of NumPy arrays.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import h5py

from pairwise_combat import PairwiseComBAT
from pairwise_combat.quantile_regressor import MultiQuantileRegressor


class PairwiseComBATDataFrame:
    """
    DataFrame interface for PairwiseComBAT harmonization using Polars.
    
    This class provides a higher-level interface that works directly with
    Polars DataFrames, handling data validation, conversion, and reshaping
    automatically.
    """
    model: PairwiseComBAT
    site_id_col: Optional[str]
    data_cols: Optional[List[str]]
    covariate_cols: Optional[List[str]]
    is_fitted_: bool
    
    def __init__(
        self,
        metric: str = "metric",
        source_site: str = "source",
        target_site: str = "target",
        max_iter: int = 30,
        tol: float = 1e-6,
    ):
        """
        Initialize PairwiseComBAT DataFrame interface.
        
        Args:
            source_site: Name of the source site (must match values in site_id column)
            target_site: Name of the target/reference site (must match values in site_id column)
            max_iter: Maximum iterations for Bayesian estimation
            tol: Convergence tolerance
        """
        # Initialize the underlying PairwiseComBAT model
        self.model = PairwiseComBAT(
            metric=metric,
            source_site=source_site,
            target_site=target_site,
            max_iter=max_iter,
            tol=tol
        )
        
        # Store column information for validation
        self.site_id_col = None
        self.data_cols: None
        self.covariate_cols: None
        self.is_fitted_: bool = False
    
    def _validate_dataframe(
        self,
        df: pl.DataFrame,
        site_id_col: str,
        data_cols: List[str],
        covariate_cols: List[str],
    ) -> None:
        """
        Validate the input DataFrame structure and contents.
        
        Args:
            df: Input DataFrame
            site_id_col: Name of the site identifier column
            data_cols: List of data column names (locations/voxels)
            covariate_cols: List of covariate column names
            
        Raises:
            ValueError: If validation fails
        """
        # Check that required columns exist
        missing_cols = []
        if site_id_col not in df.columns:
            missing_cols.append(site_id_col)
        
        for col in data_cols + covariate_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        # Check that site_id column contains required site values
        site_values = df[site_id_col].unique().to_list()
        if self.model.source_site not in site_values:
            raise ValueError(f"Source site '{self.model.source_site}' not found in {site_id_col} column. Available: {site_values}")
        if self.model.target_site not in site_values:
            raise ValueError(f"Target site '{self.model.target_site}' not found in {site_id_col} column. Available: {site_values}")
        
        # Check that data and covariate columns are numeric
        non_numeric_cols = []
        for col in data_cols + covariate_cols:
            if not df[col].dtype.is_numeric():
                non_numeric_cols.append(f"{col} ({df[col].dtype})")
        
        if non_numeric_cols:
            raise ValueError(f"Non-numeric columns found: {non_numeric_cols}. All data and covariate columns must be numeric.")
        
        # Check that we have data for both sites
        source_count = df.filter(pl.col(site_id_col) == self.model.source_site).height
        target_count = df.filter(pl.col(site_id_col) == self.model.target_site).height
        
        if source_count == 0:
            raise ValueError(f"No data found for source site '{self.model.source_site}'")
        if target_count == 0:
            raise ValueError(f"No data found for target site '{self.model.target_site}'")
        
    
    def _prepare_arrays(
        self,
        df: pl.DataFrame,
        site_id_col: str,
        data_cols: List[str],
        covariate_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert DataFrame to NumPy arrays for the underlying PairwiseComBAT model.
        
        Args:
            df: Input DataFrame
            site_id_col: Name of the site identifier column
            data_cols: List of data column names
            covariate_cols: List of covariate column names
            
        Returns:
            Tuple of (covars_ref, Y_ref, covars_moving, Y_moving)
        """
        # Ensure all data and covariate columns are float64
        df = df.with_columns([
            pl.col(col).cast(pl.Float64) for col in data_cols + covariate_cols
        ])
        
        # Split data by site
        df_ref = df.filter(pl.col(site_id_col) == self.model.target_site)
        df_moving = df.filter(pl.col(site_id_col) == self.model.source_site)
        
        # Extract reference site data
        Y_ref = df_ref.select(data_cols).to_numpy().T  # Shape: (n_locations, n_samples)
        covars_ref = df_ref.select(covariate_cols).to_numpy().T  # Shape: (n_covariates, n_samples)
        
        # Extract moving site data
        Y_moving = df_moving.select(data_cols).to_numpy().T  # Shape: (n_locations, n_samples)
        covars_moving = df_moving.select(covariate_cols).to_numpy().T  # Shape: (n_covariates, n_samples)
        
        return covars_ref, Y_ref, covars_moving, Y_moving
    
    def fit(
        self,
        df: pl.DataFrame,
        site_id_col: str,
        data_cols: List[str],
        covariate_cols: List[str],
    ) -> "PairwiseComBATDataFrame":
        """
        Fit the PairwiseComBAT model using DataFrame data.
        
        Args:
            df: Training DataFrame containing data from both sites
            site_id_col: Name of column containing site identifiers
            data_cols: List of column names containing response data (locations/voxels)
            covariate_cols: List of column names containing covariates
            
        Returns:
            self: Fitted harmonizer instance
            
        Example:
            >>> combat_df = PairwiseComBATDataFrame(source_site="site_A", target_site="site_B")
            >>> combat_df.fit(
            ...     df=train_data,
            ...     site_id_col="site",
            ...     data_cols=["voxel_1", "voxel_2", "voxel_3"],
            ...     covariate_cols=["age", "sex"]
            ... )
        """
        # Store column information for later use
        self.site_id_col = site_id_col
        self.data_cols = data_cols.copy()
        self.covariate_cols = covariate_cols.copy()
        
        # Validate input
        self._validate_dataframe(df, site_id_col, data_cols, covariate_cols)
        
        # Prepare arrays
        covars_ref, Y_ref, covars_moving, Y_moving = self._prepare_arrays(
            df, site_id_col, data_cols, covariate_cols
        )
        
        # Fit the underlying model
        self.model.fit(
            covars_ref=covars_ref,
            Y_ref=Y_ref,
            covars_moving=covars_moving,
            Y_moving=Y_moving
        )
        
        self.is_fitted_ = True
        
        return self
    
    def transform(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Harmonize data from source site to target site characteristics.
        
        Args:
            df: DataFrame containing data to harmonize (must contain source site data)
            site_id_col: Name of site identifier column (uses fitted column if None)
            data_cols: List of data column names (uses fitted columns if None)
            covariate_cols: List of covariate column names (uses fitted columns if None)
            
        Returns:
            DataFrame with harmonized data columns (original columns are replaced)
            
        Example:
            >>> harmonized_df = combat_df.transform(test_data)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transformation. Call fit() first.")
        
        # Use fitted column names if not provided
        site_id_col = self.site_id_col
        data_cols = self.data_cols
        covariate_cols = self.covariate_cols
        
        # Validate that we have the required information
        if not all([site_id_col, data_cols, covariate_cols]):
            raise ValueError("Column names must be provided either during fit() or transform()")
        
        # Filter to source site data only
        source_data = df.filter(pl.col(site_id_col) == self.model.source_site)
        if source_data.height == 0:
            raise ValueError(f"No data found for source site '{self.model.source_site}' in transform data")
        
        # Prepare arrays for transformation
        source_data_float = source_data.with_columns([
            pl.col(col).cast(pl.Float64) for col in data_cols + covariate_cols
        ])
        
        Y_moving = source_data_float.select(data_cols).to_numpy().T
        covars_moving = source_data_float.select(covariate_cols).to_numpy().T
        
        # Apply harmonization
        Y_harmonized = self.model.predict(
            covars_moving=covars_moving,
            Y_moving=Y_moving
        )
        
        # Convert back to DataFrame format
        harmonized_data = pl.DataFrame(
            Y_harmonized.T,  # Transpose back to (samples, locations)
            schema=data_cols
        )
        
        # Create result DataFrame by replacing data columns
        result = source_data.drop(data_cols).hstack(harmonized_data)
        
        # If input contained non-source site data, include it unchanged
        other_sites = df.filter(pl.col(site_id_col) != self.model.source_site)
        if other_sites.height > 0:
            # Use diagonal concatenation to handle column mismatches
            result = pl.concat([result, other_sites], how="diagonal")
        
        return result
    
    
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving. Call fit() first.")
        
        self.model.save_model(filepath)
        # now save the additional data to this file
        with h5py.File(filepath, 'a') as f:  
            config_group = f["config"]
            # Save as comma-separated strings for lists
            config_group.attrs["site_id_col"] = self.site_id_col
            config_group.attrs["data_cols"] = ",".join(self.data_cols)
            config_group.attrs["covariate_cols"] = ",".join(self.covariate_cols)

            
            
    @staticmethod
    def load_model(filepath: str) -> "PairwiseComBATDataFrame":
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            PairwiseComBATDataFrame instance with loaded model
        """
        combat = PairwiseComBATDataFrame()
        combat.model.load_model(filepath)
        # load the rest manually
        with h5py.File(filepath, 'r') as f:
            config_group = f["config"]

            # site_id_col
            site_id_col = config_group.attrs.get("site_id_col", None)
            if isinstance(site_id_col, bytes):
                site_id_col = site_id_col.decode()
            combat.site_id_col = site_id_col if site_id_col else None

            # data_cols
            data_cols = config_group.attrs.get("data_cols", None)
            if data_cols is not None:
                if isinstance(data_cols, bytes):
                    data_cols = data_cols.decode()
                combat.data_cols = data_cols.split(",") if data_cols else None
            else:
                combat.data_cols = None

            # covariate_cols
            covariate_cols = config_group.attrs.get("covariate_cols", None)
            if covariate_cols is not None:
                if isinstance(covariate_cols, bytes):
                    covariate_cols = covariate_cols.decode()
                combat.covariate_cols = covariate_cols.split(",") if covariate_cols else None
            else:
                combat.covariate_cols = None

        combat.is_fitted_ = True
        return combat
    
    
    def get_model_info(self) -> dict:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first.")
        
        return {
            "source_site": self.model.source_site,
            "target_site": self.model.target_site,
            "n_locations": self.model.n_locations,
            "n_covariates": self.model.n_covariates,
            "n_samples_ref": self.model.n_samples_ref,
            "n_samples_moving": self.model.n_samples_moving,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "is_fitted": self.is_fitted_,
            "data_columns": self.data_cols,
            "covariate_columns": self.covariate_cols,
            "site_id_column": self.site_id_col,
        }


def create_example_dataframe(
    n_samples_per_site: int = 50,
    n_locations: int = 3,
    source_site: str = "site_A",
    target_site: str = "site_B",
    random_seed: int = 42,
) -> pl.DataFrame:
    """
    Create an example DataFrame for testing PairwiseComBAT DataFrame interface.
    
    Args:
        n_samples_per_site: Number of samples per site
        n_locations: Number of data locations/voxels
        n_covariates: Number of covariates
        source_site: Name of source site
        target_site: Name of target site
        random_seed: Random seed for reproducibility
        
    Returns:
        Example DataFrame with synthetic data
    """
    np.random.seed(random_seed)
    
    # Generate covariate data
    ages_source = np.random.uniform(20, 80, n_samples_per_site)
    ages_target = np.random.uniform(25, 75, n_samples_per_site)
    
    sex_source = np.random.randint(0, 2, n_samples_per_site)
    sex_target = np.random.randint(0, 2, n_samples_per_site)
    
    # Generate response data with site effects
    # Target site (reference)
    baseline_target = np.random.normal(5, 2, (n_samples_per_site, n_locations))
    age_effect_target = np.outer(ages_target - 50, np.random.uniform(0.02, 0.05, n_locations))
    sex_effect_target = np.outer(sex_target, np.random.uniform(-0.5, 0.5, n_locations))
    data_target = baseline_target + age_effect_target + sex_effect_target
    
    # Source site (with site effects)
    baseline_source = np.random.normal(5.5, 2.2, (n_samples_per_site, n_locations))  # Different baseline
    age_effect_source = np.outer(ages_source - 50, np.random.uniform(0.025, 0.055, n_locations))  # Different age effect
    sex_effect_source = np.outer(sex_source, np.random.uniform(-0.6, 0.6, n_locations))  # Different sex effect
    data_source = baseline_source + age_effect_source + sex_effect_source
    
    # Create DataFrames
    df_target = pl.DataFrame({
        "site_id": [target_site] * n_samples_per_site,
        "age": ages_target,
        "sex": sex_target.astype(float),
        **{f"voxel_{i+1}": data_target[:, i] for i in range(n_locations)}
    })
    
    df_source = pl.DataFrame({
        "site_id": [source_site] * n_samples_per_site,
        "age": ages_source,
        "sex": sex_source.astype(float),
        **{f"voxel_{i+1}": data_source[:, i] for i in range(n_locations)}
    })
    
    # Combine and shuffle
    df_combined = pl.concat([df_target, df_source], how="vertical")
    
    # Add sample IDs
    df_combined = df_combined.with_row_index("sample_id")
    
    return df_combined.sample(fraction=1.0, seed=random_seed)  # Shuffle rows


class QuantileRegressorDataFrame:
    """
    DataFrame interface for MultiQuantileRegressor using Polars.
    
    This class provides a higher-level interface for training quantile regression 
    models on multiple metrics and saving/loading them to/from a single HDF5 file.
    """
    
    def __init__(self, alpha: float = 1.0, solver: str = "highs", fit_intercept: bool = True):
        """
        Initialize the QuantileRegressorDataFrame.
        
        Args:
            alpha: Regularization strength for quantile regression
            solver: Solver to use ('highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex')
            fit_intercept: Whether to fit an intercept term
        """
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.models = {}  # metric_name -> MultiQuantileRegressor
        self.fitted_metrics = set()
        self.feature_cols = {}  # metric_name -> List[str] - stores feature column names for each fitted model
        
    def _validate_columns(self, df: pl.DataFrame, feature_cols: List[str], target_col: str) -> None:
        """
        Validate that required columns exist in the DataFrame.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            target_col: Target column name
            
        Raises:
            ValueError: If validation fails
        """
        missing_cols = []
        
        # Check feature columns
        for col in feature_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        # Check target column
        if target_col not in df.columns:
            missing_cols.append(target_col)
        
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
            
        # Check for missing values
        for col in feature_cols + [target_col]:
            if df[col].null_count() > 0:
                raise ValueError(f"Column '{col}' contains missing values")
    
    def fit(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> "QuantileRegressorDataFrame":
        """
        Fit a quantile regression model for a specific metric.
        
        Args:
            df: Training DataFrame
            feature_cols: List of column names to use as features (X)
            target_col: Column name to use as target (y)
            metric_name: Name to assign to this metric. If None, uses target_col
            
        Returns:
            self: Fitted instance
            
        Example:
            >>> qr_df = QuantileRegressorDataFrame(alpha=0.1)
            >>> qr_df.fit(
            ...     df=train_data,
            ...     feature_cols=["age", "sex", "education"],
            ...     target_col="cortical_thickness",
            ...     metric_name="cortical_thickness"
            ... )
        """
        # Use target_col as metric_name if not provided
            
        # Validate input
        self._validate_columns(df, feature_cols, target_col)
        
        # Extract arrays
        X = df.select(feature_cols).to_numpy()
        y = df.select(target_col).to_numpy().flatten()
        
        model = MultiQuantileRegressor(
            metric_name=target_col,
            alpha=self.alpha,
            solver=self.solver,
            fit_intercept=self.fit_intercept
        )
        
        print(f"Training quantile regressor for metric: {target_col}")
        model.fit(X, y)
        
        # Store model and feature columns
        self.models[target_col] = model
        self.fitted_metrics.add(target_col)
        self.feature_cols[target_col] = feature_cols.copy()  # Store the feature column names used during training
        
        return self
    
    def predict(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
        metric_name: str = None,
        percentile: int = None
    ) -> np.ndarray:
        """
        Predict values for a specific metric and percentile.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names. If None, uses the same columns that were used during training
            metric_name: Name of the fitted metric to use for prediction
            percentile: Percentile to predict (1-99)
            
        Returns:
            Predicted values as numpy array
            
        Raises:
            ValueError: If metric is not fitted or percentile is invalid
        """
        if metric_name not in self.fitted_metrics:
            raise ValueError(f"Metric '{metric_name}' has not been fitted. Available metrics: {list(self.fitted_metrics)}")
        
        # Use stored feature columns if not provided
        if feature_cols is None:
            if self.feature_cols[metric_name] is None:
                raise ValueError(
                    f"No feature column information available for metric '{metric_name}'. "
                    "This may be an older model file. Please provide feature_cols explicitly."
                )
            feature_cols = self.feature_cols[metric_name]
        else:
            # Validate that provided feature_cols match training feature_cols (if available)
            if self.feature_cols[metric_name] is not None:
                training_feature_cols = self.feature_cols[metric_name]
                if feature_cols != training_feature_cols:
                    raise ValueError(
                        f"Feature columns mismatch for metric '{metric_name}'. "
                        f"Training used: {training_feature_cols}, but got: {feature_cols}"
                    )
        
        # Validate columns (without target)
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")
        
        # Extract features in the same order as training
        X = df.select(feature_cols).to_numpy()
        
        # Predict
        return self.models[metric_name].predict(X, percentile)
    
    def predict_single(
        self,
        feature_values: dict,
        metric_name: str,
        percentile: int
    ) -> float:
        """
        Predict a single value for specific feature values.
        
        Args:
            feature_values: Dictionary mapping feature names to values
            metric_name: Name of the fitted metric to use for prediction
            percentile: Percentile to predict (1-99)
            
        Returns:
            Predicted value as float
            
        Example:
            >>> prediction = qr_df.predict_single(
            ...     feature_values={"age": 65, "sex": 1, "education": 16},
            ...     metric_name="cortical_thickness",
            ...     percentile=50
            ... )
        """
        if metric_name not in self.fitted_metrics:
            raise ValueError(f"Metric '{metric_name}' has not been fitted. Available metrics: {list(self.fitted_metrics)}")
        
        # Get the training feature columns for this metric
        if self.feature_cols[metric_name] is None:
            raise ValueError(
                f"No feature column information available for metric '{metric_name}'. "
                "This may be an older model file. Please retrain the model or provide a newer model file."
            )
        training_feature_cols = self.feature_cols[metric_name]
        
        # Validate that all required features are provided
        missing_features = [col for col in training_feature_cols if col not in feature_values]
        if missing_features:
            raise ValueError(f"Missing feature values for metric '{metric_name}': {missing_features}")
        
        # Create feature array in the same order as training
        feature_array = np.array([feature_values[col] for col in training_feature_cols])
        
        return self.models[metric_name].predict_single(feature_array, percentile)
    
    def predict_all_percentiles(
        self,
        df: pl.DataFrame,
        feature_cols: Optional[List[str]] = None,
        metric_name: str = None
    ) -> np.ndarray:
        """
        Predict all percentiles for a specific metric.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names. If None, uses the same columns that were used during training
            metric_name: Name of the fitted metric to use for prediction
            
        Returns:
            Predictions array with shape (n_samples, n_percentiles)
        """
        if metric_name not in self.fitted_metrics:
            raise ValueError(f"Metric '{metric_name}' has not been fitted. Available metrics: {list(self.fitted_metrics)}")
        
        # Use stored feature columns if not provided
        if feature_cols is None:
            if self.feature_cols[metric_name] is None:
                raise ValueError(
                    f"No feature column information available for metric '{metric_name}'. "
                    "This may be an older model file. Please provide feature_cols explicitly."
                )
            feature_cols = self.feature_cols[metric_name]
        else:
            # Validate that provided feature_cols match training feature_cols (if available)
            if self.feature_cols[metric_name] is not None:
                training_feature_cols = self.feature_cols[metric_name]
                if feature_cols != training_feature_cols:
                    raise ValueError(
                        f"Feature columns mismatch for metric '{metric_name}'. "
                        f"Training used: {training_feature_cols}, but got: {feature_cols}"
                    )
        
        # Validate columns
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")
        
        # Extract features in the same order as training
        X = df.select(feature_cols).to_numpy()
        
        # Get model and predict all percentiles manually since the method doesn't exist
        model = self.models[metric_name]
        n_samples = X.shape[0]
        n_percentiles = len(model.percentiles_)
        predictions = np.zeros((n_samples, n_percentiles))
        
        for i, percentile in enumerate(model.percentiles_):
            predictions[:, i] = model.predict(X, percentile)
        
        return predictions
    
    def find_closest_percentile(
        self,
        feature_values: dict,
        target_value: float,
        metric_name: str
    ) -> int:
        """
        Find the percentile whose prediction is closest to the target value.
        
        Args:
            feature_values: Dictionary mapping feature names to values
            target_value: Target value to match
            metric_name: Name of the fitted metric to use
            
        Returns:
            Closest percentile (int)
        """
        if metric_name not in self.fitted_metrics:
            raise ValueError(f"Metric '{metric_name}' has not been fitted. Available metrics: {list(self.fitted_metrics)}")
        
        # Get the training feature columns for this metric
        if self.feature_cols[metric_name] is None:
            raise ValueError(
                f"No feature column information available for metric '{metric_name}'. "
                "This may be an older model file. Please retrain the model or provide a newer model file."
            )
        training_feature_cols = self.feature_cols[metric_name]
        
        # Validate that all required features are provided
        missing_features = [col for col in training_feature_cols if col not in feature_values]
        if missing_features:
            raise ValueError(f"Missing feature values for metric '{metric_name}': {missing_features}")
        
        # Create feature array in the same order as training
        feature_array = np.array([feature_values[col] for col in training_feature_cols])
        
        return self.models[metric_name].find_closest_percentile(feature_array, target_value)
    
    def save(self, filepath: str) -> None:
        """
        Save all fitted models to a single HDF5 file.
        
        Args:
            filepath: Path where to save the models
            
        Example:
            >>> qr_df.save("my_quantile_models.h5")
        """
        if not self.fitted_metrics:
            raise ValueError("No models have been fitted. Nothing to save.")
        
        filepath = Path(filepath)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")
        
        with h5py.File(filepath, "w") as f:
            # Save global configuration
            config_group = f.create_group("global_config")
            config_group.attrs["alpha"] = self.alpha
            config_group.attrs["solver"] = self.solver
            config_group.attrs["fit_intercept"] = self.fit_intercept
            config_group.attrs["n_metrics"] = len(self.fitted_metrics)
            
            # Save list of fitted metrics
            metrics_array = np.array(list(self.fitted_metrics), dtype='S')
            f.create_dataset("fitted_metrics", data=metrics_array)
            
            # Save each model
            for metric_name, model in self.models.items():
                metric_group = f.create_group(f"metric_{metric_name}")
                
                # Save model parameters
                params_group = metric_group.create_group("parameters")
                config_group_metric = metric_group.create_group("config")
                metadata_group = metric_group.create_group("metadata")
                
                # Save coefficients for each percentile
                coeffs_group = params_group.create_group("coefficients")
                intercepts_group = params_group.create_group("intercepts")
                
                for percentile in model.percentiles_:
                    coeffs_group.create_dataset(
                        f"percentile_{percentile}",
                        data=model.coefficients_[percentile]
                    )
                    intercepts_group.create_dataset(
                        f"percentile_{percentile}",
                        data=model.intercepts_[percentile]
                    )
                
                # Save model configuration
                config_group_metric.attrs["alpha"] = model.alpha
                config_group_metric.attrs["solver"] = model.solver
                config_group_metric.attrs["fit_intercept"] = model.fit_intercept
                config_group_metric.attrs["n_features"] = model.n_features_
                
                # Save percentiles array
                params_group.create_dataset("percentiles", data=np.array(model.percentiles_))
                
                # Save metadata
                metadata_group.attrs["metric_name"] = metric_name
                metadata_group.attrs["model_type"] = "MultiQuantileRegressor"
                metadata_group.attrs["version"] = "1.0"
                metadata_group.attrs["fitted"] = True
                
                # Save feature column names used during training
                feature_cols_bytes = [col.encode('utf-8') for col in self.feature_cols[metric_name]]
                metadata_group.create_dataset("feature_cols", data=feature_cols_bytes)
        
        print(f"Saved {len(self.fitted_metrics)} models to {filepath}")
        print(f"  Metrics: {', '.join(self.fitted_metrics)}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
    
    def load(self, filepath: str) -> "QuantileRegressorDataFrame":
        """
        Load all models from an HDF5 file.
        
        Args:
            filepath: Path to the saved models file
            
        Returns:
            self: Loaded instance
            
        Example:
            >>> qr_df = QuantileRegressorDataFrame()
            >>> qr_df.load("my_quantile_models.h5")
        """
        filepath = Path(filepath)
        
        with h5py.File(filepath, "r") as f:
            # Load global configuration
            global_config = f["global_config"]
            self.alpha = global_config.attrs["alpha"]
            self.solver = global_config.attrs["solver"]
            self.fit_intercept = global_config.attrs["fit_intercept"]
            
            # Load fitted metrics list
            fitted_metrics_data = f["fitted_metrics"][:]
            fitted_metrics = [metric.decode('utf-8') for metric in fitted_metrics_data]
            
            # Load each model
            self.models = {}
            self.fitted_metrics = set()
            self.feature_cols = {}  # Initialize feature_cols dictionary
            
            for metric_name in fitted_metrics:
                metric_group = f[f"metric_{metric_name}"]
                
                # Create model
                model = MultiQuantileRegressor(
                    metric_name=metric_name,
                    alpha=self.alpha,
                    solver=self.solver,
                    fit_intercept=self.fit_intercept
                )
                
                # Load model configuration
                config_group = metric_group["config"]
                model.alpha = config_group.attrs["alpha"]
                model.solver = config_group.attrs["solver"]
                model.fit_intercept = config_group.attrs["fit_intercept"]
                model.n_features_ = config_group.attrs["n_features"]
                
                # Load percentiles
                params_group = metric_group["parameters"]
                model.percentiles_ = params_group["percentiles"][:].tolist()
                
                # Load coefficients and intercepts
                coeffs_group = params_group["coefficients"]
                intercepts_group = params_group["intercepts"]
                
                model.coefficients_ = {}
                model.intercepts_ = {}
                
                for percentile in model.percentiles_:
                    model.coefficients_[percentile] = coeffs_group[f"percentile_{percentile}"][:]
                    model.intercepts_[percentile] = intercepts_group[f"percentile_{percentile}"][()]
                
                # Set fitted status
                model.is_fitted_ = True
                
                # Load feature column names if available
                metadata_group = metric_group["metadata"]
                if "feature_cols" in metadata_group:
                    feature_cols_bytes = metadata_group["feature_cols"][:]
                    feature_cols = [col.decode('utf-8') for col in feature_cols_bytes]
                    self.feature_cols[metric_name] = feature_cols
                else:
                    # For backward compatibility with older model files
                    print(f"Warning: No feature column information found for metric '{metric_name}'. "
                          "You will need to provide feature_cols explicitly during prediction.")
                    self.feature_cols[metric_name] = None
                
                # Store model
                self.models[metric_name] = model
                self.fitted_metrics.add(metric_name)
        
        print(f"Loaded {len(self.fitted_metrics)} models from {filepath}")
        print(f"  Metrics: {', '.join(self.fitted_metrics)}")
        
        return self
    
    @staticmethod
    def load_from_file(filepath: str) -> "QuantileRegressorDataFrame":
        """
        Load models from file (static method).
        
        Args:
            filepath: Path to the saved models file
            
        Returns:
            Loaded QuantileRegressorDataFrame instance
        """
        instance = QuantileRegressorDataFrame()
        return instance.load(filepath)
    
    def get_fitted_metrics(self) -> List[str]:
        """
        Get list of fitted metric names.
        
        Returns:
            List of fitted metric names
        """
        return list(self.fitted_metrics)
    
    def get_model_info(self, metric_name: str) -> dict:
        """
        Get information about a specific fitted model.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary containing model information
        """
        if metric_name not in self.fitted_metrics:
            raise ValueError(f"Metric '{metric_name}' has not been fitted. Available metrics: {list(self.fitted_metrics)}")
        
        model = self.models[metric_name]
        
        return {
            "model_type": "MultiQuantileRegressor",
            "metric_name": metric_name,
            "n_percentiles": len(model.percentiles_),
            "percentile_range": f"{min(model.percentiles_)}-{max(model.percentiles_)}",
            "n_features": model.n_features_,
            "alpha": model.alpha,
            "solver": model.solver,
            "fit_intercept": model.fit_intercept,
            "is_fitted": model.is_fitted_
        }
    
    def get_all_models_info(self) -> dict:
        """
        Get information about all fitted models.
        
        Returns:
            Dictionary mapping metric names to model information
        """
        return {metric_name: self.get_model_info(metric_name) for metric_name in self.fitted_metrics}

