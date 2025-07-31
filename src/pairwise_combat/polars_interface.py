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

from .core import PairwiseComBAT


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
