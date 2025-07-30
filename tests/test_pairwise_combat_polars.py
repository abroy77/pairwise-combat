"""
Tests for the Polars DataFrame interface to PairwiseComBAT.
"""

import pytest
import polars as pl
import numpy as np
import tempfile
from pathlib import Path
from sklearn.metrics import mean_squared_error
from tests.sites import FeatureProperties, Site

from pairwise_combat.polars_interface import PairwiseComBATDataFrame, create_example_dataframe

# Test constants  
MIN_AGE = 18
MAX_AGE = 90
N_SAMPLES = 100


# Fixtures from original test file
@pytest.fixture
def feature_property_fixture_2D():
    """Fixture for a simple FeatureProperties instance."""
    # num_loc = 2
    # num_cont_covars = 2
    alphas = np.array([0.5, 0.2])
    betas = np.array([[0.04, -0.03], [0.09, 0.02]])
    sigmas = np.array([0.2, 0.3])
    feature_props = FeatureProperties(alphas, betas, sigmas)
    assert isinstance(feature_props, FeatureProperties), \
        "feature_props must be an instance of FeatureProperties"
    return feature_props


@pytest.fixture
def feature_property_fixture_scalar():
    """Fixture for a FeatureProperties instance with scalar parameters."""
    alphas = 0.5
    betas = 0.04
    sigmas = 0.5
    return FeatureProperties(alphas, betas, sigmas)


@pytest.fixture
def ref_site_1d(feature_property_fixture_scalar):
    """Reference site fixture for harmonization tests (1D: 1 region, 1 feature)."""
    return Site(gamma=0.3, delta=0.2, feature_properties=feature_property_fixture_scalar)


@pytest.fixture
def test_site_1d(feature_property_fixture_scalar):
    """Test site fixture for harmonization tests (1D: 1 region, 1 feature), with unique alphas and betas."""
    return Site(gamma=0.5, delta=0.5, feature_properties=feature_property_fixture_scalar)


@pytest.fixture
def ref_site_2d(feature_property_fixture_2D):
    """Reference site fixture for harmonization tests (2D: 2 regions, 2 features)."""
    return Site(gamma=[0.3, 0.4], delta=[0.2, 0.3], feature_properties=feature_property_fixture_2D)


@pytest.fixture
def test_site_2d(feature_property_fixture_2D):
    """Test site fixture for harmonization tests (2D: 2 regions, 2 features), with unique alphas and betas."""
    return Site(gamma=[0.5, 0.6], delta=[0.5, 0.6], feature_properties=feature_property_fixture_2D)


@pytest.fixture
def baseline_and_covars_1d():
    """Fixture for baseline noise and covariates for 1D data generation."""
    n_samples = N_SAMPLES
    covars = np.random.uniform(0, 100, n_samples).reshape(1, -1)
    baseline_noise = np.random.normal(0, 1, n_samples).reshape(1, -1)
    return baseline_noise, covars


@pytest.fixture
def baseline_and_covars_2d():
    """Fixture for baseline noise and covariates for 2D data generation."""
    n_samples = N_SAMPLES
    covar1 = np.random.uniform(0, 100, n_samples).reshape(1, -1)
    covar2 = np.random.randint(0, 2, n_samples).reshape(1, -1)
    covars = np.vstack((covar1, covar2))  # Combine into 2D array
    baseline_noise = np.random.normal(0, 1, n_samples * 2).reshape(2, -1)
    return baseline_noise, covars


@pytest.fixture
def dataframe_1d(ref_site_1d, test_site_1d, baseline_and_covars_1d):
    """Create a DataFrame with 1D sites for testing."""
    baseline_data, covars = baseline_and_covars_1d
    
    # Generate data for both sites
    ref_data = ref_site_1d.generate_site_data(baseline_data, covars)
    test_data = test_site_1d.generate_site_data(baseline_data, covars)
    
    # Create DataFrame for reference site
    ref_df = pl.DataFrame({
        "site_id": ["ref_site"] * N_SAMPLES,
        "age": covars.flatten(),
        "voxel_1": ref_data.flatten(),
        "sample_id": range(N_SAMPLES)
    })
    
    # Create DataFrame for test site
    test_df = pl.DataFrame({
        "site_id": ["test_site"] * N_SAMPLES,
        "age": covars.flatten(),
        "voxel_1": test_data.flatten(),
        "sample_id": range(N_SAMPLES, 2 * N_SAMPLES)
    })
    
    # Combine and shuffle
    combined_df = pl.concat([ref_df, test_df], how="vertical")
    return combined_df.sample(fraction=1.0, seed=42)


@pytest.fixture
def dataframe_2d(ref_site_2d, test_site_2d, baseline_and_covars_2d):
    """Create a DataFrame with 2D sites for testing."""
    baseline_data, covars = baseline_and_covars_2d
    
    # Generate data for both sites
    ref_data = ref_site_2d.generate_site_data(baseline_data, covars)
    test_data = test_site_2d.generate_site_data(baseline_data, covars)
    
    # Create DataFrame for reference site
    ref_df = pl.DataFrame({
        "site_id": ["ref_site"] * N_SAMPLES,
        "age": covars[0, :].flatten(),
        "sex": covars[1, :].flatten(),
        "voxel_1": ref_data[0, :].flatten(),
        "voxel_2": ref_data[1, :].flatten(),
        "sample_id": range(N_SAMPLES)
    })
    
    # Create DataFrame for test site
    test_df = pl.DataFrame({
        "site_id": ["test_site"] * N_SAMPLES,
        "age": covars[0, :].flatten(),
        "sex": covars[1, :].flatten(),
        "voxel_1": test_data[0, :].flatten(),
        "voxel_2": test_data[1, :].flatten(),
        "sample_id": range(N_SAMPLES, 2 * N_SAMPLES)
    })
    
    # Combine and shuffle
    combined_df = pl.concat([ref_df, test_df], how="vertical")
    return combined_df.sample(fraction=1.0, seed=42)


class TestPairwiseComBATDataFrame:
    """Test the DataFrame interface for PairwiseComBAT."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return create_example_dataframe(
            n_samples_per_site=30,
            n_locations=3,
            n_covariates=2,
            source_site="site_A",
            target_site="site_B",
            random_seed=42
        )
    
    @pytest.fixture
    def combat_df(self):
        """Create a PairwiseComBATDataFrame instance."""
        return PairwiseComBATDataFrame(
            source_site="site_A",
            target_site="site_B",
            max_iter=10,
            tol=1e-4
        )
    
    def test_fit(self, combat_df, sample_dataframe):
        """Test fitting the model with DataFrame data."""
        result = combat_df.fit(
            df=sample_dataframe,
            site_id_col="site_id",
            data_cols=["voxel_1", "voxel_2", "voxel_3"],
            covariate_cols=["age", "sex"]
        )
        
        # Check that fit returns self
        assert result is combat_df
        
        # Check that model is fitted
        assert combat_df.is_fitted_
        assert combat_df.model.is_fitted_
        
        # Check underlying model parameters
        assert combat_df.model.n_locations == 3
        assert combat_df.model.n_covariates == 2
        assert combat_df.model.alpha_hat_ is not None
        assert combat_df.model.beta_hat_ is not None
        assert combat_df.model.sigma_hat_ is not None
    
    def test_transform_after_fit(self, combat_df, sample_dataframe):
        """Test transforming data after fitting."""
        # Fit the model
        combat_df.fit(
            df=sample_dataframe,
            site_id_col="site_id",
            data_cols=["voxel_1", "voxel_2", "voxel_3"],
            covariate_cols=["age", "sex"]
        )
        
        # Transform the data
        harmonized = combat_df.transform(sample_dataframe)
        
        # Check basic properties
        assert isinstance(harmonized, pl.DataFrame)
        assert harmonized.height == sample_dataframe.height
        assert set(harmonized.columns) == set(sample_dataframe.columns)
        
        # Check that only source site data was harmonized
        source_original = sample_dataframe.filter(pl.col("site_id") == "site_A")
        source_harmonized = harmonized.filter(pl.col("site_id") == "site_A")
        
        # Data should be different after harmonization
        original_voxel_1 = source_original.select("voxel_1").to_numpy().flatten()
        harmonized_voxel_1 = source_harmonized.select("voxel_1").to_numpy().flatten()
        
        assert not np.allclose(original_voxel_1, harmonized_voxel_1)
    
    
    def test_save_load_model(self, combat_df, sample_dataframe):
        """Test saving and loading the model."""
        # Fit the model
        combat_df.fit(
            df=sample_dataframe,
            site_id_col="site_id",
            data_cols=["voxel_1", "voxel_2", "voxel_3"],
            covariate_cols=["age", "sex"]
        )
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.h5"
            combat_df.save_model(str(model_path))
            
            # Create new instance and load
            new_combat = PairwiseComBATDataFrame(
                source_site="site_A",
                target_site="site_B"
            )
            new_combat.load_model(str(model_path))
            
            # Check that model is loaded
            assert new_combat.is_fitted_
            assert new_combat.model.is_fitted_
            
            # Test that loaded model produces same results
            original_transform = combat_df.transform(sample_dataframe)
            loaded_transform = new_combat.transform(
                sample_dataframe,
                site_id_col="site_id",
                data_cols=["voxel_1", "voxel_2", "voxel_3"],
                covariate_cols=["age", "sex"]
            )
            
            # Results should be very close (allowing for numerical precision)
            original_data = original_transform.select(["voxel_1", "voxel_2", "voxel_3"]).to_numpy()
            loaded_data = loaded_transform.select(["voxel_1", "voxel_2", "voxel_3"]).to_numpy()
            
            np.testing.assert_allclose(original_data, loaded_data, rtol=1e-10)
    

 
    def test_harmonization_1d_with_sites(self, dataframe_1d):
        """Test harmonization with 1D Site fixtures to verify MSE reduction."""
        combat = PairwiseComBATDataFrame(
            source_site="test_site",
            target_site="ref_site",
            max_iter=30,
            tol=1e-6
        )
        
        # Fit and transform
        harmonized_df = combat.fit_transform(
            df=dataframe_1d,
            site_id_col="site_id",
            data_cols=["voxel_1"],
            covariate_cols=["age"]
        )
        
        # Extract original and harmonized data for MSE calculation
        ref_data = dataframe_1d.filter(pl.col("site_id") == "ref_site")["voxel_1"].to_numpy()
        test_original = dataframe_1d.filter(pl.col("site_id") == "test_site")["voxel_1"].to_numpy()
        test_harmonized = harmonized_df.filter(pl.col("site_id") == "test_site")["voxel_1"].to_numpy()
        
        # Calculate MSE before and after harmonization
        mse_pre_combat = mean_squared_error(ref_data, test_original)
        mse_post_combat = mean_squared_error(ref_data, test_harmonized)
        
        # Harmonization should reduce MSE
        assert mse_post_combat < mse_pre_combat, \
            f"MSE not reduced: before={mse_pre_combat:.4f}, after={mse_post_combat:.4f}"
        
        # Calculate improvement percentage
        improvement = (mse_pre_combat - mse_post_combat) / mse_pre_combat * 100
        assert improvement > 10, f"Improvement {improvement:.2f}% is too small"
        
        print(f"1D Harmonization: MSE reduced from {mse_pre_combat:.4f} to {mse_post_combat:.4f} "
              f"({improvement:.2f}% improvement)")
    
    def test_harmonization_2d_with_sites(self, dataframe_2d):
        """Test harmonization with 2D Site fixtures to verify MSE reduction."""
        combat = PairwiseComBATDataFrame(
            source_site="test_site",
            target_site="ref_site", 
            max_iter=30,
            tol=1e-6
        )
        
        # Fit and transform
        harmonized_df = combat.fit_transform(
            df=dataframe_2d,
            site_id_col="site_id",
            data_cols=["voxel_1", "voxel_2"],
            covariate_cols=["age", "sex"]
        )
        
        # Extract data for MSE calculation
        ref_data = dataframe_2d.filter(pl.col("site_id") == "ref_site").select(["voxel_1", "voxel_2"]).to_numpy()
        test_original = dataframe_2d.filter(pl.col("site_id") == "test_site").select(["voxel_1", "voxel_2"]).to_numpy()
        test_harmonized = harmonized_df.filter(pl.col("site_id") == "test_site").select(["voxel_1", "voxel_2"]).to_numpy()
        
        # Calculate MSE before and after harmonization
        mse_pre_combat = mean_squared_error(ref_data.flatten(), test_original.flatten())
        mse_post_combat = mean_squared_error(ref_data.flatten(), test_harmonized.flatten())
        
        # Harmonization should reduce MSE
        assert mse_post_combat < mse_pre_combat, \
            f"MSE not reduced: before={mse_pre_combat:.4f}, after={mse_post_combat:.4f}"
        
        # Calculate improvement percentage
        improvement = (mse_pre_combat - mse_post_combat) / mse_pre_combat * 100
        assert improvement > 10, f"Improvement {improvement:.2f}% is too small"
        
        print(f"2D Harmonization: MSE reduced from {mse_pre_combat:.4f} to {mse_post_combat:.4f} "
              f"({improvement:.2f}% improvement)")
