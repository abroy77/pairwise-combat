"""
Tests for the MultiQuantileRegressor.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from pairwise_combat.quantile_regressor.quantile_regressor import MultiQuantileRegressor


class TestMultiQuantileRegressor:
    """Test the MultiQuantileRegressor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        X = np.linspace(0, 10, n_samples)
        # True function with increasing variance
        true_y = 2 * X + 1
        noise_std = 0.5 + 0.1 * X
        y = true_y + np.random.normal(0, noise_std, n_samples)
        return X.reshape(-1, 1), y
    
    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted model for testing."""
        X, y = sample_data
        model = MultiQuantileRegressor(alpha=0.1)
        model.fit(X, y)
        return model
    
    def test_initialization(self):
        """Test MultiQuantileRegressor initialization."""
        model = MultiQuantileRegressor(alpha=0.5, solver="highs-ds", fit_intercept=False)
        
        assert model.alpha == 0.5
        assert model.solver == "highs-ds"
        assert model.fit_intercept is False
        assert not model.is_fitted_
        assert len(model.percentiles_) == 99
        assert model.percentiles_[0] == 1
        assert model.percentiles_[-1] == 99
    
    def test_fit(self, sample_data):
        """Test fitting the model."""
        X, y = sample_data
        model = MultiQuantileRegressor(alpha=0.1)
        
        result = model.fit(X, y)
        
        # Check that fit returns self
        assert result is model
        
        # Check that model is fitted
        assert model.is_fitted_
        assert model.n_features_ == 1
        
        # Check that coefficients and intercepts are stored
        assert len(model.coefficients_) == 99
        assert len(model.intercepts_) == 99
        
        # Check that all percentiles have coefficients
        for percentile in model.percentiles_:
            assert percentile in model.coefficients_
            assert percentile in model.intercepts_
            assert model.coefficients_[percentile].shape == (1,)  # 1 feature
            assert isinstance(model.intercepts_[percentile], (float, np.floating))
    
    def test_predict(self, fitted_model, sample_data):
        """Test prediction for specific percentiles."""
        X, _ = sample_data
        
        # Test prediction for median
        predictions = fitted_model.predict(X, 50)
        assert predictions.shape == (X.shape[0],)
        assert all(np.isfinite(predictions))
        
        # Test prediction for different percentiles
        pred_5 = fitted_model.predict(X, 5)
        pred_95 = fitted_model.predict(X, 95)
        
        # Higher percentiles should generally give higher predictions
        assert np.mean(pred_95) > np.mean(pred_5)
    
    def test_predict_single(self, fitted_model):
        """Test single value prediction."""
        x_value = 5.0
        
        # Test prediction for different percentiles
        pred_25 = fitted_model.predict_single(x_value, 25)
        pred_50 = fitted_model.predict_single(x_value, 50)
        pred_75 = fitted_model.predict_single(x_value, 75)
        
        assert isinstance(pred_25, float)
        assert isinstance(pred_50, float)
        assert isinstance(pred_75, float)
        
        # Predictions should be reasonably ordered (allowing some flexibility)
        # Individual quantile regression models aren't guaranteed to be perfectly monotonic
        assert pred_25 <= pred_75  # 25th should be <= 75th percentile
    
    def test_predict_all_percentiles(self, fitted_model, sample_data):
        """Test prediction for all percentiles."""
        X, _ = sample_data
        
        predictions = fitted_model.predict_all_percentiles(X[:10])  # Test with subset
        
        assert predictions.shape == (10, 99)  # 10 samples, 99 percentiles
        
        # Check that lower percentiles tend to be lower than higher percentiles
        # (Note: individual quantile regression models may not be perfectly monotonic)
        for i in range(10):
            sample_preds = predictions[i, :]
            # Check that the 5th percentile is generally lower than the 95th
            assert sample_preds[4] <= sample_preds[94]  # 5th vs 95th percentile
            # Check that the 25th percentile is generally lower than the 75th
            assert sample_preds[24] <= sample_preds[74]  # 25th vs 75th percentile
    
    def test_predict_without_fit(self, sample_data):
        """Test that predict fails without fitting first."""
        X, _ = sample_data
        model = MultiQuantileRegressor()
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X, 50)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict_single(5.0, 50)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict_all_percentiles(X)
    
    def test_invalid_percentile(self, fitted_model, sample_data):
        """Test prediction with invalid percentile."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="Percentile 0 not available"):
            fitted_model.predict(X, 0)
        
        with pytest.raises(ValueError, match="Percentile 100 not available"):
            fitted_model.predict(X, 100)
        
        with pytest.raises(ValueError, match="Percentile 150 not available"):
            fitted_model.predict_single(5.0, 150)
    
    def test_1d_input_handling(self, fitted_model):
        """Test that 1D input is properly handled."""
        # Test with 1D array
        X_1d = np.array([1, 2, 3, 4, 5])
        predictions = fitted_model.predict(X_1d, 50)
        assert predictions.shape == (5,)
        
        # Test single value prediction
        pred = fitted_model.predict_single(3.0, 50)
        assert isinstance(pred, float)
    
    def test_save_load_model(self, fitted_model):
        """Test saving and loading the model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_quantile_model.h5"
            
            # Save model
            fitted_model.save_model(str(model_path))
            
            # Load model
            loaded_model = MultiQuantileRegressor()
            loaded_model.load_model(str(model_path))
            
            # Check that model is loaded correctly
            assert loaded_model.is_fitted_
            assert loaded_model.alpha == fitted_model.alpha
            assert loaded_model.solver == fitted_model.solver
            assert loaded_model.fit_intercept == fitted_model.fit_intercept
            assert loaded_model.n_features_ == fitted_model.n_features_
            assert loaded_model.percentiles_ == fitted_model.percentiles_
            
            # Check that coefficients and intercepts match
            for percentile in fitted_model.percentiles_:
                np.testing.assert_array_almost_equal(
                    loaded_model.coefficients_[percentile],
                    fitted_model.coefficients_[percentile]
                )
                np.testing.assert_almost_equal(
                    loaded_model.intercepts_[percentile],
                    fitted_model.intercepts_[percentile]
                )
            
            # Test that loaded model produces same predictions
            X_test = np.array([[1], [2], [3]])
            original_pred = fitted_model.predict(X_test, 50)
            loaded_pred = loaded_model.predict(X_test, 50)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
