"""
Tests for the QuantileRegressorDataFrame class.
"""

import pytest
import numpy as np
import polars as pl
import tempfile
from pathlib import Path

from pairwise_combat.polars_interface import QuantileRegressorDataFrame


class TestQuantileRegressorDataFrame:
    """Test the QuantileRegressorDataFrame class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create features
        age = np.random.uniform(20, 80, n_samples)
        sex = np.random.randint(0, 2, n_samples)
        education = np.random.uniform(8, 20, n_samples)
        
        # Create multiple target metrics with different relationships
        cortical_thickness = 2.5 + 0.02 * age + 0.1 * sex + 0.05 * education + np.random.normal(0, 0.2, n_samples)
        volume = 1000 + 5 * age + 20 * sex + 10 * education + np.random.normal(0, 50, n_samples)
        
        df = pl.DataFrame({
            "age": age,
            "sex": sex,
            "education": education,
            "cortical_thickness": cortical_thickness,
            "volume": volume
        })
        
        return df
    
    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted model for testing."""
        qr_df = QuantileRegressorDataFrame(alpha=0.1)
        
        # Fit on cortical thickness
        qr_df.fit(
            df=sample_data,
            feature_cols=["age", "sex", "education"],
            target_col="cortical_thickness",
            metric_name="cortical_thickness"
        )
        
        # Fit on volume
        qr_df.fit(
            df=sample_data,
            feature_cols=["age", "sex", "education"],
            target_col="volume",
            metric_name="volume"
        )
        
        return qr_df
    
    def test_initialization(self):
        """Test QuantileRegressorDataFrame initialization."""
        qr_df = QuantileRegressorDataFrame(alpha=0.5, solver="highs-ds", fit_intercept=False)
        
        assert qr_df.alpha == 0.5
        assert qr_df.solver == "highs-ds"
        assert qr_df.fit_intercept is False
        assert len(qr_df.models) == 0
        assert len(qr_df.fitted_metrics) == 0
    
    def test_fit_single_metric(self, sample_data):
        """Test fitting a single metric."""
        qr_df = QuantileRegressorDataFrame(alpha=0.1)
        
        result = qr_df.fit(
            df=sample_data,
            feature_cols=["age", "sex", "education"],
            target_col="cortical_thickness"
        )
        
        # Check that fit returns self
        assert result is qr_df
        
        # Check that model is fitted
        assert "cortical_thickness" in qr_df.fitted_metrics
        assert "cortical_thickness" in qr_df.models
        assert qr_df.models["cortical_thickness"].is_fitted_
        assert qr_df.models["cortical_thickness"].n_features_ == 3
    
    def test_fit_multiple_metrics(self, sample_data):
        """Test fitting multiple metrics."""
        qr_df = QuantileRegressorDataFrame(alpha=0.1)
        
        # Fit cortical thickness
        qr_df.fit(
            df=sample_data,
            feature_cols=["age", "sex", "education"],
            target_col="cortical_thickness",
            metric_name="cortical_thickness"
        )
        
        # Fit volume
        qr_df.fit(
            df=sample_data,
            feature_cols=["age", "sex"],  # Different features
            target_col="volume",
            metric_name="volume"
        )
        
        # Check both models are fitted
        assert len(qr_df.fitted_metrics) == 2
        assert "cortical_thickness" in qr_df.fitted_metrics
        assert "volume" in qr_df.fitted_metrics
        
        # Check different feature dimensions
        assert qr_df.models["cortical_thickness"].n_features_ == 3
        assert qr_df.models["volume"].n_features_ == 2
    
    def test_predict(self, fitted_model, sample_data):
        """Test prediction for specific percentiles."""
        # Test cortical thickness prediction
        predictions = fitted_model.predict(
            df=sample_data[:10],
            feature_cols=["age", "sex", "education"],
            metric_name="cortical_thickness",
            percentile=50
        )
        
        assert predictions.shape == (10,)
        assert all(np.isfinite(predictions))
        
        # Test volume prediction
        predictions_vol = fitted_model.predict(
            df=sample_data[:10],
            feature_cols=["age", "sex", "education"],
            metric_name="volume",
            percentile=75
        )
        
        assert predictions_vol.shape == (10,)
        assert all(np.isfinite(predictions_vol))
    
    def test_predict_single(self, fitted_model):
        """Test single value prediction."""
        feature_values = {"age": 65, "sex": 1, "education": 16}
        
        # Test cortical thickness
        pred_ct = fitted_model.predict_single(
            feature_values=feature_values,
            metric_name="cortical_thickness",
            percentile=50
        )
        
        assert isinstance(pred_ct, float)
        assert np.isfinite(pred_ct)
        
        # Test volume
        pred_vol = fitted_model.predict_single(
            feature_values=feature_values,
            metric_name="volume",
            percentile=25
        )
        
        assert isinstance(pred_vol, float)
        assert np.isfinite(pred_vol)
    
    def test_predict_all_percentiles(self, fitted_model, sample_data):
        """Test prediction for all percentiles."""
        predictions = fitted_model.predict_all_percentiles(
            df=sample_data[:5],
            feature_cols=["age", "sex", "education"],
            metric_name="cortical_thickness"
        )
        
        assert predictions.shape == (5, 99)  # 5 samples, 99 percentiles
        
        # Check that predictions are reasonably ordered
        for i in range(5):
            sample_preds = predictions[i, :]
            assert sample_preds[4] <= sample_preds[94]  # 5th <= 95th percentile
    
    def test_find_closest_percentile(self, fitted_model):
        """Test finding closest percentile."""
        feature_values = {"age": 50, "sex": 0, "education": 12}
        
        # Get prediction for 50th percentile
        pred_50 = fitted_model.predict_single(
            feature_values=feature_values,
            metric_name="cortical_thickness",
            percentile=50
        )
        
        # Find closest percentile to this prediction
        closest = fitted_model.find_closest_percentile(
            feature_values=feature_values,
            target_value=pred_50,
            metric_name="cortical_thickness"
        )
        
        # Should be close to 50
        assert abs(closest - 50) <= 10
    
    def test_save_load(self, fitted_model, sample_data):
        """Test saving and loading multiple models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_quantile_models.h5"
            
            # Save models
            fitted_model.save(str(model_path))
            
            # Load models
            loaded_model = QuantileRegressorDataFrame()
            loaded_model.load(str(model_path))
            
            # Check that models are loaded correctly
            assert loaded_model.fitted_metrics == fitted_model.fitted_metrics
            assert loaded_model.alpha == fitted_model.alpha
            assert loaded_model.solver == fitted_model.solver
            assert loaded_model.fit_intercept == fitted_model.fit_intercept
            
            # Check that predictions match
            test_data = sample_data[:3]
            
            # Test cortical thickness
            original_pred = fitted_model.predict(
                df=test_data,
                feature_cols=["age", "sex", "education"],
                metric_name="cortical_thickness",
                percentile=50
            )
            loaded_pred = loaded_model.predict(
                df=test_data,
                feature_cols=["age", "sex", "education"],
                metric_name="cortical_thickness",
                percentile=50
            )
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
            # Test volume
            original_pred_vol = fitted_model.predict(
                df=test_data,
                feature_cols=["age", "sex", "education"],
                metric_name="volume",
                percentile=75
            )
            loaded_pred_vol = loaded_model.predict(
                df=test_data,
                feature_cols=["age", "sex", "education"],
                metric_name="volume",
                percentile=75
            )
            
            np.testing.assert_array_almost_equal(original_pred_vol, loaded_pred_vol)
    
    def test_static_load(self, fitted_model):
        """Test static load method."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_static_load.h5"
            
            # Save models
            fitted_model.save(str(model_path))
            
            # Load using static method
            loaded_model = QuantileRegressorDataFrame.load_from_file(str(model_path))
            
            # Check that models are loaded correctly
            assert loaded_model.fitted_metrics == fitted_model.fitted_metrics
    
    def test_error_cases(self, sample_data):
        """Test various error conditions."""
        qr_df = QuantileRegressorDataFrame()
        
        # Test predicting without fitting
        with pytest.raises(ValueError, match="has not been fitted"):
            qr_df.predict(
                df=sample_data,
                feature_cols=["age", "sex"],
                metric_name="nonexistent",
                percentile=50
            )
        
        # Test with missing columns
        with pytest.raises(ValueError, match="Missing columns"):
            qr_df.fit(
                df=sample_data,
                feature_cols=["age", "nonexistent_col"],
                target_col="cortical_thickness"
            )
        
        # Test saving without fitted models
        qr_df_empty = QuantileRegressorDataFrame()
        with pytest.raises(ValueError, match="No models have been fitted"):
            qr_df_empty.save("test.h5")
    
    def test_model_info(self, fitted_model):
        """Test getting model information."""
        # Test single model info
        info = fitted_model.get_model_info("cortical_thickness")
        assert info["model_type"] == "MultiQuantileRegressor"
        assert info["n_features"] == 3
        assert info["is_fitted"] is True
        
        # Test all models info
        all_info = fitted_model.get_all_models_info()
        assert "cortical_thickness" in all_info
        assert "volume" in all_info
        assert len(all_info) == 2
    
    def test_get_fitted_metrics(self, fitted_model):
        """Test getting fitted metrics list."""
        metrics = fitted_model.get_fitted_metrics()
        assert len(metrics) == 2
        assert "cortical_thickness" in metrics
        assert "volume" in metrics
