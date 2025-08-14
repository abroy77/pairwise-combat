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
    def sample_data_2d(self):
        """Create multi-dimensional sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        X1 = np.linspace(0, 10, n_samples)
        X2 = np.linspace(-5, 5, n_samples)
        X = np.column_stack([X1, X2])
        # True function: y = 2*X1 + 0.5*X2 + 1
        true_y = 2 * X1 + 0.5 * X2 + 1
        noise_std = 0.5 + 0.1 * np.abs(X1)
        y = true_y + np.random.normal(0, noise_std, n_samples)
        return X, y

    @pytest.fixture
    def fitted_model(self, sample_data):
        """Create a fitted model for testing."""
        X, y = sample_data
        model = MultiQuantileRegressor(metric_name="test_metric", alpha=0.1)
        model.fit(X, y)
        return model

    @pytest.fixture
    def fitted_model_2d(self, sample_data_2d):
        """Create a fitted 2D model for testing."""
        X, y = sample_data_2d
        model = MultiQuantileRegressor(metric_name="test_metric", alpha=0.1)
        model.fit(X, y)
        return model

    def test_initialization(self):
        """Test MultiQuantileRegressor initialization."""
        model = MultiQuantileRegressor(
            metric_name="test_metric", alpha=0.5, solver="highs-ds", fit_intercept=False
        )

        assert model.alpha == 0.5
        assert model.solver == "highs-ds"
        assert model.fit_intercept is False
        assert not model.is_fitted_
        assert len(model.percentiles_) == 99
        assert model.percentiles_[0] == 1
        assert model.percentiles_[-1] == 99

    def test_fit(self, sample_data, sample_data_2d):
        """Test fitting the model with both 1D and 2D features."""
        # Test 1D features
        X, y = sample_data
        model = MultiQuantileRegressor(metric_name="test_metric", alpha=0.1)

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

        # Test 2D features
        X_2d, y_2d = sample_data_2d
        model_2d = MultiQuantileRegressor(metric_name="test_metric", alpha=0.1)

        result_2d = model_2d.fit(X_2d, y_2d)

        # Check that fit returns self
        assert result_2d is model_2d

        # Check that model is fitted
        assert model_2d.is_fitted_
        assert model_2d.n_features_ == 2

        # Check that coefficients and intercepts are stored
        assert len(model_2d.coefficients_) == 99
        assert len(model_2d.intercepts_) == 99

        # Check that all percentiles have coefficients
        for percentile in model_2d.percentiles_:
            assert percentile in model_2d.coefficients_
            assert percentile in model_2d.intercepts_
            assert model_2d.coefficients_[percentile].shape == (2,)  # 2 features
            assert isinstance(model_2d.intercepts_[percentile], (float, np.floating))

    def test_predict(self, fitted_model, fitted_model_2d, sample_data, sample_data_2d):
        """Test prediction for specific percentiles."""
        # Test 1D prediction
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

        # Test 2D prediction
        X_2d, _ = sample_data_2d

        # Test prediction for median
        predictions_2d = fitted_model_2d.predict(X_2d, 50)
        assert predictions_2d.shape == (X_2d.shape[0],)
        assert all(np.isfinite(predictions_2d))

        # Test prediction for different percentiles
        pred_5_2d = fitted_model_2d.predict(X_2d, 5)
        pred_95_2d = fitted_model_2d.predict(X_2d, 95)

        # Higher percentiles should generally give higher predictions
        assert np.mean(pred_95_2d) > np.mean(pred_5_2d)

    def test_predict_single(self, fitted_model, fitted_model_2d):
        """Test single value prediction."""
        # Test 1D single prediction
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

        # Test 2D single prediction
        x_value_2d = np.array([5.0, 2.0])

        # Test prediction for different percentiles
        pred_25_2d = fitted_model_2d.predict_single(x_value_2d, 25)
        pred_50_2d = fitted_model_2d.predict_single(x_value_2d, 50)
        pred_75_2d = fitted_model_2d.predict_single(x_value_2d, 75)

        assert isinstance(pred_25_2d, float)
        assert isinstance(pred_50_2d, float)
        assert isinstance(pred_75_2d, float)

        # Predictions should be reasonably ordered
        assert pred_25_2d <= pred_75_2d  # 25th should be <= 75th percentile

    def test_predict_without_fit(self, sample_data):
        """Test that predict fails without fitting first."""
        X, _ = sample_data
        model = MultiQuantileRegressor(metric_name="test_metric")

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X, 50)

        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict_single(5.0, 50)


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

    def test_feature_mismatch_errors(self, fitted_model, fitted_model_2d):
        """Test that feature mismatch raises appropriate errors."""
        # Test predicting with wrong number of features
        X_wrong = np.random.randn(10, 3)  # 3 features, but models expect 1 or 2

        with pytest.raises(
            ValueError,
            match="Input has 3 features, but model was fitted with 1 features",
        ):
            fitted_model.predict(X_wrong, 50)

        with pytest.raises(
            ValueError,
            match="Input has 3 features, but model was fitted with 2 features",
        ):
            fitted_model_2d.predict(X_wrong, 50)

        # Test predict_single with wrong number of features
        with pytest.raises(
            ValueError, match="Model expects 2 features, but got single value"
        ):
            fitted_model_2d.predict_single(5.0, 50)  # 2D model expects array

        with pytest.raises(ValueError, match="Model expects 2 features, but got 3"):
            fitted_model_2d.predict_single(
                np.array([1, 2, 3]), 50
            )  # Wrong number of features

    

    def test_save_load_model(self, fitted_model, fitted_model_2d):
        """Test saving and loading the model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test 1D model
            model_path_1d = Path(tmp_dir) / "test_quantile_model_1d.h5"

            # Save model
            fitted_model.save_model(str(model_path_1d))

            # Load model
            loaded_model_1d = MultiQuantileRegressor(metric_name="test_metric")
            loaded_model_1d.load_model(str(model_path_1d))

            # Check that model is loaded correctly
            assert loaded_model_1d.is_fitted_
            assert loaded_model_1d.alpha == fitted_model.alpha
            assert loaded_model_1d.solver == fitted_model.solver
            assert loaded_model_1d.fit_intercept == fitted_model.fit_intercept
            assert loaded_model_1d.n_features_ == fitted_model.n_features_
            assert loaded_model_1d.percentiles_ == fitted_model.percentiles_

            # Check that coefficients and intercepts match
            for percentile in fitted_model.percentiles_:
                np.testing.assert_array_almost_equal(
                    loaded_model_1d.coefficients_[percentile],
                    fitted_model.coefficients_[percentile],
                )
                np.testing.assert_almost_equal(
                    loaded_model_1d.intercepts_[percentile],
                    fitted_model.intercepts_[percentile],
                )

            # Test that loaded model produces same predictions
            X_test_1d = np.array([[1], [2], [3]])
            original_pred_1d = fitted_model.predict(X_test_1d, 50)
            loaded_pred_1d = loaded_model_1d.predict(X_test_1d, 50)

            np.testing.assert_array_almost_equal(original_pred_1d, loaded_pred_1d)

            # Test 2D model
            model_path_2d = Path(tmp_dir) / "test_quantile_model_2d.h5"

            # Save model
            fitted_model_2d.save_model(str(model_path_2d))

            # Load model
            loaded_model_2d = MultiQuantileRegressor()
            loaded_model_2d.load_model(str(model_path_2d))

            # Check that model is loaded correctly
            assert loaded_model_2d.is_fitted_
            assert loaded_model_2d.alpha == fitted_model_2d.alpha
            assert loaded_model_2d.solver == fitted_model_2d.solver
            assert loaded_model_2d.fit_intercept == fitted_model_2d.fit_intercept
            assert loaded_model_2d.n_features_ == fitted_model_2d.n_features_ == 2
            assert loaded_model_2d.percentiles_ == fitted_model_2d.percentiles_

            # Check that coefficients and intercepts match
            for percentile in fitted_model_2d.percentiles_:
                np.testing.assert_array_almost_equal(
                    loaded_model_2d.coefficients_[percentile],
                    fitted_model_2d.coefficients_[percentile],
                )
                np.testing.assert_almost_equal(
                    loaded_model_2d.intercepts_[percentile],
                    fitted_model_2d.intercepts_[percentile],
                )

            # Test that loaded model produces same predictions
            X_test_2d = np.array([[1, 0.5], [2, 1.0], [3, 1.5]])
            original_pred_2d = fitted_model_2d.predict(X_test_2d, 50)
            loaded_pred_2d = loaded_model_2d.predict(X_test_2d, 50)

            np.testing.assert_array_almost_equal(original_pred_2d, loaded_pred_2d)

    def test_find_closest_percentile(self, fitted_model, fitted_model_2d):
        """Test finding closest percentile for both 1D and 2D models."""
        # Test 1D model
        X_single_1d = np.array([5.0])
        y_value = fitted_model.predict_single(5.0, 50)
        closest_percentile_1d = fitted_model.find_closest_percentile(
            X_single_1d, y_value
        )

        # Should be close to 50th percentile since we used 50th percentile prediction
        assert abs(closest_percentile_1d - 50) <= 10  # Allow some tolerance

        # Test 2D model
        X_single_2d = np.array([5.0, 2.0])
        y_value_2d = fitted_model_2d.predict_single(X_single_2d, 50)
        closest_percentile_2d = fitted_model_2d.find_closest_percentile(
            X_single_2d, y_value_2d
        )

        # Should be close to 50th percentile since we used 50th percentile prediction
        assert abs(closest_percentile_2d - 50) <= 10  # Allow some tolerance
