from sklearn.metrics import mean_squared_error
import pytest
import numpy as np
from pairwise_combat.core import PairwiseComBAT
from .sites import FeatureProperties, Site

# Test constants
MIN_AGE = 18
MAX_AGE = 90


class TestFeatureProperties:
    """ "Test suite for FeatureProperties class"""

    def test_feature_properties_initialisation_scalar(self):
        alphas = 0.5
        betas = 0.04
        sigmas = 0.5
        feature_properties = FeatureProperties(alphas, betas, sigmas)
        assert feature_properties.num_locations == 1
        assert feature_properties.num_cont_covars == 1

    def test_feature_properties_initialisation_array(self):
        alphas = np.array([0.5, 0.6])
        betas = np.array([[0.04, -0.02], [0.05, 0.01]])
        sigmas = np.array([0.5, 0.6])
        feature_properties = FeatureProperties(alphas, betas, sigmas)
        assert feature_properties.num_locations == 2
        assert feature_properties.num_cont_covars == 2
        assert np.array_equal(feature_properties.alphas, alphas.reshape(-1, 1))
        assert np.array_equal(feature_properties.betas, betas)
        assert np.array_equal(feature_properties.sigmas, sigmas.reshape(-1, 1))
        assert feature_properties.num_locations == alphas.shape[0]
        assert feature_properties.num_cont_covars == betas.shape[1]
        assert feature_properties.alphas.shape == (2, 1)
        assert feature_properties.betas.shape == (2, 2)
        assert feature_properties.sigmas.shape == (2, 1)
        # Test ValueError for invalid shapes

    def test_feature_properties_invalid_shapes(self):
        # List of invalid (alphas_shape, betas_shape, sigmas_shape) configurations
        invalid_shapes = [
            # Only shapes matter, values are all ones
            ((1, 1), (1, 2), (1, 1)),
            ((1, 2), (2, 1), (1, 1)),
            ((1, 1), (2, 2), (1, 1)),
            ((1, 1), (1, 2), (1, 2)),
            ((1, 1), (1, 2), (2, 2)),
            ((1, 1), (1, 2), (2, 1)),
            ((1, 1), (1, 2), (3, 2)),
            ((1, 1), (1, 2), (3, 1)),
        ]
        for alphas_shape, betas_shape, sigmas_shape in invalid_shapes:
            alphas = np.ones(alphas_shape)
            betas = np.ones(betas_shape)
            sigmas = np.ones(sigmas_shape)
            with pytest.raises(ValueError):
                FeatureProperties(alphas, betas, sigmas)


@pytest.fixture
def feature_property_fixture_2D():
    """Fixture for a simple FeatureProperties instance."""
    # num_loc = 2
    # num_cont_covars = 2
    alphas = np.array([0.5, 0.2])
    betas = np.array([[0.04, -0.03], [0.09, 0.02]])
    sigmas = np.array([0.2, 0.3])
    feature_props = FeatureProperties(alphas, betas, sigmas)
    assert isinstance(
        feature_props, FeatureProperties
    ), "feature_props must be an instance of FeatureProperties"
    return feature_props


@pytest.fixture
def feature_property_fixture_scalar():
    """Fixture for a FeatureProperties instance with scalar parameters."""
    alphas = 0.5
    betas = 0.04
    sigmas = 0.5
    return FeatureProperties(alphas, betas, sigmas)


@pytest.fixture
def site_fixture_1d_scalar(feature_property_fixture_scalar):
    """Fixture for a 1D Site with scalar FeatureProperties."""
    gamma = 0.7
    delta = 1.5
    return Site(
        gamma=gamma, delta=delta, feature_properties=feature_property_fixture_scalar
    )


@pytest.fixture
def site_fixture_2d(feature_property_fixture_2D):
    """Fixture for a 2D Site with 2D FeatureProperties."""
    gamma = [0.3, 0.5]
    delta = [1.1, 1.3]
    return Site(
        gamma=gamma, delta=delta, feature_properties=feature_property_fixture_2D
    )


class TestDataGeneration:
    """Test suite for data generation and site initialization"""

    def test_site_initialization_scalar_feature(self, feature_property_fixture_scalar):
        """Test Site initialization for 1 location, 1 covariate using scalar FeatureProperties fixture"""
        # Use the scalar fixture for parameters
        fp = feature_property_fixture_scalar
        # Should be 1 location, 1 covariate
        assert fp.num_locations == 1
        assert fp.num_cont_covars == 1
        # Create Site using FeatureProperties values
        gamma = 0.7
        delta = 1.5
        site = Site(gamma=gamma, delta=delta, feature_properties=fp)
        # Check that gamma and delta are arrays of length 1
        assert np.array_equal(site.gamma, np.asarray([[gamma]]))
        assert np.array_equal(site.delta, np.asarray([[delta]]))
        assert site.feature_properties == fp

    def test_site_initialization_2D_feature(self, feature_property_fixture_2D):
        """Test Site initialization for 2 locations, 2 covariates using 2D FeatureProperties fixture"""
        # Use the 2D fixture for parameters
        fp = feature_property_fixture_2D
        # Should be 2 locations, 2 covariates
        assert fp.num_locations == 2
        assert fp.num_cont_covars == 2
        # Create Site using FeatureProperties values
        gamma = [0.3, 0.5]
        delta = [1.1, 1.3]
        site = Site(gamma=gamma, delta=delta, feature_properties=fp)
        # Check that gamma and delta are arrays of correct shape
        assert np.array_equal(site.gamma, np.asarray([[0.3], [0.5]]))
        assert np.array_equal(site.delta, np.asarray([[1.1], [1.3]]))
        assert site.feature_properties == fp

    def test_site_initialization_invalid_shapes(
        self, feature_property_fixture_2D, feature_property_fixture_scalar
    ):
        """Test Site initialization with invalid shapes"""
        # Use the 2D fixture for parameters
        fp = feature_property_fixture_2D
        # Should be 2 locations, 2 covariates
        assert fp.num_locations == 2
        assert fp.num_cont_covars == 2

        # Invalid gamma shape (should be 2D)
        with pytest.raises(ValueError):
            Site(gamma=[0.3, 0.5, 0.7], delta=[1.1, 1.3], feature_properties=fp)

        # Invalid delta shape (should be 2D)
        with pytest.raises(ValueError):
            Site(gamma=[0.3, 0.5], delta=[1.1], feature_properties=fp)

        with pytest.raises(ValueError):
            Site(gamma=[0.3, 0.5], delta=[1.1, 1.3, 1.4], feature_properties=fp)

    def test_site_data_generation_1D(self, site_fixture_1d_scalar):
        """Test data generation for 1 location, 1 covariate using scalar FeatureProperties fixture"""

        # Generate data for a single covariate (e.g., age)
        n_samples = 100
        covars = np.random.uniform(0, 100, n_samples).reshape(
            1, -1
        )  # 1D array for single covariate
        baseline_noise = np.random.normal(0, 1, n_samples).reshape(
            1, -1
        )  # 1D baseline noise

        data = site_fixture_1d_scalar.generate_site_data(baseline_noise, covars)

        # Check that data shape is (1, n_samples) for single region
        assert data.shape == (1, n_samples)
        assert isinstance(data, np.ndarray)
        ## test that the output data can be transformed back to the baseline noise by doing the inverse operations
        back_transformed_data = (
            data
            - site_fixture_1d_scalar.feature_properties.alphas
            - np.dot(site_fixture_1d_scalar.feature_properties.betas, covars)
        ) / site_fixture_1d_scalar.feature_properties.sigmas
        # remove the site effects
        back_transformed_data = (
            back_transformed_data - site_fixture_1d_scalar.gamma
        ) / site_fixture_1d_scalar.delta

        np.testing.assert_allclose(
            back_transformed_data,
            baseline_noise,
            rtol=1e-5,
            err_msg="Back transformed data does not match baseline noise",
        )

    def test_site_data_generation_2D(self, site_fixture_2d):
        """Test data generation for 2 locations, 2 covariates using 2D FeatureProperties fixture"""

        # Generate data for two covariates (e.g., age and IQ)
        n_samples = 100
        covars = np.random.uniform(0, 100, n_samples * 2).reshape(
            2, -1
        )  # 2D array for two covariates
        baseline_noise = np.random.normal(0, 1, n_samples * 2).reshape(
            2, -1
        )  # 2D baseline noise
        data = site_fixture_2d.generate_site_data(baseline_noise, covars)
        # Check that data shape is (2, n_samples) for two regions
        assert data.shape == (2, n_samples)
        assert isinstance(data, np.ndarray)
        # Test that the output data can be transformed back to the baseline noise by doing the inverse operations
        back_transformed_data = (
            data
            - site_fixture_2d.feature_properties.alphas
            - np.dot(site_fixture_2d.feature_properties.betas, covars)
        ) / site_fixture_2d.feature_properties.sigmas
        # remove the site effects
        back_transformed_data = (
            back_transformed_data - site_fixture_2d.gamma
        ) / site_fixture_2d.delta
        np.testing.assert_allclose(
            back_transformed_data,
            baseline_noise,
            rtol=1e-5,
            err_msg="Back transformed data does not match baseline noise",
        )


@pytest.fixture
def ref_site_1d(feature_property_fixture_scalar):
    """Reference site fixture for harmonization tests (2D: 2 regions, 1 feature)."""
    return Site(
        gamma=0.3, delta=0.2, feature_properties=feature_property_fixture_scalar
    )


@pytest.fixture
def test_site_1d(feature_property_fixture_scalar):
    """Test site fixture for harmonization tests (2D: 2 regions, 1 feature), with unique alphas and betas."""
    # Different alphas and betas for each region, different from ref_site
    return Site(
        gamma=0.5, delta=0.5, feature_properties=feature_property_fixture_scalar
    )


@pytest.fixture
def ref_site_2d(feature_property_fixture_2D):
    """Reference site fixture for harmonization tests (2D: 2 regions, 2 features)."""
    return Site(
        gamma=[0.3, 0.4],
        delta=[0.2, 0.3],
        feature_properties=feature_property_fixture_2D,
    )


@pytest.fixture
def test_site_2d(feature_property_fixture_2D):
    """Test site fixture for harmonization tests (2D: 2 regions, 2 features), with unique alphas and betas."""
    # Different alphas and betas for each region, different from ref_site
    return Site(
        gamma=[0.5, 0.6],
        delta=[0.5, 0.6],
        feature_properties=feature_property_fixture_2D,
    )


@pytest.fixture
def baseline_and_covars_1d():
    """Fixture for baseline noise and covariates for 1D data generation."""
    n_samples = 100
    covars = np.random.uniform(0, 100, n_samples).reshape(1, -1)
    baseline_noise = np.random.normal(0, 1, n_samples).reshape(1, -1)
    return baseline_noise, covars


@pytest.fixture
def baseline_and_covars_2d():
    """Fixture for baseline noise and covariates for 2D data generation."""
    n_samples = 100
    covar1 = np.random.uniform(0, 100, n_samples).reshape(1, -1)
    covar2 = np.random.randint(0, 2, n_samples).reshape(1, -1)
    covars = np.vstack((covar1, covar2))  # Combine into 2D array
    baseline_noise = np.random.normal(0, 1, n_samples * 2).reshape(2, -1)
    return baseline_noise, covars


class TestPairwiseComBAT:
    """Test suite for PairwiseComBAT algorithm"""

    def test_harmonization_1d(self, ref_site_1d, test_site_1d, baseline_and_covars_1d):
        """Test that harmonization reduces MSE between sites using fixtures"""
        baseline_data, covars = baseline_and_covars_1d
        # Generate synthetic ages for ref and moving
        ref_data = ref_site_1d.generate_site_data(baseline_data, covars.reshape(1, -1))
        moving_data = test_site_1d.generate_site_data(
            baseline_data, covars.reshape(1, -1)
        )
        combat = PairwiseComBAT()

        # using the same covars because we're assuming they're the same samples
        # with different site effects
        combat.fit(
            covars_ref=covars.reshape(1, -1),
            Y_ref=ref_data,
            covars_moving=covars.reshape(1, -1),
            Y_moving=moving_data,
        )

        combatted_data = combat.predict(
            covars_moving=covars.reshape(1, -1), Y_moving=moving_data
        )

        mse_post_combat = mean_squared_error(
            ref_data.flatten(), combatted_data.flatten()
        )
        mse_pre_combat = mean_squared_error(ref_data.flatten(), moving_data.flatten())

        assert (
            mse_post_combat < mse_pre_combat
        ), f"MSE not reduced: before={mse_pre_combat}, after={mse_post_combat}"
        improvement = (mse_pre_combat - mse_post_combat) / mse_pre_combat * 100
        assert improvement > 10, f"Improvement {improvement}% is too small"

    def test_harmonization_2d(self, ref_site_2d, test_site_2d, baseline_and_covars_2d):
        """Test that harmonization reduces MSE between sites using fixtures"""
        baseline_data, covars = baseline_and_covars_2d
        # Generate synthetic ages for ref and moving
        ref_data = ref_site_2d.generate_site_data(baseline_data, covars)
        moving_data = test_site_2d.generate_site_data(baseline_data, covars)
        combat = PairwiseComBAT()

        # using the same covars because we're assuming they're the same samples
        # with different site effects

        combat.fit(
            covars_ref=covars,
            Y_ref=ref_data,
            covars_moving=covars,
            Y_moving=moving_data,
        )

        combatted_data = combat.predict(covars_moving=covars, Y_moving=moving_data)

        mse_post_combat = mean_squared_error(
            ref_data.flatten(), combatted_data.flatten()
        )
        mse_pre_combat = mean_squared_error(ref_data.flatten(), moving_data.flatten())

        assert (
            mse_post_combat < mse_pre_combat
        ), f"MSE not reduced: before={mse_pre_combat}, after={mse_post_combat}"
        improvement = (mse_pre_combat - mse_post_combat) / mse_pre_combat * 100
        assert improvement > 10, f"Improvement {improvement}% is too small"


class TestSaveLoad:
    """Test suite for model save/load functionality"""

    def test_save_load_1d(
        self, tmp_path, ref_site_1d, test_site_1d, baseline_and_covars_1d
    ):
        """Test that harmonization reduces MSE between sites using fixtures"""
        baseline_data, covars = baseline_and_covars_1d
        # Generate synthetic ages for ref and moving
        ref_data = ref_site_1d.generate_site_data(baseline_data, covars.reshape(1, -1))
        moving_data = test_site_1d.generate_site_data(
            baseline_data, covars.reshape(1, -1)
        )
        combat = PairwiseComBAT()

        # using the same covars because we're assuming they're the same samples
        # with different site effects
        combat.fit(
            covars_ref=covars.reshape(1, -1),
            Y_ref=ref_data,
            covars_moving=covars.reshape(1, -1),
            Y_moving=moving_data,
        )

        combatted_data = combat.predict(
            covars_moving=covars.reshape(1, -1), Y_moving=moving_data
        )

        # save the model
        model_path = tmp_path / "harmonization_model_1d"
        combat.save_model(str(model_path))
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))

        # predict using the loaded model
        combatted_data_loaded = combat_loaded.predict(
            covars_moving=covars.reshape(1, -1), Y_moving=moving_data
        )

        np.testing.assert_allclose(
            combatted_data,
            combatted_data_loaded,
            rtol=1e-5,
            err_msg="Loaded model predictions do not match original predictions",
        )

    def test_save_load_2d(
        self, tmp_path, ref_site_2d, test_site_2d, baseline_and_covars_2d
    ):
        """Test that harmonization reduces MSE between sites using fixtures"""
        baseline_data, covars = baseline_and_covars_2d
        # Generate synthetic ages for ref and moving
        ref_data = ref_site_2d.generate_site_data(baseline_data, covars)
        moving_data = test_site_2d.generate_site_data(baseline_data, covars)
        combat = PairwiseComBAT()

        # using the same covars because we're assuming they're the same samples
        # with different site effects
        combat.fit(
            covars_ref=covars,
            Y_ref=ref_data,
            covars_moving=covars,
            Y_moving=moving_data,
        )

        combatted_data = combat.predict(covars_moving=covars, Y_moving=moving_data)

        # save the model
        model_path = tmp_path / "harmonization_model_2d"
        combat.save_model(str(model_path))
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))

        # predict using the loaded model
        combatted_data_loaded = combat_loaded.predict(
            covars_moving=covars, Y_moving=moving_data
        )

        np.testing.assert_allclose(
            combatted_data,
            combatted_data_loaded,
            rtol=1e-5,
            err_msg="Loaded model predictions do not match original predictions",
        )

    def test_loaded_attributes(
        self, tmp_path, ref_site_1d, test_site_1d, baseline_and_covars_1d
    ):
        """Test that loaded model retains attributes"""
        baseline_data, covars = baseline_and_covars_1d
        ref_data = ref_site_1d.generate_site_data(baseline_data, covars.reshape(1, -1))
        moving_data = test_site_1d.generate_site_data(
            baseline_data, covars.reshape(1, -1)
        )
        combat = PairwiseComBAT()
        combat.fit(
            covars_ref=covars.reshape(1, -1),
            Y_ref=ref_data,
            covars_moving=covars.reshape(1, -1),
            Y_moving=moving_data,
        )

        # Save and reload the model
        model_path = tmp_path / "harmonization_model_attributes"
        combat.save_model(str(model_path))
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))

        # Loop through all non-private attributes of the original model
        for attr in dir(combat):
            # skip methods
            val = getattr(combat, attr)
            if callable(val):
                continue
            loaded_val = getattr(combat_loaded, attr, None)
            if isinstance(val, np.ndarray):
                np.testing.assert_allclose(
                    val,
                    loaded_val,
                    rtol=1e-5,
                    err_msg=f"Attribute {attr} does not match after loading",
                )
            else:
                assert (
                    val == loaded_val
                ), f"Attribute {attr} does not match after loading"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
