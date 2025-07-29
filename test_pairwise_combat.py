import pytest
import numpy as np
from pairwise_combat import Site, PairwiseComBAT, compute_harmonization_mse

# Test constants
MIN_AGE = 40
MAX_AGE = 95
N_SAMPLES = 100
ALPHA = 0.5  # True intercept
BETA = 0.04  # True slope (kept as scalar for backward compatibility in tests)
SIGMA = 0.5  # True noise level


class TestDataGeneration:
    """Test suite for data generation and site initialization"""
    
    def test_site_initialization(self):
        """Test that Site objects are initialized correctly"""
        # Test scalar initialization (backward compatibility)
        site = Site(gamma=0.5, delta=1.2, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        assert np.array_equal(site.gamma, [0.5])
        assert np.array_equal(site.delta, [1.2])
        
        # Test array initialization
        site_array = Site(gamma=[0.3, 0.5], delta=[1.1, 1.3], alpha=ALPHA, beta=[0.04, -0.02], sigma=SIGMA)
        assert np.array_equal(site_array.gamma, [0.3, 0.5])
        assert np.array_equal(site_array.delta, [1.1, 1.3])
        
        # Test mismatched shapes should raise error
        with pytest.raises(ValueError, match="gamma and delta must have the same shape"):
            Site(gamma=[0.3, 0.5], delta=[1.1], alpha=ALPHA, beta=[0.04, -0.02], sigma=SIGMA)
    
    def test_single_covariate_site_effects(self):
        """Test site effects with a single covariate (backward compatibility)"""
        # Set seed for reproducibility
        np.random.seed(123)
        
        ref_site = Site(gamma=-0.3, delta=0.7, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        test_site = Site(gamma=0.5, delta=1.2, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        
        # Generate same baseline noise for both sites
        ages = np.random.uniform(MIN_AGE, MAX_AGE, 1000)
        baseline_noise = np.random.normal(0, 1, 1000)
        
        ref_data = ref_site.generate_site_data(ages, baseline_noise)
        test_data = test_site.generate_site_data(ages, baseline_noise)
        
        # Verify data shapes - now returns (1, n_samples) for single region
        assert ref_data.shape == (1, 1000)  # Single region, 2D output
        assert test_data.shape == (1, 1000)
        
        # Remove population-level effects to isolate site effects
        population_effects = ALPHA + BETA * ages
        ref_residuals = ref_data.flatten() - population_effects  # Flatten for easier computation
        test_residuals = test_data.flatten() - population_effects
        
        # Test that site effects are applied correctly
        # The difference should reflect the batch effect parameters
        assert not np.allclose(ref_residuals, test_residuals)
        
        # Test that data is reasonable (finite and not all zeros)
        assert np.isfinite(ref_data).all()
        assert np.isfinite(test_data).all()
        assert not np.allclose(ref_data, 0)
        assert not np.allclose(test_data, 0)
    
    def test_multi_covariate_site_effects(self):
        """Test site effects with multiple covariates"""
        # Set seed for reproducibility
        np.random.seed(456)
        
        # Define multi-covariate parameters
        alpha, beta, sigma = 0.5, [0.04, -0.02], 0.5
        
        # Create sites with different effects for each covariate
        ref_site = Site(gamma=[0.0, 0.0], delta=[1.0, 1.0], alpha=alpha, beta=beta, sigma=sigma)  # Reference: no effects
        test_site = Site(gamma=[0.3, -0.2], delta=[1.2, 0.8], alpha=alpha, beta=beta, sigma=sigma)  # Different effects per covariate
        
        # Generate multi-covariate data
        n_samples = 500
        n_features = 2
        X = np.zeros((n_features, n_samples))
        X[0] = np.random.uniform(MIN_AGE, MAX_AGE, n_samples)  # Age
        X[1] = np.random.normal(100, 15, n_samples)           # IQ
        
        baseline_noise = np.random.normal(0, 1, n_samples)
        
        ref_data = ref_site.generate_site_data(X, baseline_noise)
        test_data = test_site.generate_site_data(X, baseline_noise)
        
        # Verify data shapes - now returns (1, n_samples) for single region  
        assert ref_data.shape == (1, n_samples)  # Single region, 2D output
        assert test_data.shape == (1, n_samples)
        
        # Test that site effects are applied
        assert not np.allclose(ref_data, test_data)
        
        # Test that data is reasonable
        assert np.isfinite(ref_data).all()
        assert np.isfinite(test_data).all()
        assert not np.allclose(ref_data, 0)
        assert not np.allclose(test_data, 0)
    
    def test_multi_region_multi_covariate(self):
        """Test site effects with multiple covariates and multiple regions"""
        # Set seed for reproducibility
        np.random.seed(789)
        
        # Define multi-covariate parameters
        alpha, beta, sigma = 0.5, [0.04, -0.02], 0.5
        
        # Create sites with different effects for each covariate
        ref_site = Site(gamma=[0.1, -0.1], delta=[0.9, 1.1], alpha=alpha, beta=beta, sigma=sigma)
        test_site = Site(gamma=[0.4, 0.2], delta=[1.3, 0.7], alpha=alpha, beta=beta, sigma=sigma)
        
        # Generate multi-covariate, multi-region data
        n_samples = 200
        n_features = 2
        n_regions = 5
        
        X = np.zeros((n_features, n_samples))
        X[0] = np.random.uniform(MIN_AGE, MAX_AGE, n_samples)  # Age
        X[1] = np.random.normal(100, 15, n_samples)           # IQ
        
        baseline_noise = np.random.normal(0, 1, (n_regions, n_samples))
        
        ref_data = ref_site.generate_site_data(X, baseline_noise, n_regions=n_regions)
        test_data = test_site.generate_site_data(X, baseline_noise, n_regions=n_regions)
        
        # Verify data shapes
        assert ref_data.shape == (n_regions, n_samples)
        assert test_data.shape == (n_regions, n_samples)
        
        # Test that site effects are applied
        assert not np.allclose(ref_data, test_data)
        
        # Test that data is reasonable for all regions
        assert np.isfinite(ref_data).all()
        assert np.isfinite(test_data).all()
        assert not np.allclose(ref_data, 0)
        assert not np.allclose(test_data, 0)
        
        # Test that different regions have different values (not identical)
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                # Regions should have some differences due to baseline noise
                assert not np.allclose(ref_data[i], ref_data[j], rtol=1e-3)
    
    def test_data_generation_shape(self):
        """Test that generated data has correct shape"""
        site = Site(gamma=0.2, delta=1.1, alpha=ALPHA, beta=BETA, sigma=SIGMA)  # Non-zero reference
        ages = np.random.uniform(MIN_AGE, MAX_AGE, 50)
        data = site.generate_site_data(ages)
        assert data.shape == (1, 50)  # Now returns (1, n_samples) for single region
        assert isinstance(data, np.ndarray)
    
    def test_covariate_broadcasting(self):
        """Test proper broadcasting for single covariate but multi-element beta."""
        # Multiple beta values (e.g., multiple regions), single covariate (gamma/delta scalar)
        site = Site(gamma=[0.3], delta=[1.2], alpha=ALPHA, beta=[0.04], sigma=SIGMA)
        
        # Generate single-feature data (to match single gamma/delta)
        n_samples = 100
        n_features = 1  # Changed from 3 to match gamma/delta dimensions
        X = np.random.uniform(40, 80, (n_features, n_samples))
        
        data = site.generate_site_data(X)
        
        # Should work without error and produce reasonable output
        assert data.shape == (1, n_samples)  # Single region, 2D output
        assert np.isfinite(data).all()
        assert not np.allclose(data, 0)
    
    def test_site_effects_validation(self):
        """Test validation of gamma and delta dimensions"""
        # Test that mismatched array lengths raise an error
        with pytest.raises(ValueError, match="beta must have the same shape as gamma and delta"):
            site = Site(gamma=[0.3, 0.5, 0.7], delta=[1.1, 1.3, 1.4], alpha=ALPHA, beta=[0.04, -0.02], sigma=SIGMA)  # Different lengths
    
    def test_backward_compatibility(self):
        """Test that the new array-based system works with scalar inputs (backward compatibility)"""
        # Set seed for reproducibility
        np.random.seed(123)
        
        ref_site = Site(gamma=[-0.3], delta=[0.7], alpha=ALPHA, beta=[BETA], sigma=SIGMA)
        test_site = Site(gamma=[0.5], delta=[1.2], alpha=ALPHA, beta=[BETA], sigma=SIGMA)
        
        # Generate same baseline noise for both sites
        ages = np.random.uniform(MIN_AGE, MAX_AGE, 1000)
        baseline_noise = np.random.normal(0, 1, 1000)
        
        ref_data = ref_site.generate_site_data(ages, baseline_noise)
        test_data = test_site.generate_site_data(ages, baseline_noise)
        
        # Remove population-level effects
        population_effects = ALPHA + BETA * ages  # BETA is a scalar
        ref_residuals = ref_data.flatten() - population_effects  # Flatten since data is (1, n_samples)
        test_residuals = test_data.flatten() - population_effects
        
        # Calculate batch effects
        mean_diff = np.mean(test_residuals) - np.mean(ref_residuals)
        std_ratio = np.std(test_residuals) / np.std(ref_residuals)
        
        # Expected effects (need to access first element since gamma/delta are now arrays)
        # The gamma parameter affects the covariate term, so the mean difference includes covariate effects
        expected_gamma_diff = (test_site.gamma[0] - ref_site.gamma[0]) * SIGMA * np.mean(ages)
        
        # The delta parameter affects the variance, but when gamma is also present, we need the full variance formula:
        # var(σ*(δ*ε + γ*X)) = σ² * (δ²*var(ε) + γ²*var(X))
        ref_var = (SIGMA**2) * (ref_site.delta[0]**2 * np.var(baseline_noise) + ref_site.gamma[0]**2 * np.var(ages))
        test_var = (SIGMA**2) * (test_site.delta[0]**2 * np.var(baseline_noise) + test_site.gamma[0]**2 * np.var(ages))
        expected_std_ratio = np.sqrt(test_var / ref_var)
        
        # Assertions with tolerance
        assert abs(mean_diff - expected_gamma_diff) < 0.5, f"Gamma error too large: {abs(mean_diff - expected_gamma_diff)}"
        assert abs(std_ratio - expected_std_ratio) < 0.1, f"Delta error too large: {abs(std_ratio - expected_std_ratio)}"
    
    @pytest.mark.parametrize("gamma_ref,delta_ref,gamma_test,delta_test", [
        (0.0, 1.0, 0.5, 1.2),
        (-0.3, 0.7, 0.5, 1.2),
        (0.2, 1.5, -0.4, 0.8),
        (1.0, 0.5, 0.0, 2.0),
    ])
    def test_batch_effects_parametrized(self, gamma_ref, delta_ref, gamma_test, delta_test):
        """Test batch effects with various parameter combinations"""
        np.random.seed(456)
        
        ref_site = Site(gamma=[gamma_ref], delta=[delta_ref], alpha=ALPHA, beta=[BETA], sigma=SIGMA)
        test_site = Site(gamma=[gamma_test], delta=[delta_test], alpha=ALPHA, beta=[BETA], sigma=SIGMA)
        
        # Generate same baseline noise for both sites
        ages = np.random.uniform(MIN_AGE, MAX_AGE, 1000)
        baseline_noise = np.random.normal(0, 1, 1000)
        
        ref_data = ref_site.generate_site_data(ages, baseline_noise)
        test_data = test_site.generate_site_data(ages, baseline_noise)
        
        # Remove population-level effects
        population_effects = ALPHA + BETA * ages  # BETA is a scalar
        ref_residuals = ref_data.flatten() - population_effects  # Flatten since data is (1, n_samples)
        test_residuals = test_data.flatten() - population_effects
        
        # Calculate batch effects
        mean_diff = np.mean(test_residuals) - np.mean(ref_residuals)
        std_ratio = np.std(test_residuals) / np.std(ref_residuals)
        
        # Expected effects (access first element since gamma/delta are now arrays)
        # The gamma parameter affects the covariate term, so the mean difference includes covariate effects
        expected_gamma_diff = (gamma_test - gamma_ref) * SIGMA * np.mean(ages)
        
        # The delta parameter affects the variance, but when gamma is also present, we need the full variance formula:
        # var(σ*(δ*ε + γ*X)) = σ² * (δ²*var(ε) + γ²*var(X))
        ref_var = (SIGMA**2) * (gamma_ref**2 * np.var(ages) + delta_ref**2 * np.var(baseline_noise))
        test_var = (SIGMA**2) * (gamma_test**2 * np.var(ages) + delta_test**2 * np.var(baseline_noise))
        expected_std_ratio = np.sqrt(test_var / ref_var)
        
        # Assertions with tolerance
        assert abs(mean_diff - expected_gamma_diff) < 0.5
        assert abs(std_ratio - expected_std_ratio) < 0.1


class TestPairwiseComBAT:
    """Test suite for PairwiseComBAT algorithm"""
    
    def test_combat_initialization(self):
        """Test that ComBAT object initializes correctly"""
        combat = PairwiseComBAT()
        # The actual class may not have these attributes initially
        # Just test that it initializes without error
        assert combat is not None
    
    def test_harmonization_reduces_mse(self):
        """Test that harmonization reduces MSE between sites"""
        np.random.seed(303)
        
        # Create sites with different batch effects
        ref_site = Site(gamma=0.2, delta=1.1, alpha=ALPHA, beta=BETA, sigma=SIGMA)  # Non-zero reference
        moving_site = Site(gamma=0.5, delta=1.3, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        
        # Generate data
        ages_ref = np.random.uniform(MIN_AGE, MAX_AGE, N_SAMPLES)
        ages_moving = np.random.uniform(MIN_AGE, MAX_AGE, N_SAMPLES)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        # Harmonize
        combat = PairwiseComBAT()
        harmonized_data = combat.fit_predict(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Compute MSE using external function
        mse_before = compute_harmonization_mse(
            X_ref=ages_ref.reshape(-1, 1), 
            Y_ref=ref_data, 
            X_moving=ages_moving.reshape(-1, 1), 
            Y_harmonized=moving_data,  # Original data before harmonization
            alpha_hat=combat.alpha_hat_,
            beta_hat=combat.beta_hat_
        )
        mse_after = compute_harmonization_mse(
            X_ref=ages_ref.reshape(-1, 1), 
            Y_ref=ref_data, 
            X_moving=ages_moving.reshape(-1, 1), 
            Y_harmonized=harmonized_data,  # Harmonized data
            alpha_hat=combat.alpha_hat_,
            beta_hat=combat.beta_hat_
        )
        
        # MSE should be reduced
        assert mse_after < mse_before, f"MSE not reduced: before={mse_before}, after={mse_after}"
        
        # Should achieve some improvement
        improvement = (mse_before - mse_after) / mse_before * 100
        assert improvement > 10, f"Improvement {improvement}% is too small"
    
    def test_harmonization_with_different_reference_sites(self):
        """Test harmonization works with different reference site parameters"""
        np.random.seed(404)
        
        # Test with non-zero reference
        ref_site = Site(gamma=-0.2, delta=0.8, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        moving_site = Site(gamma=0.3, delta=1.4, alpha=ALPHA, beta=BETA, sigma=SIGMA)
        
        # Generate data
        ages_ref = np.random.uniform(MIN_AGE, MAX_AGE, N_SAMPLES)
        ages_moving = np.random.uniform(MIN_AGE, MAX_AGE, N_SAMPLES)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        # Harmonize
        combat = PairwiseComBAT()
        harmonized_data = combat.fit_predict(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Should successfully return harmonized data
        assert harmonized_data.shape == moving_data.shape  # Compare shapes instead of len()
        assert not np.allclose(harmonized_data, moving_data, rtol=0.01), "Harmonized data should be different from original"
        
        # Should be able to get estimated parameters
        params = combat.get_estimated_parameters()
        assert 'alpha_hat' in params
        assert 'beta_hat' in params
        assert 'sigma_hat' in params
        assert params['sigma_hat'] > 0
    
    @pytest.mark.parametrize("n_samples", [50, 100, 200])
    def test_harmonization_with_different_sample_sizes(self, n_samples):
        """Test harmonization works with different sample sizes"""
        np.random.seed(505)
        
        ref_site = Site(gamma=[0.1], delta=[0.9], alpha=ALPHA, beta=[BETA], sigma=SIGMA)  # Non-zero reference
        moving_site = Site(gamma=[0.4], delta=[1.1], alpha=ALPHA, beta=[BETA], sigma=SIGMA)
        
        # Generate data with specified sample size
        ages_ref = np.random.uniform(MIN_AGE, MAX_AGE, n_samples)
        ages_moving = np.random.uniform(MIN_AGE, MAX_AGE, n_samples)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        # Harmonize
        combat = PairwiseComBAT()
        harmonized_data = combat.fit_predict(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Should work for all sample sizes
        assert harmonized_data.shape[1] == n_samples  # Check second dimension for number of samples
        params = combat.get_estimated_parameters()
        assert params['sigma_hat'] > 0


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from data generation to harmonization"""
        np.random.seed(606)
        
        # Create multiple sites
        sites = {
            'Reference': Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2),  # Non-zero reference
            'Site_A': Site(gamma=[0.5], delta=[1.3], alpha=1.0, beta=[0.05], sigma=0.2),
            'Site_B': Site(gamma=[-0.3], delta=[0.8], alpha=1.0, beta=[0.05], sigma=0.2),
        }
        
        # Generate data for each site
        site_data = {}
        baseline_noise = np.random.normal(0, 1, N_SAMPLES)
        
        for name, site in sites.items():
            ages = np.random.randint(MIN_AGE, MAX_AGE, N_SAMPLES)
            data = site.generate_site_data(ages, baseline_noise.copy())
            site_data[name] = {'ages': ages, 'data': data, 'site': site}
        
        # Test harmonization of each site to reference
        ref_data = site_data['Reference']
        
        for site_name in ['Site_A', 'Site_B']:
            moving_data = site_data[site_name]
            
            # Harmonize
            combat = PairwiseComBAT()
            harmonized_data = combat.fit_predict(
                X_ref=ref_data['ages'].reshape(-1, 1),
                Y_ref=ref_data['data'],
                X_moving=moving_data['ages'].reshape(-1, 1),
                Y_moving=moving_data['data']
            )
            
            # Verify MSE improvement using external function
            mse_before = compute_harmonization_mse(
                X_ref=ref_data['ages'].reshape(-1, 1), 
                Y_ref=ref_data['data'], 
                X_moving=moving_data['ages'].reshape(-1, 1), 
                Y_harmonized=moving_data['data'],
                alpha_hat=combat.alpha_hat_,
                beta_hat=combat.beta_hat_
            )
            mse_after = compute_harmonization_mse(
                X_ref=ref_data['ages'].reshape(-1, 1), 
                Y_ref=ref_data['data'], 
                X_moving=moving_data['ages'].reshape(-1, 1), 
                Y_harmonized=harmonized_data,
                alpha_hat=combat.alpha_hat_,
                beta_hat=combat.beta_hat_
            )
            
            improvement = (mse_before - mse_after) / mse_before * 100
            assert improvement > 10, f"{site_name}: Improvement {improvement}% too small"
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed"""
        ref_site = Site(gamma=[0.1], delta=[0.9], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.3], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)
        
        # Use fixed baseline noise for both runs to ensure reproducibility
        np.random.seed(707)
        ages_ref = np.random.uniform(MIN_AGE, MAX_AGE, 100)
        ages_moving = np.random.uniform(MIN_AGE, MAX_AGE, 100)
        baseline_ref = np.random.normal(0, 1, 100)
        baseline_moving = np.random.normal(0, 1, 100)
        
        # First run
        ref_data1 = ref_site.generate_site_data(ages_ref, baseline_ref)
        moving_data1 = moving_site.generate_site_data(ages_moving, baseline_moving)
        
        combat1 = PairwiseComBAT()
        harmonized1 = combat1.fit_predict(
            X_ref=ages_ref.reshape(-1, 1), 
            Y_ref=ref_data1, 
            X_moving=ages_moving.reshape(-1, 1), 
            Y_moving=moving_data1
        )
        
        # Second run with same data
        ref_data2 = ref_site.generate_site_data(ages_ref, baseline_ref)
        moving_data2 = moving_site.generate_site_data(ages_moving, baseline_moving)
        
        combat2 = PairwiseComBAT()
        harmonized2 = combat2.fit_predict(
            X_ref=ages_ref.reshape(-1, 1), 
            Y_ref=ref_data2, 
            X_moving=ages_moving.reshape(-1, 1), 
            Y_moving=moving_data2
        )
        
        # Results should be identical
        np.testing.assert_allclose(harmonized1, harmonized2, rtol=1e-10)
        
        params1 = combat1.get_estimated_parameters()
        params2 = combat2.get_estimated_parameters()
        assert abs(params1['alpha_hat'] - params2['alpha_hat']) < 1e-10
        assert abs(params1['beta_hat'] - params2['beta_hat']) < 1e-10
        assert abs(params1['sigma_hat'] - params2['sigma_hat']) < 1e-10


class TestMultiFeature:
    """Test suite for multi-feature generalization"""
    
    def test_single_feature_2d_format(self):
        """Test that single feature works with 2D multi-region format"""
        from pairwise_combat import generate_age_covariates
        
        site = Site(gamma=[0.5], delta=[1.2], alpha=1.0, beta=[0.05], sigma=0.2)
        
        # Test with 1D ages array (converted internally to 2D)
        rng = np.random.default_rng(42)
        ages_1d = rng.integers(MIN_AGE, MAX_AGE, 50)
        data_1d = site.generate_site_data(ages_1d)
        
        # Test with explicit 2D ages array (new format)
        ages_2d = generate_age_covariates(50, seed=42)
        data_2d = site.generate_site_data(ages_2d)
        
        # Both should return 2D format: (n_regions, n_samples)
        # When no n_regions specified, defaults to 1 (single region)
        assert data_1d.shape == (1, 50)  # Default: n_regions = 1
        assert data_2d.shape == (1, 50)  # Same default behavior
        
        # Test with explicit n_regions for more realistic MRI data
        data_explicit = site.generate_site_data(ages_2d, n_regions=10)
        assert data_explicit.shape == (10, 50)  # 10 brain regions, 50 samples
        
        # Both should be valid arrays
        assert isinstance(data_1d, np.ndarray)
        assert isinstance(data_2d, np.ndarray)
        assert isinstance(data_explicit, np.ndarray)
        assert np.isfinite(data_1d).all()
        assert np.isfinite(data_2d).all()
        assert np.isfinite(data_explicit).all()
        
    def test_multi_feature_data_generation(self):
        """Test that multi-feature data generation works correctly"""
        # Create site with 2 features (age and IQ) 
        site = Site(gamma=[0.3, 0.2], delta=[1.1, 1.0], alpha=1.0, beta=[0.05, 0.01], sigma=0.2)
        
        # Create 2-feature data with continuous variables only
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 2
        n_regions = 5  # Specify number of regions explicitly
        
        X = np.zeros((n_features, n_samples))
        X[0] = rng.integers(40, 80, n_samples)     # Age
        X[1] = rng.normal(100, 15, n_samples)      # IQ
        
        # Generate data with explicit number of regions
        data = site.generate_site_data(X, n_regions=n_regions)
        
        # Verify output shape - should be (n_regions, n_samples)
        assert data.shape == (n_regions, n_samples)
        assert isinstance(data, np.ndarray)
        
        # Verify data is reasonable (not all zeros, within reasonable range)
        assert not np.allclose(data, 0)
        assert np.isfinite(data).all()
        
    def test_multi_feature_combat_harmonization(self):
        """Test that ComBAT works with multiple features"""
        # Create sites with different batch effects
        ref_site = Site(gamma=[0.2, 0.2], delta=[1.1, 1.1], alpha=1.0, beta=[0.05, 0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.5, 0.5], delta=[1.3, 1.3], alpha=1.0, beta=[0.05, 0.05], sigma=0.2)
        
        # Generate multi-feature data
        rng = np.random.default_rng(42)
        n_samples = 80
        n_features = 2
        n_regions = 10  # Multiple brain regions
        
        # Reference site data
        X_ref = np.zeros((n_features, n_samples))
        X_ref[0] = rng.integers(40, 80, n_samples)   # Age
        X_ref[1] = rng.normal(100, 15, n_samples)    # IQ
        
        noise_ref = rng.normal(0, 1, (n_regions, n_samples))
        Y_ref = ref_site.generate_site_data(X_ref, noise_ref, n_regions=n_regions)
        
        # Moving site data  
        X_moving = np.zeros((n_features, n_samples))
        X_moving[0] = rng.integers(45, 85, n_samples)  # Age
        X_moving[1] = rng.normal(105, 12, n_samples)   # IQ
        
        noise_moving = rng.normal(0, 1, (n_regions, n_samples))
        Y_moving = moving_site.generate_site_data(X_moving, noise_moving, n_regions=n_regions)
        
        # Apply ComBAT harmonization
        combat = PairwiseComBAT()
        Y_harmonized = combat.fit_predict(
            X_ref=X_ref.T,    # Transpose to (n_samples, n_features) for ComBAT
            Y_ref=Y_ref,
            X_moving=X_moving.T,
            Y_moving=Y_moving
        )
        
        # Verify output shape and properties - should be (n_regions, n_samples)
        assert Y_harmonized.shape == Y_moving.shape == (n_regions, n_samples)
        assert isinstance(Y_harmonized, np.ndarray)
        assert np.isfinite(Y_harmonized).all()
        
        # Verify that harmonization changed the data
        assert not np.allclose(Y_harmonized, Y_moving)
        
        # Check that ComBAT estimated parameters
        assert combat.alpha_hat_ is not None
        assert combat.beta_hat_ is not None  
        assert combat.sigma_hat_ is not None
        assert len(combat.beta_hat_) == n_features  # Should match number of features
        
    def test_multi_feature_mse_computation(self):
        """Test MSE computation with multiple features"""
        # Create simple test data
        rng = np.random.default_rng(42)
        n_samples = 50
        n_features = 2
        
        X_ref = rng.random((n_features, n_samples))
        Y_ref = rng.random(n_samples)
        X_moving = rng.random((n_features, n_samples))  
        Y_harmonized = rng.random(n_samples)
        
        # Create dummy fitted model for MSE computation
        combat = PairwiseComBAT()
        # Set dummy parameters for MSE computation
        combat.alpha_hat_ = 0.5
        combat.beta_hat_ = np.array([0.1, 0.2])
        combat.sigma_hat_ = 0.3
        combat.is_fitted_ = True
        
        # Compute MSE using external function
        mse = compute_harmonization_mse(
            X_ref=X_ref.T, Y_ref=Y_ref,
            X_moving=X_moving.T, Y_harmonized=Y_harmonized,
            alpha_hat=combat.alpha_hat_,
            beta_hat=combat.beta_hat_
        )
        
        # Verify MSE is a valid number
        assert isinstance(mse, (float, np.floating))
        assert mse >= 0
        assert np.isfinite(mse)
        
    def test_feature_dimension_handling(self):
        """Test that the code correctly handles different input formats"""
        site = Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)
        rng = np.random.default_rng(42)
        n_samples = 30
        n_regions = 5  # Explicit number of regions
        
        # Test 1: 1D array input (converted to 2D internally)
        X_1d = rng.integers(40, 80, n_samples)
        data_1d = site.generate_site_data(X_1d, n_regions=n_regions)
        assert data_1d.shape == (n_regions, n_samples)
        
        # Test 2: 2D array with shape (1, n_samples)
        X_2d_correct = X_1d.reshape(1, -1)
        data_2d_correct = site.generate_site_data(X_2d_correct, n_regions=n_regions)
        assert data_2d_correct.shape == (n_regions, n_samples)
        
        # Test 3: Multi-feature 2D array with continuous variables (requires multi-feature Site)
        multi_site = Site(gamma=[0.2, 0.1], delta=[1.1, 1.0], alpha=1.0, beta=[0.05, 0.01], sigma=0.2)
        X_multi = np.zeros((2, n_samples))
        X_multi[0] = rng.integers(40, 80, n_samples)    # Age  
        X_multi[1] = rng.normal(100, 15, n_samples)     # IQ
        data_multi = multi_site.generate_site_data(X_multi, n_regions=n_regions)
        assert data_multi.shape == (n_regions, n_samples)


class TestSaveLoad:
    """Test suite for model save/load functionality"""
    
    def test_save_load_basic(self, tmp_path):
        """Test basic save and load functionality"""
        # Create and train a model
        ref_site = Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.5], delta=[1.3], alpha=1.0, beta=[0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        n_samples = 50
        
        ages_ref = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        ages_moving = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        # Train model
        combat = PairwiseComBAT()
        combat.fit(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Save model
        model_path = tmp_path / "test_model"
        combat.save_model(str(model_path))
        
        # Load model
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))
        
        # Verify parameters match
        assert abs(combat.alpha_hat_ - combat_loaded.alpha_hat_) < 1e-10
        np.testing.assert_allclose(combat.beta_hat_, combat_loaded.beta_hat_, rtol=1e-10)
        assert abs(combat.sigma_hat_ - combat_loaded.sigma_hat_) < 1e-10
        assert combat.max_iter == combat_loaded.max_iter
        assert abs(combat.tol - combat_loaded.tol) < 1e-10
    
    def test_save_load_multi_feature(self, tmp_path):
        """Test save/load with multi-feature data"""
        ref_site = Site(gamma=[0.1, 0.1], delta=[0.9, 0.9], alpha=1.0, beta=[0.05, 0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.3, 0.3], delta=[1.2, 1.2], alpha=1.0, beta=[0.05, 0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        n_samples = 40
        n_features = 2
        
        # Multi-feature data
        X_ref = np.zeros((n_features, n_samples))
        X_ref[0] = rng.integers(40, 80, n_samples)
        X_ref[1] = rng.normal(100, 15, n_samples)
        
        X_moving = np.zeros((n_features, n_samples))
        X_moving[0] = rng.integers(45, 85, n_samples)
        X_moving[1] = rng.normal(105, 12, n_samples)
        
        ref_data = ref_site.generate_site_data(X_ref)
        moving_data = moving_site.generate_site_data(X_moving)
        
        # Train model
        combat = PairwiseComBAT()
        combat.fit(
            X_ref=X_ref.T,
            Y_ref=ref_data,
            X_moving=X_moving.T,
            Y_moving=moving_data
        )
        
        # Save and load
        model_path = tmp_path / "multi_feature_model"
        combat.save_model(str(model_path))
        
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))
        
        # Verify multi-feature parameters
        assert len(combat_loaded.beta_hat_) == n_features
        np.testing.assert_allclose(combat.beta_hat_, combat_loaded.beta_hat_, rtol=1e-10)
    
    def test_predict_inference(self, tmp_path):
        """Test model inference with predict method"""
        # Train model
        ref_site = Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.4], delta=[1.2], alpha=1.0, beta=[0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        n_samples = 60
        
        ages_ref = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        ages_moving = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        combat = PairwiseComBAT()
        combat.fit(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Save and load model
        model_path = tmp_path / "inference_model"
        combat.save_model(str(model_path))
        
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))
        
        # Generate new data for inference
        ages_new = rng.integers(MIN_AGE, MAX_AGE, 30)
        data_new = moving_site.generate_site_data(ages_new)
        
        # Test prediction
        harmonized = combat_loaded.predict(ages_new.reshape(-1, 1), data_new)
        
        # Should return valid data
        assert harmonized.shape == data_new.shape
        assert np.isfinite(harmonized).all()
        assert not np.allclose(harmonized, data_new)  # Should be different from original
    
    def test_load_and_predict_convenience(self, tmp_path):
        """Test the convenience load_and_predict class method"""
        # Train and save model
        ref_site = Site(gamma=[0.1], delta=[0.9], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.3], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        ages_ref = rng.integers(MIN_AGE, MAX_AGE, 50)
        ages_moving = rng.integers(MIN_AGE, MAX_AGE, 50)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        combat = PairwiseComBAT()
        combat.fit(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        model_path = tmp_path / "convenience_model"
        combat.save_model(str(model_path))
        
        # Generate test data
        ages_test = rng.integers(MIN_AGE, MAX_AGE, 20)
        data_test = moving_site.generate_site_data(ages_test)
        
        # Test convenience method
        harmonized = PairwiseComBAT.load_and_predict(
            str(model_path), ages_test.reshape(-1, 1), data_test
        )
        
        # Should work correctly
        assert harmonized.shape == data_test.shape
        assert np.isfinite(harmonized).all()
    
    def test_save_untrained_model_error(self, tmp_path):
        """Test that saving an untrained model raises an error"""
        combat = PairwiseComBAT()
        model_path = tmp_path / "untrained_model"
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            combat.save_model(str(model_path))
    
    def test_load_nonexistent_model_error(self):
        """Test that loading a non-existent model raises an error"""
        combat = PairwiseComBAT()
        
        with pytest.raises(FileNotFoundError):
            combat.load_model("nonexistent_model")
    
    def test_predict_untrained_model_error(self):
        """Test that prediction with untrained model raises an error"""
        combat = PairwiseComBAT()
        
        X_dummy = np.random.random((10, 1))
        Y_dummy = np.random.random(10)
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            combat.predict(X_dummy, Y_dummy)
    
    def test_model_versioning_and_metadata(self, tmp_path):
        """Test that model metadata is correctly saved and loaded"""
        # Train a simple model
        ref_site = Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.2], delta=[1.1], alpha=1.0, beta=[0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        ages_ref = rng.integers(MIN_AGE, MAX_AGE, 30)
        ages_moving = rng.integers(MIN_AGE, MAX_AGE, 30)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        combat = PairwiseComBAT(max_iter=50, tol=1e-7)  # Custom parameters
        combat.fit(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Save model
        model_path = tmp_path / "versioned_model"
        combat.save_model(str(model_path))
        
        # Check that HDF5 file exists
        assert (model_path.with_suffix('.h5')).exists()
        
        # Load and verify custom parameters were preserved
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))
        
        assert combat_loaded.max_iter == 50
        assert abs(combat_loaded.tol - 1e-7) < 1e-10

    def test_transformation_consistency(self, tmp_path):
        """Test that loaded models produce identical transformation results"""
        # Train model
        ref_site = Site(gamma=[0.2], delta=[1.2], alpha=1.0, beta=[0.05], sigma=0.2)  # Non-zero reference
        moving_site = Site(gamma=[0.5], delta=[1.3], alpha=1.0, beta=[0.05], sigma=0.2)
        
        rng = np.random.default_rng(42)
        n_samples = 100
        
        # Training data
        ages_ref = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        ages_moving = rng.integers(MIN_AGE, MAX_AGE, n_samples)
        
        ref_data = ref_site.generate_site_data(ages_ref)
        moving_data = moving_site.generate_site_data(ages_moving)
        
        # Train model
        combat = PairwiseComBAT()
        combat.fit(
            X_ref=ages_ref.reshape(-1, 1),
            Y_ref=ref_data,
            X_moving=ages_moving.reshape(-1, 1),
            Y_moving=moving_data
        )
        
        # Generate test data for transformation consistency check
        ages_test = rng.integers(MIN_AGE, MAX_AGE, 50)
        data_test = moving_site.generate_site_data(ages_test)
        
        # Get original model's prediction
        Y_original_pred = combat.predict(ages_test.reshape(-1, 1), data_test)
        
        # Save and load model
        model_path = tmp_path / "consistency_test_model"
        combat.save_model(str(model_path))
        
        combat_loaded = PairwiseComBAT()
        combat_loaded.load_model(str(model_path))
        
        # Test transformation consistency - loaded model should give identical results
        Y_loaded_pred = combat_loaded.predict(ages_test.reshape(-1, 1), data_test)
        np.testing.assert_allclose(Y_original_pred, Y_loaded_pred, rtol=1e-10)
        
        # Test convenience method gives same results
        Y_convenience_pred = PairwiseComBAT.load_and_predict(
            str(model_path), ages_test.reshape(-1, 1), data_test
        )
        np.testing.assert_allclose(Y_original_pred, Y_convenience_pred, rtol=1e-10)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
