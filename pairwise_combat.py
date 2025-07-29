import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LinearRegression
from typing import Tuple
from pathlib import Path
from datetime import datetime
import h5py

MIN_AGE = 40
MAX_AGE = 95
N_SAMPLES_PER_SITE = 100


class Site:
    """Represents a data collection site with specific batch effects"""
    
    def __init__(self, gamma: ArrayLike, delta: ArrayLike, alpha: float, beta: ArrayLike, sigma: float):
        """
        Initialize site with batch effects and population parameters
        
        Args:
            gamma: Additive site effect for each covariate feature
                   Shape: (n_features,) where n_features is the number of continuous covariates
            delta: Multiplicative site effect for each covariate feature
                   Shape: (n_features,) where n_features is the number of continuous covariates
            alpha: Population intercept parameter
            beta: Population slope parameters for each covariate feature
                  Shape: (n_features,) where n_features is the number of continuous covariates
            sigma: Population noise standard deviation
        """
        self.gamma = np.atleast_1d(gamma)
        self.delta = np.atleast_1d(delta)
        self.alpha = alpha
        self.beta = np.atleast_1d(beta)
        self.sigma = sigma
        
        # Ensure gamma and delta have the same shape
        if self.gamma.shape != self.delta.shape:
            raise ValueError(f"gamma and delta must have the same shape. Got gamma: {self.gamma.shape}, delta: {self.delta.shape}")
        
        # Ensure beta matches gamma and delta shape
        if self.beta.shape != self.gamma.shape:
            raise ValueError(f"beta must have the same shape as gamma and delta. Got beta: {self.beta.shape}, gamma: {self.gamma.shape}")

    def add_site_effects(self, baseline_data: ArrayLike, X: ArrayLike, population_noise: ArrayLike = None) -> ArrayLike:
        """
        Apply site-specific additive and multiplicative effects to baseline data
        
        Following the correct data generation process:
        1. Start with baseline normal distribution (residuals from population model)
        2. Apply site multiplicative effect (delta)
        3. Apply site additive effect (gamma)  
        4. Scale by population standard deviation (sigma)
        5. Add back population slope and intercept effects
        
        Args:
            baseline_data: Normal distribution array as starting point 
                          Shape: (n_regions, n_samples)
            X: Covariate matrix of shape (n_features, n_samples) 
            population_noise: Overall population variance component (unused in corrected version)
            
        Returns:
            Site-specific data with proper batch effects
            Shape: (n_regions, n_samples)
        """
        # Convert inputs to arrays and ensure they are 2D
        X = np.asarray(X)
        baseline_data = np.asarray(baseline_data)
        
        # Ensure X is 2D with shape (n_features, n_samples)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Make (n_samples,) into (1, n_samples)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array. Got {X.ndim}D")
        
        # Ensure baseline_data is 2D with shape (n_regions, n_samples)
        if baseline_data.ndim == 1:
            baseline_data = baseline_data.reshape(1, -1)  # Make (n_samples,) into (1, n_samples)
        elif baseline_data.ndim != 2:
            raise ValueError(f"baseline_data must be 1D or 2D array. Got {baseline_data.ndim}D")

        # Validate dimensions match
        n_features, n_samples = X.shape
        n_regions, n_samples_baseline = baseline_data.shape
        
        if n_samples != n_samples_baseline:
            raise ValueError(f"Number of samples mismatch: X has {n_samples} samples, baseline_data has {n_samples_baseline} samples")
        
        # Validate site parameters match number of features
        if len(self.gamma) != n_features:
            raise ValueError(f"gamma must have length {n_features} (number of features). Got length {len(self.gamma)}")
        if len(self.delta) != n_features:
            raise ValueError(f"delta must have length {n_features} (number of features). Got length {len(self.delta)}")
        if len(self.beta) != n_features:
            raise ValueError(f"beta must have length {n_features} (number of features). Got length {len(self.beta)}")
        
        # Step 1: Start with baseline data (already provided)
        
        # Step 2: Apply site effects correctly following the model Y = α + β*X + σ*(δ*ε + γ*X)
        # For each feature, we need to apply its specific site effects
        
        # Calculate base population effects: α + β*X
        base_covariate_effects = self.alpha + np.sum(self.beta[:, np.newaxis] * X, axis=0)  # Shape: (n_samples,)
        
        # Calculate site effects: σ*(δ*ε + γ*X)
        # δ*ε term: multiply baseline noise by the average delta (for regions)
        # Since delta can vary per feature but baseline noise is per region, use first delta for noise
        delta_noise_term = baseline_data * self.delta[0]  # Shape: (n_regions, n_samples)
        
        # γ*X term: sum over features of gamma[i] * X[i]
        gamma_covariate_term = np.sum(self.gamma[:, np.newaxis] * X, axis=0)  # Shape: (n_samples,)
        gamma_covariate_term = gamma_covariate_term[np.newaxis, :]  # Shape: (1, n_samples) -> broadcasts to (n_regions, n_samples)
        
        # Combine site effects and scale by sigma
        site_effects = self.sigma * (delta_noise_term + gamma_covariate_term)  # Shape: (n_regions, n_samples)
        
        # Final result: population effects + site effects
        final_data = base_covariate_effects[np.newaxis, :] + site_effects  # Shape: (n_regions, n_samples)
        
        return final_data

    def generate_site_data(self, X: ArrayLike, baseline_noise: ArrayLike = None, n_regions: int = None) -> ArrayLike:
        """
        Generate synthetic data with site effects using the correct process
        
        This follows the proper data generation model:
        1. Start with normal distribution (baseline residuals)
        2. Apply site effects (delta, gamma)
        3. Scale by population standard deviation
        4. Add population-level slope and intercept
        
        Args:
            X: Covariate matrix of shape (n_features, n_samples)
            baseline_noise: Optional baseline normal distribution; if None, generates from standard normal
                          Shape: (n_regions, n_samples) for multi-region data
            n_regions: Number of brain regions/ROIs. If None, defaults to 1 (single region)
            
        Returns:
            Site-specific data with proper batch effects
            Shape: (n_regions, n_samples)
        """
        # Convert X to array and ensure it's 2D
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Make (n_samples,) into (1, n_samples)
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D array. Got {X.ndim}D")
        
        n_features, n_samples = X.shape
        
        # Validate that site parameters match number of features
        if len(self.gamma) != n_features:
            raise ValueError(f"gamma must have length {n_features} (number of features). Got length {len(self.gamma)}")
        if len(self.delta) != n_features:
            raise ValueError(f"delta must have length {n_features} (number of features). Got length {len(self.delta)}")
        if len(self.beta) != n_features:
            raise ValueError(f"beta must have length {n_features} (number of features). Got length {len(self.beta)}")
        
        # Determine number of regions
        if n_regions is None:
            if baseline_noise is not None:
                baseline_noise = np.asarray(baseline_noise)
                if baseline_noise.ndim == 2:
                    n_regions = baseline_noise.shape[0]
                elif baseline_noise.ndim == 1:
                    n_regions = 1  # Single region when baseline_noise is 1D
                else:
                    raise ValueError(f"baseline_noise must be 1D or 2D array. Got {baseline_noise.ndim}D")
            else:
                n_regions = 1  # Default: single region when no baseline_noise provided
        
        # Generate or validate baseline normal distribution
        if baseline_noise is None:
            rng = np.random.default_rng()
            baseline_noise = rng.normal(0, 1, (n_regions, n_samples))
        else:
            baseline_noise = np.asarray(baseline_noise)
            if baseline_noise.ndim == 1:
                if n_regions != 1:
                    raise ValueError(f"baseline_noise is 1D but n_regions={n_regions}. For multiple regions, baseline_noise must be 2D")
                baseline_noise = baseline_noise.reshape(1, -1)  # Make (n_samples,) into (1, n_samples)
            elif baseline_noise.ndim == 2:
                if baseline_noise.shape != (n_regions, n_samples):
                    raise ValueError(f"baseline_noise shape {baseline_noise.shape} doesn't match expected ({n_regions}, {n_samples})")
            else:
                raise ValueError(f"baseline_noise must be 1D or 2D array. Got {baseline_noise.ndim}D")
        
        # Apply site effects using the corrected process
        return self.add_site_effects(baseline_noise, X, None)


class PairwiseComBAT:
    """
    Implementation of Pairwise-ComBAT algorithm based on the paper
    "ComBAT Harmonization for diffusion MRI: Challenges and Best Practices"
    
    This implements the mathematical formulation from Equations 1-17 in the paper.
    Follows scikit-learn conventions with fit() and predict() methods.
    """
    
    def __init__(self, max_iter: int = 30, tol: float = 1e-6):
        """
        Initialize Pairwise-ComBAT harmonizer
        
        Args:
            max_iter: Maximum iterations for Bayesian estimation
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        
        # Parameters fitted during training
        self.alpha_hat_ = None
        self.beta_hat_ = None
        self.sigma_hat_ = None
        self.is_fitted_ = False
        
    def _normalize_covariate_shape(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize covariate array to 2D shape (n_samples, n_features)
        
        Adds dummy dimension if input is 1D to ensure consistent 2D processing.
        Does not perform automatic transposition - expects correct orientation.
        
        Args:
            X: Input covariate array of shape (n_samples,) or (n_samples, n_features)
            
        Returns:
            Normalized array of shape (n_samples, n_features) with n_features >= 1
        """
        # If 1D, reshape to (n_samples, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            # Ensure it's at least 2D
            X = np.atleast_2d(X)
            
        return X
    
    def _normalize_response_shape(self, Y: np.ndarray) -> np.ndarray:
        """
        Normalize response array to 2D shape (n_regions, n_samples)
        
        Adds dummy dimension if input is 1D to ensure consistent 2D processing.
        Does not perform automatic transposition - expects correct orientation.
        
        Args:
            Y: Input response array of shape (n_samples,) or (n_regions, n_samples)
            
        Returns:
            Normalized array of shape (n_regions, n_samples) with n_regions >= 1
        """
        # If 1D, reshape to (1, n_samples) for single region
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        else:
            # Ensure it's at least 2D
            Y = np.atleast_2d(Y)
            
        return Y
        
    def _estimate_global_parameters(self, X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """
        Estimate global parameters alpha_hat, beta_hat, and sigma_hat using OLS
        
        This implements the parameter estimation from Equation (3) in the paper:
        [alpha_hat, beta_hat] = (X^T X)^{-1} X^T Y
        
        For multi-region data, this is applied to all regions pooled together.
        
        Args:
            X: Covariate matrix (n_samples, n_covariates) 
            Y: Response matrix (n_regions, n_samples) for multi-region data
            
        Returns:
            Tuple of (alpha_hat, beta_hat, sigma_hat)
            - alpha_hat: scalar intercept
            - beta_hat: (n_covariates,) slope coefficients  
            - sigma_hat: scalar residual standard deviation
        """
        # X and Y are already normalized to 2D by caller
        n_regions, n_samples = Y.shape
        
        # Reshape to (n_regions * n_samples,) and replicate X accordingly
        Y_pooled = Y.T.flatten()  # Flatten to (n_regions * n_samples,)
        X_pooled = np.tile(X, (n_regions, 1))  # Replicate X for each region
        
        # Fit linear regression on pooled data
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X_pooled, Y_pooled)
        
        alpha_hat = reg.intercept_
        beta_hat = reg.coef_[0] if len(reg.coef_) == 1 else reg.coef_
        
        # Calculate residual standard deviation from pooled residuals
        Y_pred_pooled = reg.predict(X_pooled)
        residuals_pooled = Y_pooled - Y_pred_pooled
        sigma_hat = np.sqrt(np.sum(residuals_pooled**2) / (len(Y_pooled) - X_pooled.shape[1] - 1))
        
        return alpha_hat, beta_hat, sigma_hat
    
    def _standardize_data(self, X: np.ndarray, Y: np.ndarray, 
                         alpha_hat: float, beta_hat: np.ndarray, sigma_hat: float) -> np.ndarray:
        """
        Standardize data according to Equation (4) in the paper:
        z_ijv = (y_ijv - alpha_hat - x_ij^T * beta_hat) / sigma_hat
        
        For multi-region data, standardization is applied to each region independently.
        
        Args:
            X: Covariate matrix (n_samples, n_covariates)
            Y: Response data (n_regions, n_samples)
            alpha_hat, beta_hat, sigma_hat: Global parameters
            
        Returns:
            Standardized data z with same shape as Y
        """
        # X and Y are already normalized to 2D by caller
        
        # Compute predicted values from covariates
        predicted = alpha_hat + X.dot(np.atleast_1d(beta_hat))
        
        # Y is (n_regions, n_samples), predicted is (n_samples,)
        # Broadcast predicted to match Y shape
        z = (Y - predicted[np.newaxis, :]) / sigma_hat
        
        return z
    
    def _estimate_site_parameters(self, z_site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate site-specific parameters gamma_star and delta_star for standardized data
        
        This provides initial estimates before Bayesian refinement.
        Parameters are estimated for each region independently.
        
        Args:
            z_site: Standardized data for a specific site
                   Shape: (n_regions, n_samples)
            
        Returns:
            Tuple of (gamma_star_hat, delta_star_hat)
            - gamma_star_hat: (n_regions,) array
            - delta_star_hat: (n_regions,) array
        """
        # z_site is always (n_regions, n_samples) due to normalization
        gamma_star_hat = np.mean(z_site, axis=1)  # Mean across samples for each region
        delta_star_hat_sq = np.var(z_site, axis=1, ddof=1)  # Variance across samples for each region
        
        # Handle case where variance is zero for some regions
        delta_star_hat_sq = np.where(delta_star_hat_sq > 0, delta_star_hat_sq, 1.0)
        
        return gamma_star_hat, np.sqrt(delta_star_hat_sq)
    
    def _estimate_hyperparameters(self, gamma_stars: np.ndarray, delta_stars_sq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate hyperparameters for Bayesian estimation 
        
        This implements Equations (11-13) from the paper for estimating
        the hyperparameters of the prior distributions.
        For multi-region data, hyperparameters are estimated for each region independently.
        
        Args:
            gamma_stars: Array of gamma_star estimates
                        Shape: (n_sites,) for single region or (n_regions, n_sites) for multi-region
            delta_stars_sq: Array of delta_star^2 estimates  
                           Shape: (n_sites,) for single region or (n_regions, n_sites) for multi-region
            
        Returns:
            Tuple of (mu_bar, tau_sq_bar, lambda_bar, theta_bar)
            Each is scalar for single region or (n_regions,) array for multi-region
        """
        if gamma_stars.ndim == 1:
            # Single region case - treat as one site with scalar values
            mu_bar = np.mean(gamma_stars)
            tau_sq_bar = np.var(gamma_stars, ddof=1) if len(gamma_stars) > 1 else 1.0
            
            # For delta (Equation 12)
            G_bar = np.mean(delta_stars_sq)
            S_sq_bar = np.var(delta_stars_sq, ddof=1) if len(delta_stars_sq) > 1 else G_bar * 0.1
            
            # Calculate lambda and theta (Equation 13)
            if S_sq_bar > 0:
                lambda_bar = (G_bar**2 + 2*S_sq_bar) / S_sq_bar
                theta_bar = (G_bar**3 + G_bar*S_sq_bar) / S_sq_bar
            else:
                # Fallback when variance is zero
                lambda_bar = 2.1  # Minimum for inverse gamma to have finite mean
                theta_bar = G_bar * (lambda_bar - 1)
                
        else:
            # Multi-region case: gamma_stars is (n_regions, n_sites)
            mu_bar = np.mean(gamma_stars, axis=1)  # Mean across sites for each region
            tau_sq_bar = np.var(gamma_stars, axis=1, ddof=1)
            tau_sq_bar = np.where(tau_sq_bar > 0, tau_sq_bar, 1.0)  # Handle zero variance
            
            # For delta (Equation 12)
            G_bar = np.mean(delta_stars_sq, axis=1)  # Mean across sites for each region
            S_sq_bar = np.var(delta_stars_sq, axis=1, ddof=1)
            S_sq_bar = np.where(S_sq_bar > 0, S_sq_bar, G_bar * 0.1)  # Handle zero variance
            
            # Calculate lambda and theta (Equation 13)
            lambda_bar = np.where(S_sq_bar > 0, 
                                 (G_bar**2 + 2*S_sq_bar) / S_sq_bar,
                                 2.1)  # Fallback for each region
            theta_bar = np.where(S_sq_bar > 0,
                               (G_bar**3 + G_bar*S_sq_bar) / S_sq_bar,
                               G_bar * (lambda_bar - 1))
        
        return mu_bar, tau_sq_bar, lambda_bar, theta_bar
    
    def _bayesian_site_estimation(self, z_site: np.ndarray, mu_bar: np.ndarray, tau_sq_bar: np.ndarray,
                                 lambda_bar: np.ndarray, theta_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian estimation of site effects using Equations (14-15) from the paper
        
        This implements the iterative Bayesian estimation procedure for refining
        the site-specific parameters using prior information.
        Estimation is performed for each region independently.
        
        Args:
            z_site: Standardized data for site
                   Shape: (n_regions, n_samples) or (n_samples,) for single region
            mu_bar, tau_sq_bar, lambda_bar, theta_bar: Hyperparameters
                   Each is (n_regions,) array or scalar for single region
            
        Returns:
            Tuple of (gamma_star_final, delta_star_final)
            Each is (n_regions,) array or scalar for single region
        """
        # Ensure z_site is always 2D: (n_regions, n_samples)
        if z_site.ndim == 1:
            z_site = z_site.reshape(1, -1)  # Convert (n_samples,) to (1, n_samples)
        
        n_regions, n_samples = z_site.shape
        
        # Ensure hyperparameters are arrays
        mu_bar = np.atleast_1d(mu_bar)
        tau_sq_bar = np.atleast_1d(tau_sq_bar)
        lambda_bar = np.atleast_1d(lambda_bar)
        theta_bar = np.atleast_1d(theta_bar)
        
        # Initial estimates for each region
        gamma_star_emp = np.mean(z_site, axis=1)  # (n_regions,)
        delta_star_sq_emp = np.var(z_site, axis=1, ddof=1)  # (n_regions,)
        delta_star_sq_emp = np.where(delta_star_sq_emp > 0, delta_star_sq_emp, 1.0)
        
        # Iterative estimation for each region
        gamma_star = gamma_star_emp.copy()
        delta_star_sq = delta_star_sq_emp.copy()
        
        for _ in range(self.max_iter):
            gamma_star_old = gamma_star.copy()
            delta_star_sq_old = delta_star_sq.copy()
            
            # Update gamma_star (Equation 14) for each region
            valid_tau = tau_sq_bar > 0
            numerator = n_samples * tau_sq_bar * gamma_star_emp + delta_star_sq * mu_bar
            denominator = n_samples * tau_sq_bar + delta_star_sq
            gamma_star = np.where(valid_tau, numerator / denominator, gamma_star_emp)
            
            # Update delta_star_sq (Equation 15) for each region
            residuals_sq = np.sum((z_site - gamma_star[:, np.newaxis])**2, axis=1)  # (n_regions,)
            numerator = theta_bar + 0.5 * residuals_sq
            denominator = n_samples/2 + lambda_bar - 1
            valid_denom = denominator > 0
            delta_star_sq = np.where(valid_denom, numerator / denominator, delta_star_sq_emp)
            
            # Check convergence for all regions
            gamma_converged = np.all(np.abs(gamma_star - gamma_star_old) < self.tol)
            delta_converged = np.all(np.abs(delta_star_sq - delta_star_sq_old) < self.tol)
            
            if gamma_converged and delta_converged:
                break
                
        # Return scalar values if input was 1D (single region)
        if len(gamma_star) == 1:
            return gamma_star[0], np.sqrt(np.maximum(delta_star_sq[0], 1e-6))
        else:
            return gamma_star, np.sqrt(np.maximum(delta_star_sq, 1e-6))
    
    def fit(self, X_ref: np.ndarray, Y_ref: np.ndarray,
            X_moving: np.ndarray, Y_moving: np.ndarray) -> 'PairwiseComBAT':
        """
        Fit the PairwiseComBAT harmonizer using reference and moving site data
        
        This learns the global parameters (alpha_hat, beta_hat, sigma_hat) that will
        be used for harmonizing future data.
        
        Args:
            X_ref: Covariates for reference site (n_ref, n_covariates)
            Y_ref: Response values for reference site 
                   Shape: (n_regions, n_ref) for multi-region data
            X_moving: Covariates for moving site (n_moving, n_covariates)  
            Y_moving: Response values for moving site
                     Shape: (n_regions, n_moving) for multi-region data
            
        Returns:
            self: Fitted harmonizer instance
        """
        # Normalize input shapes to ensure 2D arrays with dummy dimensions if needed
        X_ref = self._normalize_covariate_shape(X_ref)
        X_moving = self._normalize_covariate_shape(X_moving)
        Y_ref = self._normalize_response_shape(Y_ref)
        Y_moving = self._normalize_response_shape(Y_moving)
        
        # Validate that number of samples matches between X and Y matrices
        assert X_ref.shape[0] == Y_ref.shape[1], f"Number of samples mismatch: X_ref has {X_ref.shape[0]} samples, Y_ref has {Y_ref.shape[1]} samples"
        assert X_moving.shape[0] == Y_moving.shape[1], f"Number of samples mismatch: X_moving has {X_moving.shape[0]} samples, Y_moving has {Y_moving.shape[1]} samples"
            
        # Combine data for global parameter estimation
        X_combined = np.vstack([X_ref, X_moving])
        
        # Concatenate along samples dimension (axis=1)
        Y_combined = np.concatenate([Y_ref, Y_moving], axis=1)
        
        # Estimate global parameters (Equation 3)
        alpha_hat, beta_hat, sigma_hat = self._estimate_global_parameters(X_combined, Y_combined)
        self.alpha_hat_ = alpha_hat
        self.beta_hat_ = beta_hat  
        self.sigma_hat_ = sigma_hat
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X_moving: np.ndarray, Y_moving: np.ndarray, 
                gamma_ref: float = 0.0, delta_ref: float = 1.0) -> np.ndarray:
        """
        Harmonize moving site data to reference site characteristics
        
        This applies the full Pairwise-ComBAT harmonization procedure using the
        fitted global parameters to harmonize new data.
        
        Args:
            X_moving: Covariates for data to harmonize (n_samples, n_features)
            Y_moving: Response values for data to harmonize
                     Shape: (n_regions, n_samples) for multi-region data
            gamma_ref: Target reference site gamma (default: 0.0)
            delta_ref: Target reference site delta (default: 1.0)
            
        Returns:
            Y_harmonized: Harmonized response values with same shape as Y_moving
        """
        self._check_is_fitted()
        
        # Normalize input shapes to ensure 2D arrays
        X_moving = self._normalize_covariate_shape(X_moving)
        Y_moving = self._normalize_response_shape(Y_moving)
        
        # Validate that number of samples matches between X and Y matrices
        assert X_moving.shape[0] == Y_moving.shape[1], f"Number of samples mismatch: X_moving has {X_moving.shape[0]} samples, Y_moving has {Y_moving.shape[1]} samples"
        
        # Standardize the moving data using fitted global parameters
        z_moving = self._standardize_data(X_moving, Y_moving, self.alpha_hat_, self.beta_hat_, self.sigma_hat_)
        
        # Estimate site parameters for the moving data
        gamma_moving_init, delta_moving_init = self._estimate_site_parameters(z_moving)
        
        # For inference, we use simple parameter estimates without Bayesian refinement
        # since we don't have multiple sites to estimate hyperparameters from
        gamma_moving_final = gamma_moving_init
        delta_moving_final = delta_moving_init
        
        # Apply harmonization to target reference characteristics
        predicted_moving = self.alpha_hat_ + X_moving.dot(np.atleast_1d(self.beta_hat_))
        
        # Apply harmonization for all regions
        n_regions = Y_moving.shape[0]
        z_harmonized = np.zeros_like(z_moving)
        Y_harmonized = np.zeros_like(Y_moving)
        
        for r in range(n_regions):
            # Transform to target reference site characteristics for this region
            z_harmonized[r] = (delta_ref / delta_moving_final[r]) * (z_moving[r] - gamma_moving_final[r]) + gamma_ref
            
            # Convert back to original scale for this region
            Y_harmonized[r] = self.sigma_hat_ * z_harmonized[r] + predicted_moving
        
        return Y_harmonized
    
    def fit_predict(self, X_ref: np.ndarray, Y_ref: np.ndarray,
                    X_moving: np.ndarray, Y_moving: np.ndarray,
                    gamma_ref: float = None, delta_ref: float = None) -> np.ndarray:
        """
        Fit the harmonizer and predict harmonized values in one step
        
        This method trains on the provided reference and moving site data, then
        harmonizes the moving site data to the reference site characteristics.
        
        Args:
            X_ref: Covariates for reference site (n_ref, n_covariates)
            Y_ref: Response values for reference site
                   Shape: (n_regions, n_ref) for multi-region data
            X_moving: Covariates for moving site (n_moving, n_covariates)  
            Y_moving: Response values for moving site
                     Shape: (n_regions, n_moving) for multi-region data
            gamma_ref: Target reference site gamma (None = use estimated from ref data)
            delta_ref: Target reference site delta (None = use estimated from ref data)
            
        Returns:
            Y_harmonized: Harmonized response values for moving site with same shape as Y_moving
        """
        # Normalize input shapes to ensure 2D arrays
        X_ref = self._normalize_covariate_shape(X_ref)
        X_moving = self._normalize_covariate_shape(X_moving)
        Y_ref = self._normalize_response_shape(Y_ref)
        Y_moving = self._normalize_response_shape(Y_moving)
        
        # Validate that number of samples matches between X and Y matrices
        assert X_ref.shape[0] == Y_ref.shape[1], f"Number of samples mismatch: X_ref has {X_ref.shape[0]} samples, Y_ref has {Y_ref.shape[1]} samples"
        assert X_moving.shape[0] == Y_moving.shape[1], f"Number of samples mismatch: X_moving has {X_moving.shape[0]} samples, Y_moving has {Y_moving.shape[1]} samples"
        
        # Fit the model
        self.fit(X_ref, Y_ref, X_moving, Y_moving)
        
        # If target reference parameters not specified, estimate them from reference data
        if gamma_ref is None or delta_ref is None:
            z_ref = self._standardize_data(X_ref, Y_ref, self.alpha_hat_, self.beta_hat_, self.sigma_hat_)
            gamma_ref_est, delta_ref_est = self._estimate_site_parameters(z_ref)
            
            if gamma_ref is None:
                gamma_ref = gamma_ref_est
            if delta_ref is None:
                delta_ref = delta_ref_est
        
        # Standardize moving data and estimate its site parameters
        z_moving = self._standardize_data(X_moving, Y_moving, self.alpha_hat_, self.beta_hat_, self.sigma_hat_)
        gamma_moving_init, delta_moving_init = self._estimate_site_parameters(z_moving)
        
        # Apply harmonization for all regions
        n_regions = Y_ref.shape[0]
        Y_harmonized = np.zeros_like(Y_moving)
        predicted_moving = self.alpha_hat_ + X_moving.dot(np.atleast_1d(self.beta_hat_))
        
        for r in range(n_regions):
            # For each region, estimate hyperparameters and apply Bayesian refinement
            gamma_estimates = np.array([gamma_ref[r] if hasattr(gamma_ref, '__len__') else gamma_ref, 
                                      gamma_moving_init[r]])
            delta_estimates_sq = np.array([(delta_ref[r] if hasattr(delta_ref, '__len__') else delta_ref)**2, 
                                         delta_moving_init[r]**2])
            
            mu_bar, tau_sq_bar, lambda_bar, theta_bar = self._estimate_hyperparameters(
                gamma_estimates, delta_estimates_sq)
            
            # Bayesian refinement for both sites for this region
            z_ref_r = z_ref[r] if 'z_ref' in locals() else self._standardize_data(X_ref, Y_ref, self.alpha_hat_, self.beta_hat_, self.sigma_hat_)[r]
            gamma_ref_final, delta_ref_final = self._bayesian_site_estimation(
                z_ref_r, mu_bar, tau_sq_bar, lambda_bar, theta_bar)
            gamma_moving_final, delta_moving_final = self._bayesian_site_estimation(
                z_moving[r], mu_bar, tau_sq_bar, lambda_bar, theta_bar)
            
            # Apply harmonization for this region (Equation 17)
            z_harmonized = (delta_ref_final / delta_moving_final) * (z_moving[r] - gamma_moving_final) + gamma_ref_final
            Y_harmonized[r] = self.sigma_hat_ * z_harmonized + predicted_moving
        
        return Y_harmonized
    
    def _check_is_fitted(self):
        """Check if the harmonizer has been fitted"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
    
    def get_estimated_parameters(self) -> dict:
        """
        Return the estimated ComBAT parameters
        
        Returns:
            Dictionary containing alpha_hat, beta_hat, sigma_hat
        """
        self._check_is_fitted()
        return {
            'alpha_hat': self.alpha_hat_,
            'beta_hat': self.beta_hat_, 
            'sigma_hat': self.sigma_hat_
        }
    
    def get_site_effects(self, X_site: np.ndarray, Y_site: np.ndarray) -> dict:
        """
        Estimate site-specific gamma and delta effects for a given site
        
        Args:
            X_site: Site covariates
            Y_site: Site response values
            
        Returns:
            Dictionary containing gamma_star and delta_star estimates
        """
        self._check_is_fitted()
        
        # Normalize input shapes to ensure 2D arrays
        X_site = self._normalize_covariate_shape(X_site)
        Y_site = self._normalize_response_shape(Y_site)
        
        # Validate that number of samples matches between X and Y matrices
        assert X_site.shape[0] == Y_site.shape[1], f"Number of samples mismatch: X_site has {X_site.shape[0]} samples, Y_site has {Y_site.shape[1]} samples"
            
        # Standardize the site data
        z_site = self._standardize_data(X_site, Y_site, self.alpha_hat_, self.beta_hat_, self.sigma_hat_)
        
        # Get site parameter estimates
        gamma_star, delta_star = self._estimate_site_parameters(z_site)
        
        return {
            'gamma_star': gamma_star,
            'delta_star': delta_star
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained PairwiseComBAT model to a single HDF5 file
        
        This saves all necessary parameters for inference including:
        - Global parameters (alpha_hat, beta_hat, sigma_hat)
        - Model configuration (max_iter, tol)
        - Metadata (version, creation time, etc.)
        
        Args:
            filepath: Path where to save the model (will add .h5 extension if not present)
        """
        self._check_is_fitted()
        
        # Ensure .h5 extension
        filepath = Path(filepath)
        if filepath.suffix != '.h5':
            filepath = filepath.with_suffix('.h5')
        
        with h5py.File(filepath, 'w') as f:
            # Create groups for organization
            params_group = f.create_group('parameters')
            config_group = f.create_group('config')
            metadata_group = f.create_group('metadata')
            
            # Save numerical parameters
            params_group.create_dataset('alpha_hat', data=self.alpha_hat_)
            params_group.create_dataset('beta_hat', data=np.atleast_1d(self.beta_hat_))
            params_group.create_dataset('sigma_hat', data=self.sigma_hat_)
            
            # Save configuration
            config_group.attrs['max_iter'] = self.max_iter
            config_group.attrs['tol'] = float(self.tol)
            
            # Save metadata
            metadata_group.attrs['version'] = '1.0'
            metadata_group.attrs['model_type'] = 'PairwiseComBAT'
            metadata_group.attrs['creation_time'] = datetime.now().isoformat()
            metadata_group.attrs['n_features'] = len(np.atleast_1d(self.beta_hat_))
            metadata_group.attrs['fitted'] = True
            
        print(f"Model saved to {filepath}")
        print("  Model version: 1.0")
        print(f"  Features: {len(np.atleast_1d(self.beta_hat_))}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained PairwiseComBAT model from HDF5 file
        
        Args:
            filepath: Path to the saved model (.h5 extension will be added if not present)
        """
        # Ensure .h5 extension
        filepath = Path(filepath)
        if filepath.suffix != '.h5':
            filepath = filepath.with_suffix('.h5')
        
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Validate model type
            model_type = f['metadata'].attrs.get('model_type', '').decode() if isinstance(f['metadata'].attrs.get('model_type'), bytes) else f['metadata'].attrs.get('model_type', '')
            if model_type != 'PairwiseComBAT':
                raise ValueError(f"Invalid model type: {model_type}")
            
            # Load numerical parameters
            self.alpha_hat_ = float(f['parameters']['alpha_hat'][()])
            self.beta_hat_ = f['parameters']['beta_hat'][:]
            self.sigma_hat_ = float(f['parameters']['sigma_hat'][()])
            
            # Load configuration
            self.max_iter = int(f['config'].attrs['max_iter'])
            self.tol = float(f['config'].attrs['tol'])
            
            # Mark as fitted
            self.is_fitted_ = True
            
            # Extract metadata for display
            version = f['metadata'].attrs.get('version', 'unknown')
            creation_time = f['metadata'].attrs.get('creation_time', 'unknown')
            n_features = int(f['metadata'].attrs.get('n_features', 0))
            
            if isinstance(version, bytes):
                version = version.decode()
            if isinstance(creation_time, bytes):
                creation_time = creation_time.decode()
        
        print(f"Model loaded from {filepath}")
        print(f"  Model version: {version}")
        print(f"  Created: {creation_time}")
        print(f"  Features: {n_features}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
    
    @classmethod
    def load_and_predict(cls, model_path: str, X_moving: np.ndarray, Y_moving: np.ndarray,
                        gamma_ref: float = 0.0, delta_ref: float = 1.0) -> np.ndarray:
        """
        Convenience method to load a model and make predictions in one step
        
        Args:
            model_path: Path to the saved model
            X_moving: Covariates for data to harmonize
            Y_moving: Response values for data to harmonize  
            gamma_ref: Target reference site gamma
            delta_ref: Target reference site delta
            
        Returns:
            Y_harmonized: Harmonized response values
        """
        model = cls()
        model.load_model(model_path)
        return model.predict(X_moving, Y_moving, gamma_ref, delta_ref)


def compute_harmonization_mse(X_ref: np.ndarray, Y_ref: np.ndarray,
                             X_moving: np.ndarray, Y_harmonized: np.ndarray,
                             alpha_hat: float, beta_hat: np.ndarray) -> float:
    """
    Compute Mean Squared Error between reference and harmonized data
    
    This measures how well the harmonized data matches the reference distribution
    by comparing their residual distributions after removing covariate effects.
    
    Args:
        X_ref: Reference site covariates (n_ref, n_features)
        Y_ref: Reference site response values (n_ref,)
        X_moving: Moving site covariates (n_moving, n_features)
        Y_harmonized: Harmonized moving site response values (n_moving,)
        alpha_hat: Estimated intercept parameter
        beta_hat: Estimated slope parameters
        
    Returns:
        MSE between reference and harmonized residual distributions
    """
    # Ensure proper array shapes
    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_moving.ndim == 1:
        X_moving = X_moving.reshape(-1, 1)
        
    # Compute predicted values using global parameters
    pred_ref = alpha_hat + X_ref.dot(np.atleast_1d(beta_hat))
    pred_moving = alpha_hat + X_moving.dot(np.atleast_1d(beta_hat))
    
    # Compute residuals after removing covariate effects
    residuals_ref = Y_ref - pred_ref
    residuals_harmonized = Y_harmonized - pred_moving
    
    # Compare distribution properties (mean and variance)
    mean_diff_sq = (np.mean(residuals_ref) - np.mean(residuals_harmonized))**2
    var_diff_sq = (np.var(residuals_ref) - np.var(residuals_harmonized))**2
    
    return mean_diff_sq + var_diff_sq


def make_raw_data(n_features: int = 1, alpha: float = 0.5, beta: ArrayLike = None, sigma: float = 0.5) -> Tuple[ArrayLike, ArrayLike]:
    """Generate raw data without site effects
    
    Args:
        n_features: Number of features/covariates to generate
        alpha: Population intercept parameter
        beta: Population slope parameters for each feature. If None, defaults to [0.04] for single feature or [0.04, -0.02, 0.01, ...] for multiple
        sigma: Population noise standard deviation
        
    Returns:
        Tuple of (X, Y) where X is shape (n_features, n_samples) and Y is shape (n_samples,)
    """
    rng = np.random.default_rng()
    
    # Set default beta values if not provided
    if beta is None:
        if n_features == 1:
            beta = np.array([0.04])
        else:
            # Generate different beta values for different features
            beta = np.array([0.04, -0.02, 0.01] + [0.02] * max(0, n_features - 3))[:n_features]
    else:
        beta = np.atleast_1d(beta)
        if len(beta) != n_features:
            raise ValueError(f"beta must have length {n_features} (number of features). Got length {len(beta)}")
    
    if n_features == 1:
        # For backward compatibility, generate ages as before
        X = rng.integers(low=MIN_AGE, high=MAX_AGE, size=(1, N_SAMPLES_PER_SITE))
    else:
        # Generate multiple features with different ranges
        X = np.zeros((n_features, N_SAMPLES_PER_SITE))
        for i in range(n_features):
            # Use different ranges for different features
            low = MIN_AGE + i * 10
            high = MAX_AGE + i * 10
            X[i] = rng.uniform(low=low, high=high, size=N_SAMPLES_PER_SITE)
    
    noise = rng.normal(loc=0, scale=sigma, size=N_SAMPLES_PER_SITE)
    data = np.dot(beta, X) + alpha + noise
    return X, data


def generate_age_covariates(n_samples: int, seed: int = None) -> np.ndarray:
    """Helper function to generate age covariates in the expected format
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Age covariates as shape (1, n_samples) for single-feature compatibility
    """
    rng = np.random.default_rng(seed)
    ages = rng.integers(low=MIN_AGE, high=MAX_AGE, size=n_samples)
    return ages.reshape(1, -1)


# All demonstrations and testing functionality has been moved to test_pairwise_combat.py
# Run pytest to execute comprehensive tests of the Pairwise-ComBAT implementation
