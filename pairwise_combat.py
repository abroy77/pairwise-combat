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


class PairwiseComBAT:
    """
    Implementation of Pairwise-ComBAT algorithm based on the paper
    "ComBAT Harmonization for diffusion MRI: Challenges and Best Practices"

    This implements the mathematical formulation from Equations 1-17 in the paper.
    Follows scikit-learn conventions with fit() and predict() methods.
    """

    # metadata
    source_site: str
    target_site: str
    # Training Data parameters
    n_samples_ref: int
    n_samples_moving: int
    n_locations: int
    n_covariates: int
    # Global parameters estimated during training
    alpha_hat_: ArrayLike
    beta_hat_: ArrayLike
    sigma_hat_: ArrayLike
    # Parameters for Bayesian estimation
    max_iter: int
    tol: float
    # Site-specific parameters for reference and moving sites
    # These are the final estimates after Bayesian refinement
    gamma_star_ref: ArrayLike
    delta_star_ref: ArrayLike
    gamma_star_moving: ArrayLike
    delta_star_moving: ArrayLike
    is_fitted_: bool = False

    def __init__(
        self,
        source_site: str = "source",
        target_site: str = "target",
        max_iter: int = 30,
        tol: float = 1e-6,
    ):
        """
        Initialize Pairwise-ComBAT harmonizer

        Args:
            max_iter: Maximum iterations for Bayesian estimation
            tol: Convergence tolerance
        """
        self.source_site = source_site
        self.target_site = target_site
        self.max_iter = max_iter
        self.tol = tol

        # Initialize site-specific parameters
        self.gamma_star_ref = None
        self.delta_star_ref = None
        self.gamma_star_moving = None
        self.delta_star_moving = None
        # Ensure these are initialized to None
        # to avoid uninitialized access
        self.n_samples_ref = None
        self.n_samples_moving = None
        self.n_locations = None
        # Parameters fitted during training
        self.alpha_hat_ = None
        self.beta_hat_ = None
        self.sigma_hat_ = None
        self.num_locations = None
        self.is_fitted_ = False

    def _estimate_global_parameters(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate global parameters alpha_hat, beta_hat, and sigma_hat using OLS

        This implements the parameter estimation from Equation (3) in the paper,
        but fits a separate regression for each location.

        Args:
            X: Covariate matrix (n_covariates, n_samples)
            Y: Response matrix (n_locations, n_samples)
            n_samples_ref: Number of samples from the reference site (used to split residuals)

        Returns:
            Tuple of (alpha_hat, beta_hat, sigma_hat)
            - alpha_hat: (n_locations,) intercepts
            - beta_hat: (n_locations, n_covariates) slope coefficients
            - sigma_hat: (n_locations,) residual standard deviations
        """
        n_locations, n_samples = Y.shape
        n_covariates = X.shape[0]
        alpha_hat = np.zeros((n_locations, 1))
        beta_hat = np.zeros((n_locations, n_covariates))
        sigma_hat = np.zeros((n_locations, 1))

        # Transpose X to (n_samples, n_covariates) for sklearn
        X_t = X.T

        for r in range(n_locations):
            y = Y[r]
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_t, y)
            alpha_hat[r][0] = reg.intercept_
            beta_hat[r] = reg.coef_
            y_pred = reg.predict(X_t)
            residuals = y - y_pred

            # Split residuals into reference and moving
            residuals_ref = residuals[: self.n_samples_ref]
            residuals_moving = residuals[self.n_samples_ref :]

            # Compute mean for each split
            mean_ref = residuals_ref.mean() if residuals_ref.size > 0 else 0.0
            mean_moving = residuals_moving.mean() if residuals_moving.size > 0 else 0.0

            # Create y_hat for centering
            y_hat = np.concatenate(
                [
                    np.full(residuals_ref.shape, mean_ref),
                    np.full(residuals_moving.shape, mean_moving),
                ]
            )

            # Compute sigma_hat as std of centered residuals
            sigma_hat[r][0] = np.std(residuals - y_hat, ddof=0)

        # returning shapes:
        # alpha_hat: (n_locations, 1), beta_hat: (n_locations, n_covariates), sigma_hat: (n_locations, 1)
        return alpha_hat, beta_hat, sigma_hat

    def _standardize_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha_hat: np.ndarray,
        beta_hat: np.ndarray,
        sigma_hat: np.ndarray,
    ) -> np.ndarray:
        """
        Standardize data according to Equation (4) in the paper:
        z_ijv = (y_ijv - alpha_hat - x_ij^T * beta_hat) / sigma_hat

        Args:
            X: Covariate matrix (n_covariates, n_samples)
            Y: Response data (n_locations, n_samples)
            alpha_hat, beta_hat, sigma_hat: Global parameters (all 2D: (n_locations, 1) or (n_locations, n_covariates))

        Returns:
            Standardized data z with same shape as Y
        """
        # alpha_hat: (n_locations, 1), beta_hat: (n_locations, n_covariates), sigma_hat: (n_locations, 1)
        # X: (n_covariates, n_samples), Y: (n_locations, n_samples)
        predicted = alpha_hat + beta_hat @ X  # (n_locations, n_samples)
        return (Y - predicted) / sigma_hat

    def _estimate_site_parameters(
        self, z_site: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate site-specific parameters gamma_star and delta_star for standardized data

        This provides initial estimates before Bayesian refinement.
        Parameters are estimated for each location independently.

        Args:
            z_site: Standardized data for a specific site
                   Shape: (n_locations, n_samples)

        Returns:
            Tuple of (gamma_star_hat, delta_star_hat)
            - gamma_star_hat: (n_locations,) array
            - delta_star_hat: (n_locations,) array
        """
        # z_site is always (n_locations, n_samples) due to normalization
        gamma_star_hat = np.mean(z_site, axis=1).reshape(
            -1, 1
        )  # Mean across samples for each location
        delta_star_hat = np.std(z_site, axis=1, ddof=1).reshape(
            -1, 1
        )  # Standard deviation across samples for each location

        # Handle case where variance is zero for some locations
        delta_star_hat = np.where(delta_star_hat > 0, delta_star_hat, 1.0)

        # shape of both is (n_locations,1)
        return gamma_star_hat, delta_star_hat

    def _estimate_mu_tau(self, gamma_stars: np.ndarray) -> Tuple[float, float]:
        """
        Estimate mu_bar and tau_sq_bar hyperparameters for Bayesian estimation.

        Args:
            gamma_stars: Array of gamma_star estimates for 1 site (n_locations, 1)

        Returns:
            Tuple of (mu_bar, tau_sq_bar), both scalars
        """
        assert gamma_stars.ndim == 2 and gamma_stars.shape[1] == 1
        mu_bar = float(np.mean(gamma_stars))
        tau_sq_bar = float(np.var(gamma_stars, ddof=1))
        if not np.isfinite(tau_sq_bar) or tau_sq_bar <= 0:
            tau_sq_bar = 1.0
        return mu_bar, tau_sq_bar

    def _estimate_theta_lambda(self, delta_stars: np.ndarray) -> Tuple[float, float]:
        """
        Estimate theta_bar and lambda_bar hyperparameters for Bayesian estimation.

        Args:
            delta_stars: Array of delta_star estimates
                        Shape: (n_locations, 1) for a given site

        Returns:
            Tuple of (lambda_bar, theta_bar)
            Each is scalar for a given site
        """
        assert (
            delta_stars.ndim == 2 and delta_stars.shape[1] == 1
        ), f"delta_stars must be a 2D array with shape (n_locations, 1). Got {delta_stars.shape}"

        # Compute G_bar and S_sq_bar as scalars (mean and variance across all locations)
        G_bar = float(np.mean(delta_stars))
        S_sq_bar = float(np.var(delta_stars, ddof=1))
        if not np.isfinite(S_sq_bar) or S_sq_bar <= 0:
            S_sq_bar = 1.0

        # Ensure lambda_bar is always > 1 and properly scaled
        lambda_bar = (G_bar**2 + 2 * S_sq_bar) / S_sq_bar

        # Compute theta_bar as a scalar
        theta_bar = (G_bar**3 + G_bar * S_sq_bar) / S_sq_bar

        return lambda_bar, theta_bar

    def _bayesian_site_estimation(
        self,
        z_site: np.ndarray,
        mu_bar: float,
        tau_sq_bar: float,
        lambda_bar: float,
        theta_bar: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bayesian estimation of site effects using Equations (14-15) from the paper

        This implements the iterative Bayesian estimation procedure for refining
        the site-specific parameters using prior information.

        Args:
            z_site: Standardized data for site
                   Shape: (n_locations, n_samples)
            mu_bar, tau_sq_bar, lambda_bar, theta_bar: Hyperparameters
                   Each is a scalar for a given site

        Returns:
            Tuple of (gamma_star_final, delta_star_final)
            Each is a (n_locations, 1) array
        """
        n_locations, n_samples = z_site.shape

        # Initial estimates for each location
        gamma_star_emp = np.mean(z_site, axis=1, keepdims=True)  # (n_locations, 1)
        delta_star_sq_emp = np.var(
            z_site, axis=1, ddof=1, keepdims=True
        )  # (n_locations, 1)
        delta_star_sq_emp = np.where(delta_star_sq_emp > 0, delta_star_sq_emp, 1.0)

        # Iterative estimation for each location
        gamma_star = gamma_star_emp.copy()
        delta_star_sq = delta_star_sq_emp.copy()

        for _ in range(self.max_iter):
            gamma_star_old = gamma_star.copy()
            delta_star_sq_old = delta_star_sq.copy()

            # Update gamma_star (Equation 14) for each location
            numerator = n_samples * tau_sq_bar * gamma_star_emp + delta_star_sq * mu_bar
            denominator = n_samples * tau_sq_bar + delta_star_sq
            gamma_star = numerator / denominator

            # Update delta_star_sq (Equation 15) for each location
            residuals_sq = np.sum(
                (z_site - gamma_star) ** 2, axis=1, keepdims=True
            )  # (n_locations, 1)
            numerator = theta_bar + 0.5 * residuals_sq
            denominator = n_samples / 2 + lambda_bar - 1
            # Avoid division by zero or negative denominator
            denominator = np.where(denominator > 0, denominator, 1.0)
            delta_star_sq = numerator / denominator

            # Check convergence for all locations
            gamma_converged = np.all(np.abs(gamma_star - gamma_star_old) < self.tol)
            delta_converged = np.all(
                np.abs(delta_star_sq - delta_star_sq_old) < self.tol
            )

            if gamma_converged and delta_converged:
                break

        # Always return (n_locations, 1) arrays
        return gamma_star, np.sqrt(np.maximum(delta_star_sq, 1e-6))

    def fit(
        self,
        covars_ref: np.ndarray,
        Y_ref: np.ndarray,
        covars_moving: np.ndarray,
        Y_moving: np.ndarray,
    ) -> "PairwiseComBAT":
        """
        Fit the PairwiseComBAT harmonizer using reference and moving site data

        This learns the global parameters (alpha_hat, beta_hat, sigma_hat) that will
        be used for harmonizing future data.

        Args:
            covars_ref: Covariates for reference site (n_covariates, n_samples_ref)
            Y_ref: Response values for reference site
                   Shape: (n_locations, n_samples_ref) for multi-location data
            covars_moving: Covariates for moving site (n_covariates, n_samples_moving)
            Y_moving: Response values for moving site
                     Shape: (n_locations, n_samples_moving) for multi-location data
        Returns:
            self: Fitted harmonizer instance
        """
        # Normalize and validate shapes
        if Y_ref.ndim == 1:  # assuming Y_ref is a single location
            Y_ref = Y_ref.reshape(1, -1)
        if Y_moving.ndim == 1:  # assuming Y_moving is a single location
            Y_moving = Y_moving.reshape(1, -1)
        assert (
            Y_ref.ndim == 2 and Y_ref.shape[0] >= 1 and Y_ref.shape[1] >= 1
        ), f"Y_ref must be a 2D array (n_locations, n_samples). got {Y_ref.shape}"
        assert (
            Y_moving.ndim == 2 and Y_moving.shape[0] >= 1 and Y_moving.shape[1] >= 1
        ), f"Y_moving must be a 2D array (n_locations, n_samples). got {Y_moving.shape}"
        self.n_samples_ref = Y_ref.shape[1]
        self.n_samples_moving = Y_moving.shape[1]

        n_locations_ref = Y_ref.shape[0]
        n_locations_moving = Y_moving.shape[0]
        assert (
            n_locations_moving == n_locations_ref
        ), f"Number of locations mismatch: Y_ref has {n_locations_ref} locations, Y_moving has {n_locations_moving} locations"
        self.n_locations = n_locations_ref

        assert (
            covars_ref.ndim == 2 and covars_ref.shape[1] == self.n_samples_ref
        ), f"covars_ref must be a 2D array (n_covariates, n_samples_ref). got {covars_ref.shape}"
        assert (
            covars_moving.ndim == 2 and covars_moving.shape[1] == self.n_samples_moving
        ), f"X_moving must be a 2D array (n_covariates, n_samples_moving). got {covars_moving.shape}"
        n_covariates_ref = covars_ref.shape[0]
        n_covariates_moving = covars_moving.shape[0]
        assert (
            n_covariates_ref == n_covariates_moving
        ), f"Covariate dimensions mismatch: covars_ref has {n_covariates_ref} covariates, X_moving has {n_covariates_moving} covariates"
        self.n_covariates = n_covariates_ref

        # Combine data for global parameter estimation
        X_combined = np.hstack(
            [covars_ref, covars_moving]
        )  # shape: (n_covariates, n_samples_ref + n_samples_moving)
        Y_combined = np.concatenate(
            [Y_ref, Y_moving], axis=1
        )  # shape: (n_locations, n_samples_ref + n_samples_moving)

        # Estimate global parameters
        alpha_hat, beta_hat, sigma_hat = self._estimate_global_parameters(
            X_combined, Y_combined
        )
        self.alpha_hat_ = alpha_hat
        self.beta_hat_ = beta_hat
        self.sigma_hat_ = sigma_hat

        # now estimate site parameters for both reference and moving sites
        # shape: (n_locations, n_samples)
        z_ref = self._standardize_data(
            covars_ref, Y_ref, self.alpha_hat_, self.beta_hat_, self.sigma_hat_
        )
        z_moving = self._standardize_data(
            covars_moving, Y_moving, self.alpha_hat_, self.beta_hat_, self.sigma_hat_
        )

        # shape: (n_locations, 1)
        # mean and std of samples for each location
        gamma_star_ref, delta_star_ref = self._estimate_site_parameters(z_ref)
        gamma_star_moving, delta_star_moving = self._estimate_site_parameters(z_moving)

        # get mu and tau for both sites
        # scalars
        mu_bar_ref, tau_sq_bar_ref = self._estimate_mu_tau(gamma_star_ref)
        mu_bar_moving, tau_sq_bar_moving = self._estimate_mu_tau(gamma_star_moving)
        # get lambda and theta for both sites
        # scalars
        lambda_bar_ref, theta_bar_ref = self._estimate_theta_lambda(delta_star_ref)
        lambda_bar_moving, theta_bar_moving = self._estimate_theta_lambda(
            delta_star_moving
        )

        # Perform Bayesian estimation for both sites
        gamma_star_ref_final, delta_star_ref_final = self._bayesian_site_estimation(
            z_ref,  # shape: (n_locations, n_samples_ref)
            mu_bar_ref,
            tau_sq_bar_ref,
            lambda_bar_ref,
            theta_bar_ref,
        )
        gamma_star_moving_final, delta_star_moving_final = (
            self._bayesian_site_estimation(
                z_moving,
                mu_bar_moving,
                tau_sq_bar_moving,
                lambda_bar_moving,
                theta_bar_moving,
            )
        )

        # store the final estimates needed for prediction
        self.gamma_star_ref = gamma_star_ref_final
        self.delta_star_ref = delta_star_ref_final
        self.gamma_star_moving = gamma_star_moving_final
        self.delta_star_moving = delta_star_moving_final

        self.is_fitted_ = True
        return self

    def predict(self, covars_moving: np.ndarray, Y_moving: np.ndarray) -> np.ndarray:
        """
        Harmonize moving site data to reference site characteristics

        This applies the full Pairwise-ComBAT harmonization procedure using the
        fitted global parameters to harmonize new data.
        model must have been fitted first

        Args:
            X_moving: Covariates for data to harmonize (n_covariates, n_samples)
            Y_moving: Response values for data to harmonize
                     Shape: (n_locations, n_samples) for multi-location data

        Returns:
            Y_harmonized: Harmonized response values with same shape as Y_moving
        """
        self._check_is_fitted()

        # Normalize input shapes to ensure 2D arrays
        # Manually validate and normalize input shapes to ensure 2D arrays
        if covars_moving.ndim == 1:
            covars_moving = covars_moving.reshape(1, -1)
        if not (
            covars_moving.ndim == 2 and covars_moving.shape[0] == self.n_covariates
        ):
            raise ValueError(
                f"covars_moving must be a 2D array with n_covariates={self.n_covariates}, got shape {covars_moving.shape}"
            )

        if Y_moving.ndim == 1:
            Y_moving = Y_moving.reshape(1, -1)

        if not (Y_moving.ndim == 2 and Y_moving.shape[0] == self.n_locations):
            raise ValueError(
                f"Y_moving must be a 2D array with n_locations={self.n_locations}, got shape {Y_moving.shape}"
            )

        # Standardize the moving data using fitted global parameters
        z_moving = self._standardize_data(
            covars_moving, Y_moving, self.alpha_hat_, self.beta_hat_, self.sigma_hat_
        )

        # Remove moving site additive bias and multiplicative scaling
        z_moving_no_site = (z_moving - self.gamma_star_moving) / self.delta_star_moving

        # Add the target mtiplicative bias and additive bias
        z_moving_to_ref_site = (
            z_moving_no_site * self.delta_star_ref + self.gamma_star_ref
        )

        # add global parameters to transform back to original scale
        out = (
            self.sigma_hat_ * z_moving_to_ref_site
            + self.alpha_hat_
            + self.beta_hat_ @ covars_moving
        )

        assert (
            out.shape == Y_moving.shape
        ), f"Output shape mismatch: expected {Y_moving.shape}, got {out.shape}"

        return out

    def _check_is_fitted(self):
        """Check if the harmonizer has been fitted"""
        if not self.is_fitted_:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )

    def save_model(self, filepath: str) -> None:
        """
        Save the trained PairwiseComBAT model to a single HDF5 file

        This saves all necessary parameters for inference including:
        - Global parameters (alpha_hat, beta_hat, sigma_hat)
        - Site-specific parameters (gamma_star_ref, delta_star_ref, gamma_star_moving, delta_star_moving)
        - Model configuration (max_iter, tol, n_samples_ref, n_samples_moving, n_locations, n_covariates)
        - Metadata (version, creation time, etc.)

        Args:
            filepath: Path where to save the model (will add .h5 extension if not present)
        """
        self._check_is_fitted()

        # Ensure .h5 extension
        filepath = Path(filepath)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")

        with h5py.File(filepath, "w") as f:
            # Create groups for organization
            params_group = f.create_group("parameters")
            config_group = f.create_group("config")
            metadata_group = f.create_group("metadata")

            # Save numerical parameters
            params_group.create_dataset("alpha_hat", data=self.alpha_hat_)
            params_group.create_dataset("beta_hat", data=self.beta_hat_)
            params_group.create_dataset("sigma_hat", data=self.sigma_hat_)
            params_group.create_dataset("gamma_star_ref", data=self.gamma_star_ref)
            params_group.create_dataset("delta_star_ref", data=self.delta_star_ref)
            params_group.create_dataset(
                "gamma_star_moving", data=self.gamma_star_moving
            )
            params_group.create_dataset(
                "delta_star_moving", data=self.delta_star_moving
            )

            # Save configuration
            config_group.attrs["max_iter"] = self.max_iter
            config_group.attrs["tol"] = float(self.tol)
            config_group.attrs["n_samples_ref"] = self.n_samples_ref
            config_group.attrs["n_samples_moving"] = self.n_samples_moving
            config_group.attrs["n_locations"] = self.n_locations
            config_group.attrs["n_covariates"] = self.n_covariates

            # Save metadata
            config_group.attrs["source_site"] = self.source_site
            config_group.attrs["target_site"] = self.target_site
            metadata_group.attrs["version"] = "1.0"
            metadata_group.attrs["model_type"] = "PairwiseComBAT"
            metadata_group.attrs["creation_time"] = datetime.now().isoformat()
            metadata_group.attrs["fitted"] = True

        print(f"Model saved to {filepath}")
        print("  Model version: 1.0")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained PairwiseComBAT model from HDF5 file

        Args:
            filepath: Path to the saved model (.h5 extension will be added if not present)
        """
        # Ensure .h5 extension
        filepath = Path(filepath)
        if filepath.suffix != ".h5":
            filepath = filepath.with_suffix(".h5")

        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            # Validate model type
            model_type = (
                f["metadata"].attrs.get("model_type", "").decode()
                if isinstance(f["metadata"].attrs.get("model_type"), bytes)
                else f["metadata"].attrs.get("model_type", "")
            )
            if model_type != "PairwiseComBAT":
                raise ValueError(f"Invalid model type: {model_type}")

            # Load numerical parameters
            self.alpha_hat_ = f["parameters"]["alpha_hat"][:]
            self.beta_hat_ = f["parameters"]["beta_hat"][:]
            self.sigma_hat_ = f["parameters"]["sigma_hat"][:]
            self.gamma_star_ref = f["parameters"]["gamma_star_ref"][:]
            self.delta_star_ref = f["parameters"]["delta_star_ref"][:]
            self.gamma_star_moving = f["parameters"]["gamma_star_moving"][:]
            self.delta_star_moving = f["parameters"]["delta_star_moving"][:]

            # Load configuration
            self.source_site = f["config"].attrs.get("source_site", "source")
            self.target_site = f["config"].attrs.get("target_site", "target")
            if isinstance(self.source_site, bytes):
                self.source_site = self.source_site.decode()
            if isinstance(self.target_site, bytes):
                self.target_site = self.target_site.decode()

            self.max_iter = int(f["config"].attrs["max_iter"])
            self.tol = float(f["config"].attrs["tol"])
            self.n_samples_ref = int(f["config"].attrs["n_samples_ref"])
            self.n_samples_moving = int(f["config"].attrs["n_samples_moving"])
            self.n_locations = int(f["config"].attrs["n_locations"])
            self.n_covariates = int(f["config"].attrs["n_covariates"])

            # Mark as fitted
            self.is_fitted_ = True

            # Extract metadata for display
            version = f["metadata"].attrs.get("version", "unknown")
            creation_time = f["metadata"].attrs.get("creation_time", "unknown")

            if isinstance(version, bytes):
                version = version.decode()
            if isinstance(creation_time, bytes):
                creation_time = creation_time.decode()

        print(f"Model loaded from {filepath}")
        print(f"  Model version: {version}")
        print(f"  Created: {creation_time}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
