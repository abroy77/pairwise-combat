import numpy as np
from numpy.typing import ArrayLike


class FeatureProperties:
    num_locations: int
    num_cont_covars: int
    alphas: ArrayLike
    betas: ArrayLike
    sigmas: ArrayLike

    def __init__(
        self,
        alphas: ArrayLike,
        betas: ArrayLike,
        sigmas: ArrayLike,
    ):
        # check if scalar
        if isinstance(alphas, (int, float)):
            alphas = np.asarray([alphas])
        if isinstance(betas, (int, float)):
            betas = np.asarray([betas])
        if isinstance(sigmas, (int, float)):
            sigmas = np.asarray([sigmas])

        alphas = np.asarray(alphas)
        betas = np.asarray(betas)
        sigmas = np.asarray(sigmas)

        # alphas must be a 1D array on entry. then we convert to 2D column vector
        if alphas.ndim != 1:
            raise ValueError(f"alphas must be 1D array, got {alphas.ndim}D")

        alphas = alphas.reshape(-1, 1)

        num_locations = alphas.shape[0]
        # betas validation
        if betas.ndim == 1:
            betas = betas.reshape(-1, 1)
        elif betas.ndim == 2:
            pass
        else:
            raise ValueError(f"betas must be 1D or 2D array, got {betas.ndim}D")

        if betas.shape[0] != num_locations:
            raise ValueError(
                f"betas must have the same number of rows as alphas, got {betas.shape[0]} rows for betas and {num_locations} for alphas"
            )

        num_cont_covars = betas.shape[1]
        # sigmas must also be 1D on entry, then we convert to 2D column vector
        if sigmas.ndim != 1:
            raise ValueError(f"sigmas must be 1D array, got {sigmas.ndim}D")

        sigmas = sigmas.reshape(-1, 1)
        if sigmas.shape[0] != num_locations:
            raise ValueError(
                f"sigmas must have the same number of rows as alphas, got {sigmas.shape[0]} rows for sigmas and {num_locations} for alphas"
            )
        self.alphas = alphas
        self.betas = betas
        self.sigmas = sigmas
        self.num_locations = num_locations
        self.num_cont_covars = num_cont_covars


class Site:
    """Represents a data collection site with specific batch effects"""

    gamma: ArrayLike  # Additive site effect shape: (num_locations, 1)
    delta: ArrayLike  # Multiplicative site effect shape: (num_locations, 1)
    feature_properties: FeatureProperties

    def __init__(
        self,
        gamma: ArrayLike,
        delta: ArrayLike,
        feature_properties: FeatureProperties,
    ):
        """
        Initialize site with batch effects and population parameters

        Args:
            gamma: Additive site effect for each location (region)
                   Shape: (num_locations,) or (num_locations, 1)
            delta: Multiplicative site effect for each location (region)
                   Shape: (num_locations,) or (num_locations, 1)
            feature_properties: FeatureProperties object containing site-specific parameters
        """
        if isinstance(gamma, (int, float)):
            gamma = np.asarray([gamma])
        if isinstance(delta, (int, float)):
            delta = np.asarray([delta])

        self.gamma = np.asarray(gamma)
        self.delta = np.asarray(delta)

        # Accept (num_locations,) or (num_locations, 1) for gamma/delta
        if self.gamma.ndim == 1:
            self.gamma = self.gamma.reshape(-1, 1)
        elif self.gamma.ndim == 2 and self.gamma.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"gamma must be 1D or 2D column vector of shape (num_locations, 1), got {self.gamma.shape}"
            )

        if self.delta.ndim == 1:
            self.delta = self.delta.reshape(-1, 1)
        elif self.delta.ndim == 2 and self.delta.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"delta must be 1D or 2D column vector of shape (num_locations, 1), got {self.delta.shape}"
            )

        assert isinstance(
            feature_properties, FeatureProperties
        ), "feature_properties must be an instance of FeatureProperties"
        self.feature_properties = feature_properties

        # Validate that gamma/delta match the number of locations
        if self.gamma.shape[0] != self.feature_properties.num_locations:
            raise ValueError(
                f"gamma must have shape ({self.feature_properties.num_locations}, 1), got {self.gamma.shape}"
            )
        if self.delta.shape[0] != self.feature_properties.num_locations:
            raise ValueError(
                f"delta must have shape ({self.feature_properties.num_locations}, 1), got {self.delta.shape}"
            )
        if self.gamma.shape != self.delta.shape:
            raise ValueError(
                f"gamma and delta must have the same shape. Got gamma: {self.gamma.shape}, delta: {self.delta.shape}"
            )

    def generate_site_data(
        self, baseline_data: ArrayLike, cont_covars: ArrayLike
    ) -> ArrayLike:
        """
        Apply site-specific additive and multiplicative effects to baseline data.

        Args:
            baseline_data: Array of shape (num_locations, n_samples)
            cont_covars: Array of shape (num_cont_covars, n_samples)

        Returns:
            Array of shape (num_locations, n_samples) with site effects applied.
        """
        baseline_data = np.asarray(baseline_data)
        cont_covars = np.asarray(cont_covars)

        # Validate baseline_data shape: (num_locations, n_samples)
        if baseline_data.ndim != 2:
            raise ValueError(
                f"baseline_data must be a 2D array of shape (num_locations, n_samples). Got {baseline_data.shape}"
            )
        if baseline_data.shape[0] != self.feature_properties.num_locations:
            raise ValueError(
                f"baseline_data must have {self.feature_properties.num_locations} rows (num_locations). Got {baseline_data.shape[0]}"
            )

        n_samples = baseline_data.shape[1]
        num_cont_covars = self.feature_properties.num_cont_covars

        # cont_covars must be (num_cont_covars, n_samples)
        if cont_covars.shape != (num_cont_covars, n_samples):
            raise ValueError(
                f"cont_covars must have shape (num_cont_covars, n_samples). Got {cont_covars.shape}"
            )

        # Shapes:
        #   alphas: (num_locations, 1)
        #   betas: (num_locations, num_cont_covars)
        #   sigmas: (num_locations, 1)
        #   delta, gamma: (num_locations, 1)
        #   baseline_data: (num_locations, n_samples)
        #   cont_covars: (num_cont_covars, n_samples)

        # Site effects: delta * baseline + gamma, then scale by sigma
        site_effect = (
            self.delta * baseline_data + self.gamma
        )  # (num_locations, n_samples)
        scaled = (
            self.feature_properties.sigmas * site_effect
        )  # (num_locations, n_samples)

        # Population-level slope and intercept
        data = (
            self.feature_properties.alphas
            + np.dot(self.feature_properties.betas, cont_covars)
            + scaled
        )  # (num_locations, n_samples)

        return data
