# Pairwise-ComBAT Implementation

## Overview
This implementation provides a complete Pairwise-ComBAT algorithm based on the paper "ComBAT Harmonization for diffusion MRI: Challenges and Best Practices". The implementation includes:

1. **Multi-feature Site utility** for generating simulated data with array-based gamma (γ) and delta (δ) batch effects
2. **PairwiseComBAT class** that estimates α, β, γ*, and δ* parameters using the mathematical formulation from the paper
3. **Harmonization transformation** that aligns non-reference sites to a reference site
4. **Comprehensive testing** with validation and quality metrics

## Key Features

- **Multi-Feature Support**: Handle multiple continuous covariates (age, cognitive scores, etc.) with feature-specific batch effects
- **Array-Based Parameters**: gamma and delta arrays match the number of continuous covariates for proper multi-dimensional harmonization
- **Explicit Parameter Validation**: No implicit conversions or broadcasting - clear dimension matching requirements
- **Bayesian Refinement**: Full implementation of iterative Bayesian site effect estimation
- **Comprehensive Testing**: 34+ test cases covering various scenarios and edge cases

## Key Components

### 1. Site Class
The `Site` class generates synthetic neuroimaging data with realistic batch effects:

```python
# Multi-feature data generation model (Equation 1 from paper):
# Y = α + β*X + σ*(δ*ε + γ*X)

class Site:
    def __init__(self, gamma, delta, alpha, beta, sigma):
        """
        Args:
            gamma: Array of additive site effects (n_features,)
            delta: Array of multiplicative site effects (n_features,)  
            alpha: Population intercept parameter
            beta: Array of population slope parameters (n_features,)
            sigma: Population noise standard deviation
        """
        
# Example: Two-feature site (age + IQ)
site = Site(
    gamma=[0.3, -0.2],  # Age effect +0.3, IQ effect -0.2
    delta=[1.2, 0.8],   # Age noise scaling 1.2x, IQ scaling 0.8x
    alpha=1.0,          # Population intercept
    beta=[0.05, 0.01],  # Age slope +0.05, IQ slope +0.01  
    sigma=0.5           # Overall noise scaling
)
```

### 2. PairwiseComBAT Class
Implements the complete harmonization algorithm with explicit parameter estimation:

#### Global Parameter Estimation (Equation 3)
```python
def fit(self, X_ref, Y_ref, X_moving, Y_moving):
    # OLS estimation: [α̂, β̂] = (X^T X)^{-1} X^T Y
    # Also computes σ̂ (residual standard deviation)
    self.alpha_hat_, self.beta_hat_, self.sigma_hat_ = self._estimate_global_parameters(X_combined, Y_combined)
```

#### Data Standardization (Equation 4)
```python
def _standardize_data(self, X, Y, alpha_hat, beta_hat, sigma_hat):
    # z_ijv = (y_ijv - α̂ - x_ij^T * β̂) / σ̂
    predicted = alpha_hat + X.dot(beta_hat)
    return (Y - predicted) / sigma_hat
```

#### Site Effect Estimation with Bayesian Refinement
```python
def _bayesian_site_estimation(self, z_site, mu_bar, tau_sq_bar, lambda_bar, theta_bar):
    # Iterative estimation of γ* and δ* using Equations 14-15
    # Converges to optimal site-specific parameters
```

#### Pairwise Harmonization (Equation 17)
```python
def predict(self, X_moving, Y_moving, gamma_ref=0.0, delta_ref=1.0):
    # Main harmonization equation:
    # Y_harmonized = σ̂ * (δ_ref/δ_moving * (z_moving - γ_moving) + γ_ref) + α̂ + X * β̂
```

## Mathematical Foundation

The implementation follows the paper's mathematical formulation with multi-feature support:

1. **Linear Model (Equation 1):**
   ```
   Y = α + X^T * β + σ * (δ ⊙ ε + γ ⊙ X)
   ```
   Where ⊙ denotes element-wise multiplication for multi-feature arrays.

2. **Global Parameter Estimation (Equation 3):**
   ```
   [α̂, β̂] = (X^T X)^{-1} X^T Y
   ```

3. **Data Standardization (Equation 4):**
   ```
   z_ijv = (y_ijv - α̂ - x_ij^T * β̂) / σ̂
   ```

4. **Bayesian Site Effects (Equations 14-15):**
   ```
   γ* = (J*τ²*γ_emp + δ²*μ) / (J*τ² + δ²)
   δ*² = (θ + 0.5*Σ(z - γ*)²) / (J/2 + λ - 1)
   ```

5. **Pairwise Harmonization (Equation 17):**
   ```
   Y_harmonized = σ̂ * (δ_ref/δ_moving * (z_moving - γ_moving) + γ_ref) + α̂ + X^T * β̂
   ```

## Usage Example

```python
import numpy as np
from pairwise_combat import Site, PairwiseComBAT

# 1. Create sites with different batch effects for multi-feature data
reference_site = Site(
    gamma=[0.0, 0.0],    # No additive effects (age, IQ)
    delta=[1.0, 1.0],    # No multiplicative effects  
    alpha=1.0,           # Population intercept
    beta=[0.05, 0.01],   # Age slope 0.05, IQ slope 0.01
    sigma=0.5            # Noise level
)

moving_site = Site(
    gamma=[0.3, -0.2],   # Age +0.3, IQ -0.2 additive effects
    delta=[1.2, 0.8],    # Age 1.2x, IQ 0.8x multiplicative effects
    alpha=1.0,           # Same population parameters
    beta=[0.05, 0.01],
    sigma=0.5
)

# 2. Generate multi-feature covariate data
n_samples = 150
X_ref = np.array([
    np.random.uniform(20, 80, n_samples),    # Age 20-80
    np.random.uniform(70, 140, n_samples)    # IQ 70-140
])
X_moving = np.array([
    np.random.uniform(25, 75, n_samples),
    np.random.uniform(75, 135, n_samples)
])

# 3. Generate site data with batch effects
Y_ref = reference_site.generate_site_data(X_ref)
Y_moving = moving_site.generate_site_data(X_moving)

# 4. Apply Pairwise-ComBAT harmonization
combat = PairwiseComBAT()
combat.fit(X_ref, Y_ref, X_moving, Y_moving)
Y_harmonized = combat.predict(X_moving, Y_moving)

# 5. Evaluate harmonization quality
print(f"Original data shape: {Y_moving.shape}")
print(f"Harmonized data shape: {Y_harmonized.shape}")
print("✅ Multi-feature harmonization complete!")
```

## Advanced Features

### Multi-Region Support
The implementation supports multi-region neuroimaging data (e.g., multiple brain regions or white matter bundles):

```python
# Generate data for multiple brain regions
n_regions = 5
Y_ref = reference_site.generate_site_data(X_ref, n_regions=n_regions)
Y_moving = moving_site.generate_site_data(X_moving, n_regions=n_regions)

# Harmonization works across all regions simultaneously
combat.fit(X_ref, Y_ref, X_moving, Y_moving)
Y_harmonized = combat.predict(X_moving, Y_moving)
print(f"Harmonized {n_regions} brain regions: {Y_harmonized.shape}")
```

### Dimension Validation
The library enforces strict dimension validation to prevent subtle bugs:

```python
# This will raise a clear error:
try:
    site = Site(gamma=[0.3, 0.5], delta=[1.2], alpha=1.0, beta=[0.04], sigma=0.5)
except ValueError as e:
    print(f"Dimension mismatch caught: {e}")
    # "gamma and delta must have the same shape"
```

### Site Effect Analysis
Extract and analyze site-specific effects:

```python
# Get estimated site effects after fitting
site_effects = combat.get_site_effects(X_moving, Y_moving)
print(f"Estimated gamma*: {site_effects['gamma_star']}")
print(f"Estimated delta*: {site_effects['delta_star']}")
```

## Installation & Requirements

This library requires Python 3.12+ and the following dependencies:

```toml
[dependencies]
h5py = ">=3.14.0"
matplotlib = ">=3.10.3"  
numpy = ">=2.3.2"
polars = ">=1.31.0"
pytest = ">=8.4.1"
scikit-learn = ">=1.7.1"
```

To install from source:
```bash
git clone <repository-url>
cd combat_pairwise
pip install -e .
```

Or with uv (recommended):
```bash
git clone <repository-url>
cd combat_pairwise
uv sync
```

## Testing

The implementation includes a comprehensive test suite with 34+ test cases:

```bash
# Run all tests
pytest test_pairwise_combat.py -v

# Run specific test categories
pytest test_pairwise_combat.py::TestDataGeneration -v       # Site initialization tests
pytest test_pairwise_combat.py::TestPairwiseComBAT -v       # Harmonization algorithm tests
pytest test_pairwise_combat.py::TestMultiFeature -v         # Multi-feature support tests
pytest test_pairwise_combat.py::TestIntegration -v          # Integration tests
pytest test_pairwise_combat.py::TestSaveLoad -v             # Model persistence tests
```

### Test Coverage

- **Site Initialization**: Parameter validation, array handling, backward compatibility
- **Data Generation**: Batch effect simulation, multi-feature support, dimension validation  
- **Harmonization Algorithm**: Parameter estimation, Bayesian refinement, convergence
- **Multi-Feature Support**: Array-based parameters, dimension matching, feature-specific effects
- **Integration Tests**: End-to-end harmonization workflows
- **Edge Cases**: Single samples, extreme parameters, dimension mismatches
- **Model Persistence**: Save/load functionality for trained models
- **Quality Validation**: MSE computation, site effect recovery, statistical properties

## Project Structure

```
combat_pairwise/
├── pairwise_combat.py          # Core implementation (Site, PairwiseComBAT)
├── test_pairwise_combat.py     # Comprehensive test suite (34+ tests)
├── main.py                     # Demonstration and comparison scripts
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Locked dependency versions (uv package manager)
├── README.md                   # This file
└── paper_tex/                  # LaTeX paper and figures
    ├── arxiv.tex               # Full mathematical derivation
    └── figures/                # Paper figures and illustrations
```

## Key Implementation Details

### Array-Based Architecture
- **gamma** and **delta** are arrays matching the number of continuous covariates
- **beta** coefficients are arrays for multi-feature slope effects
- No scalar limitations - full support for complex multi-dimensional harmonization

### Explicit Validation
- All parameters must have matching dimensions (no implicit broadcasting)
- Clear error messages for dimension mismatches
- Robust handling of edge cases (zero variance, single samples)

### Mathematical Accuracy
- Follows exact equations from the ComBAT paper
- Proper Bayesian hyperparameter estimation (Equations 11-13)  
- Iterative convergence for site effect refinement
- Numerically stable implementations

### Performance Considerations
- Efficient vectorized operations using NumPy
- Memory-efficient handling of multi-region data
- Convergence monitoring to prevent infinite loops

## Comparison with Original ComBAT

| Feature | Original ComBAT | This Implementation |
|---------|-----------------|-------------------|
| Multi-site support | ✅ Full | ⚡ Pairwise (2 sites) |
| Multi-feature covariates | ⚠️ Limited | ✅ Full array support |
| Bayesian refinement | ✅ Yes | ✅ Yes |
| Dimension validation | ⚠️ Implicit | ✅ Explicit |
| Test coverage | ⚠️ Minimal | ✅ Comprehensive (34+ tests) |
| Multi-region support | ✅ Yes | ✅ Yes |
| Parameter flexibility | ⚠️ Global constants | ✅ Explicit parameters |

## Contributing

This implementation provides a solid foundation for neuroimaging harmonization research. Contributions are welcome for:

- Additional validation metrics
- Performance optimizations  
- Extended multi-site support beyond pairwise
- Integration with neuroimaging data formats
- Visualization enhancements

## References

- Johnson, W.E., Li, C., Rabinovic, A.: Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics (2007)
- Fortin, J.P., et al.: Harmonization of cortical thickness measurements across scanners and sites. NeuroImage (2018)
- The paper "ComBAT Harmonization for diffusion MRI: Challenges and Best Practices" that this implementation is based on

This implementation enables robust, mathematically accurate harmonization of neuroimaging data with comprehensive validation and multi-feature support.
