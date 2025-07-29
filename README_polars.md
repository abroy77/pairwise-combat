# PairwiseComBAT Polars DataFrame Interface

This extension provides a user-friendly Polars DataFrame interface for the PairwiseComBAT harmonization algorithm, making it easier to work with tabular neuroimaging data.

## Features

- **DataFrame Integration**: Work directly with Polars DataFrames instead of NumPy arrays
- **Automatic Validation**: Built-in validation for data types, column names, and site requirements
- **Flexible Column Specification**: Specify data columns (brain regions/voxels) and covariates by name
- **Multi-Site Support**: Handle data from multiple sites with automatic filtering and processing
- **Save/Load Models**: Persist trained models for later use
- **Comprehensive Testing**: Full test suite with realistic synthetic data

## Installation

```bash
pip install polars  # Install polars if not already available
```

## Quick Start

```python
import polars as pl
from pairwise_combat_polars import PairwiseComBATDataFrame, create_example_dataframe

# Create example data
df = create_example_dataframe(
    n_samples_per_site=50,
    n_locations=5,
    source_site="Hospital_A",
    target_site="Hospital_B"
)

# Initialize harmonizer
combat = PairwiseComBATDataFrame(
    source_site="Hospital_A",
    target_site="Hospital_B"
)

# Fit and transform in one step
harmonized_df = combat.fit_transform(
    df=df,
    site_id_col="site_id",
    data_cols=["voxel_1", "voxel_2", "voxel_3", "voxel_4", "voxel_5"],
    covariate_cols=["age", "sex"]
)
```

## Data Requirements

### DataFrame Structure
Your DataFrame must contain:
- **Site ID column**: String column identifying which site each sample comes from
- **Data columns**: Numeric columns representing brain regions, voxels, or features to harmonize
- **Covariate columns**: Numeric columns for confounding variables (age, sex, etc.)

### Example DataFrame:
```
┌──────────┬─────────┬─────┬─────┬─────────┬─────────┬─────────┐
│ site_id  ┆ age     ┆ sex ┆ ... ┆ voxel_1 ┆ voxel_2 ┆ voxel_3 │
│ ---      ┆ ---     ┆ --- ┆     ┆ ---     ┆ ---     ┆ ---     │
│ str      ┆ f64     ┆ f64 ┆     ┆ f64     ┆ f64     ┆ f64     │
╞══════════╪═════════╪═════╪═════╪═════════╪═════════╪═════════╡
│ site_A   ┆ 25.3    ┆ 1.0 ┆ ... ┆ 4.2     ┆ 3.8     ┆ 5.1     │
│ site_B   ┆ 30.7    ┆ 0.0 ┆ ... ┆ 4.8     ┆ 4.1     ┆ 5.3     │
│ site_A   ┆ 45.1    ┆ 1.0 ┆ ... ┆ 3.9     ┆ 3.5     ┆ 4.9     │
└──────────┴─────────┴─────┴─────┴─────────┴─────────┴─────────┘
```

## API Reference

### PairwiseComBATDataFrame

#### Initialization
```python
combat = PairwiseComBATDataFrame(
    source_site="site_A",      # Source site identifier
    target_site="site_B",      # Target/reference site identifier  
    max_iter=30,               # Maximum iterations for Bayesian estimation
    tol=1e-6                   # Convergence tolerance
)
```

#### Methods

##### `fit(df, site_id_col, data_cols, covariate_cols)`
Train the harmonization model.

**Parameters:**
- `df`: Polars DataFrame containing training data from both sites
- `site_id_col`: Name of column containing site identifiers
- `data_cols`: List of column names containing response data (brain regions/voxels)
- `covariate_cols`: List of column names containing covariates

**Returns:** Self (for method chaining)

##### `transform(df, site_id_col=None, data_cols=None, covariate_cols=None)`
Harmonize source site data to target site characteristics.

**Parameters:**
- `df`: DataFrame containing data to harmonize
- `site_id_col`: Site ID column name (uses fitted value if None)
- `data_cols`: Data column names (uses fitted values if None)  
- `covariate_cols`: Covariate column names (uses fitted values if None)

**Returns:** DataFrame with harmonized data

##### `fit_transform(df, site_id_col, data_cols, covariate_cols)`
Fit model and transform data in one step.

**Parameters:** Same as `fit()`
**Returns:** DataFrame with harmonized data

##### `save_model(filepath)` / `load_model(filepath)`
Save/load trained model to/from HDF5 file.

##### `get_model_info()`
Get information about the fitted model.

**Returns:** Dictionary with model parameters and configuration

## Examples

### Basic Usage
```python
import polars as pl
from pairwise_combat_polars import PairwiseComBATDataFrame

# Load your data
df = pl.read_csv("neuroimaging_data.csv")

# Initialize harmonizer
combat = PairwiseComBATDataFrame(
    source_site="scanner_A",
    target_site="scanner_B",
    max_iter=20
)

# Fit the model
combat.fit(
    df=df,
    site_id_col="scanner_site",
    data_cols=["region_1", "region_2", "region_3"],
    covariate_cols=["age", "sex", "education"]
)

# Harmonize new test data
test_df = pl.read_csv("test_data.csv")
harmonized = combat.transform(test_df)
```

### Working with Subsets
```python
# Use only specific brain regions
brain_regions = ["frontal_cortex", "temporal_lobe", "hippocampus"]
covariates = ["age", "sex"]

harmonized = combat.fit_transform(
    df=full_dataset,
    site_id_col="site",
    data_cols=brain_regions,
    covariate_cols=covariates
)
```

### Model Persistence
```python
# Train and save model
combat.fit(df, "site", brain_regions, covariates)
combat.save_model("my_harmonization_model.h5")

# Later: load and use saved model
new_combat = PairwiseComBATDataFrame(
    source_site="scanner_A",
    target_site="scanner_B"
)
new_combat.load_model("my_harmonization_model.h5")
harmonized = new_combat.transform(new_data)
```

### Analyzing Harmonization Results
```python
# Compare site differences before and after
original_means = df.group_by("site").agg([
    pl.col("voxel_1").mean().alias("voxel_1_mean")
])

harmonized_means = harmonized_df.group_by("site").agg([
    pl.col("voxel_1").mean().alias("voxel_1_mean")
])

print("Before harmonization:")
print(original_means)
print("\\nAfter harmonization:")
print(harmonized_means)
```

## Validation and Error Handling

The DataFrame interface includes comprehensive validation:

- **Missing columns**: Checks that all specified columns exist
- **Site validation**: Ensures both source and target sites are present in data
- **Data types**: Validates that data and covariate columns are numeric
- **Shape consistency**: Ensures proper array shapes for the underlying algorithm

Common error messages:
- `"Missing columns in DataFrame: ['column_name']"`
- `"Source site 'site_A' not found in site_id column"`
- `"Non-numeric columns found: ['column_name (dtype)']"`
- `"No data found for source site 'site_A'"`

## Performance Considerations

- **Memory usage**: DataFrames are converted to NumPy arrays internally
- **Data size**: Performance scales with number of samples and brain regions
- **Iteration count**: Higher `max_iter` values improve convergence but increase runtime
- **Polars efficiency**: Leverages Polars' fast operations for data manipulation

## Testing

The module includes comprehensive tests covering:
- Data validation and error handling
- Model fitting and transformation
- Save/load functionality  
- Edge cases and boundary conditions
- Harmonization effectiveness

Run tests with:
```bash
pytest test_pairwise_combat_polars.py -v
```

## Comparison with NumPy Interface

| Feature | DataFrame Interface | NumPy Interface |
|---------|-------------------|-----------------|
| Data input | Polars DataFrame | NumPy arrays |
| Column handling | By name | By position |
| Validation | Automatic | Manual |
| Site filtering | Automatic | Manual |
| Multi-site data | Built-in support | Manual preparation |
| User-friendliness | High | Medium |
| Performance | Slight overhead | Fastest |

## Integration with Existing Workflows

The DataFrame interface is designed to integrate seamlessly with data science workflows:

```python
# Typical neuroimaging workflow
df = (
    pl.read_csv("raw_data.csv")
    .filter(pl.col("quality_check") == "pass")
    .with_columns([
        pl.col("age").cast(pl.Float64),
        pl.col("sex").map_elements(lambda x: 1.0 if x == "M" else 0.0)
    ])
)

# Harmonize data
harmonized = combat.fit_transform(
    df=df,
    site_id_col="scanner_id", 
    data_cols=[col for col in df.columns if col.startswith("voxel_")],
    covariate_cols=["age", "sex", "education_years"]
)

# Continue with analysis
results = harmonized.group_by("diagnosis").agg([
    pl.col("voxel_*").mean()
])
```
