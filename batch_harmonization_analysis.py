"""
Batch harmonization analysis script for PairwiseComBAT with Polars DataFrames.

This script trains multiple PairwiseComBAT models for different site pairs and
brain measures, then generates before/after comparison plots to assess
harmonization effectiveness.
"""

import polars as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from datetime import datetime

from pairwise_combat_polars import PairwiseComBATDataFrame


class BatchHarmonizationAnalysis:
    """
    Class for batch harmonization analysis with visualization.
    """
    
    def __init__(
        self,
        ref_site: str,
        source_sites: List[str],
        measures_dict: Dict[str, List[str]],
        covariate_cols: List[str] = ["age", "sex"],
        site_id_col: str = "site_id",
        combat_params: Optional[Dict] = None
    ):
        """
        Initialize batch harmonization analysis.
        
        Args:
            ref_site: Name of reference site for harmonization
            source_sites: List of source site names to harmonize to reference
            measures_dict: Dictionary mapping measure names to column lists
                          e.g., {"cortical_thickness": ["lh_thickness_1", "rh_thickness_1"],
                                 "volume": ["lh_volume_1", "rh_volume_1"]}
            covariate_cols: List of covariate column names (default: ["age", "sex"])
            site_id_col: Name of site identifier column (default: "site_id")
            combat_params: Optional parameters for PairwiseComBAT (max_iter, tol)
        """
        self.ref_site = ref_site
        self.source_sites = source_sites
        self.measures_dict = measures_dict
        self.covariate_cols = covariate_cols
        self.site_id_col = site_id_col
        
        # Set default ComBAT parameters
        self.combat_params = combat_params or {"max_iter": 30, "tol": 1e-6}
        
        # Results storage
        self.trained_models = {}
        self.harmonization_results = {}
        
        # Plotting parameters
        self.figure_size = (15, 10)
        self.font_size = 12
        plt.rcParams.update({'font.size': self.font_size})
        
    def validate_data(self, df: pl.DataFrame) -> None:
        """
        Validate that the DataFrame contains required columns and sites.
        
        Args:
            df: Input DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check site column exists
        if self.site_id_col not in df.columns:
            raise ValueError(f"Site column '{self.site_id_col}' not found in DataFrame")
        
        # Check sites exist
        available_sites = df[self.site_id_col].unique().to_list()
        
        if self.ref_site not in available_sites:
            raise ValueError(f"Reference site '{self.ref_site}' not found. Available: {available_sites}")
        
        missing_source_sites = [site for site in self.source_sites if site not in available_sites]
        if missing_source_sites:
            raise ValueError(f"Source sites not found: {missing_source_sites}. Available: {available_sites}")
        
        # Check covariate columns exist
        missing_covars = [col for col in self.covariate_cols if col not in df.columns]
        if missing_covars:
            raise ValueError(f"Covariate columns not found: {missing_covars}")
        
        # Check measure columns exist
        all_measure_cols = []
        for measure, cols in self.measures_dict.items():
            missing_cols = [col for col in cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Measure '{measure}' columns not found: {missing_cols}")
            all_measure_cols.extend(cols)
        
        print("✓ Data validation passed:")
        print(f"  - Sites: {len(available_sites)} available ({self.ref_site} as reference)")
        print(f"  - Source sites to harmonize: {len(self.source_sites)}")
        print(f"  - Measures: {len(self.measures_dict)} ({len(all_measure_cols)} total columns)")
        print(f"  - Covariates: {len(self.covariate_cols)}")
        
    def create_age_sex_plots(
        self,
        df_pre: pl.DataFrame,
        df_post: pl.DataFrame,
        source_site: str,
        measure_name: str,
        measure_cols: List[str]
    ) -> plt.Figure:
        """
        Create 2x2 plot showing before/after harmonization by sex.
        
        Args:
            df_pre: DataFrame before harmonization
            df_post: DataFrame after harmonization  
            source_site: Name of source site being harmonized
            measure_name: Name of the brain measure
            measure_cols: List of column names for this measure
            
        Returns:
            Single figure with 2x2 subplot layout
        """
        # Calculate mean across measure columns for plotting
        df_pre_plot = df_pre.with_columns(
            pl.concat_list(measure_cols).list.mean().alias("measure_mean")
        )
        df_post_plot = df_post.with_columns(
            pl.concat_list(measure_cols).list.mean().alias("measure_mean")
        )
        
        # Create single figure with 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
        
        # Filter data for relevant sites
        ref_pre = df_pre_plot.filter(pl.col(self.site_id_col) == self.ref_site)
        source_pre = df_pre_plot.filter(pl.col(self.site_id_col) == source_site)
        ref_post = df_post_plot.filter(pl.col(self.site_id_col) == self.ref_site)
        source_post = df_post_plot.filter(pl.col(self.site_id_col) == source_site)
        
        # Set main title
        fig.suptitle(f'{measure_name}: {source_site} → {self.ref_site} Harmonization', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Top left: Pre-Combat Males (sex=1)
        self._plot_age_relationship(axes[0, 0], ref_pre, source_pre, 
                                  "Pre-Combat: Males (sex=1)", sex_filter=1)
        
        # Top right: Post-Combat Males (sex=1)  
        self._plot_age_relationship(axes[0, 1], ref_post, source_post,
                                  "Post-Combat: Males (sex=1)", sex_filter=1)
        
        # Bottom left: Pre-Combat Females (sex=0)
        self._plot_age_relationship(axes[1, 0], ref_pre, source_pre,
                                  "Pre-Combat: Females (sex=0)", sex_filter=0)
        
        # Bottom right: Post-Combat Females (sex=0)
        self._plot_age_relationship(axes[1, 1], ref_post, source_post,
                                  "Post-Combat: Females (sex=0)", sex_filter=0)
        
        plt.tight_layout()
        return fig
    
    def _plot_age_relationship(
        self,
        ax: plt.Axes,
        ref_data: pl.DataFrame,
        source_data: pl.DataFrame,
        title: str,
        sex_filter: Optional[int] = None
    ) -> None:
        """Plot age vs measure relationship."""
        if sex_filter is not None:
            ref_data = ref_data.filter(pl.col("sex") == sex_filter)
            source_data = source_data.filter(pl.col("sex") == sex_filter)
        
        if ref_data.height > 0:
            ax.scatter(ref_data["age"], ref_data["measure_mean"], 
                      alpha=0.6, label=self.ref_site, color='blue', s=30)
        if source_data.height > 0:
            ax.scatter(source_data["age"], source_data["measure_mean"], 
                      alpha=0.6, label=source_data[self.site_id_col][0], color='red', s=30)
        
        ax.set_xlabel("Age")
        ax.set_ylabel("Mean Measure Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def train_single_model(
        self,
        df: pl.DataFrame,
        source_site: str,
        measure_name: str,
        measure_cols: List[str]
    ) -> Tuple[PairwiseComBATDataFrame, pl.DataFrame]:
        """
        Train a single PairwiseComBAT model and return harmonized data.
        
        Args:
            df: Input DataFrame
            source_site: Source site name
            measure_name: Name of brain measure
            measure_cols: List of measure column names
            
        Returns:
            Tuple of (trained_model, harmonized_dataframe)
        """
        print(f"  Training {source_site} -> {self.ref_site} for {measure_name}")
        print(f"    Columns: {len(measure_cols)} features")
        
        # Filter to only relevant sites
        site_data = df.filter(
            pl.col(self.site_id_col).is_in([source_site, self.ref_site])
        )
        
        # Initialize and train model
        combat_model = PairwiseComBATDataFrame(
            source_site=source_site,
            target_site=self.ref_site,
            **self.combat_params
        )
        
        # Fit and transform
        harmonized_df = combat_model.fit_transform(
            df=site_data,
            site_id_col=self.site_id_col,
            data_cols=measure_cols,
            covariate_cols=self.covariate_cols
        )
        
        print("    ✓ Model trained successfully")
        
        return combat_model, harmonized_df
    
    def run_batch_analysis(
        self,
        df: pl.DataFrame,
        output_pdf: str = "harmonization_analysis.pdf"
    ) -> None:
        """
        Run complete batch harmonization analysis with visualization.
        
        Args:
            df: Input DataFrame containing all sites and measures
            output_pdf: Path for output PDF file
        """
        print("🚀 Starting batch harmonization analysis...")
        print("📊 Configuration:")
        print(f"   Reference site: {self.ref_site}")
        print(f"   Source sites: {self.source_sites}")
        print(f"   Measures: {list(self.measures_dict.keys())}")
        print(f"   Output PDF: {output_pdf}")
        print()
        
        # Validate input data
        self.validate_data(df)
        print()
        
        # Create output directory if needed
        output_path = Path(output_pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDF for plots
        with PdfPages(output_pdf) as pdf:
            total_models = len(self.source_sites) * len(self.measures_dict)
            current_model = 0
            
            # Loop through source sites
            for source_site in self.source_sites:
                print(f"🔄 Processing source site: {source_site}")
                
                # Loop through measures
                for measure_name, measure_cols in self.measures_dict.items():
                    current_model += 1
                    print(f"📈 Model {current_model}/{total_models}: {measure_name}")
                    
                    try:
                        # Train model and get harmonized data
                        model, harmonized_df = self.train_single_model(
                            df, source_site, measure_name, measure_cols
                        )
                        
                        # Store results
                        model_key = f"{source_site}_{measure_name}"
                        self.trained_models[model_key] = model
                        self.harmonization_results[model_key] = harmonized_df
                        
                        # Create before/after plots
                        original_data = df.filter(
                            pl.col(self.site_id_col).is_in([source_site, self.ref_site])
                        )
                        
                        fig = self.create_age_sex_plots(
                            original_data, harmonized_df,
                            source_site, measure_name, measure_cols
                        )
                        
                        # Save plot to PDF
                        pdf.savefig(fig, bbox_inches='tight')
                        
                        # Close figure to save memory
                        plt.close(fig)
                        
                        print("    ✓ Plots saved to PDF")
                        
                    except Exception as e:
                        print(f"    ❌ Error training model: {str(e)}")
                        warnings.warn(f"Failed to process {source_site} - {measure_name}: {str(e)}")
                        continue
                
                print()
            
            # Add summary page
            self._create_summary_page(pdf, df)
        
        print(f"✅ Analysis complete! Results saved to: {output_pdf}")
        print(f"📊 Successfully trained {len(self.trained_models)} models")
        
    def _create_summary_page(self, pdf: PdfPages, df: pl.DataFrame) -> None:
        """Create a summary page with analysis overview."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.axis('off')
        
        # Analysis summary text
        summary_text = f"""
        PAIRWISE COMBAT HARMONIZATION ANALYSIS SUMMARY
        
        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Configuration:
        • Reference Site: {self.ref_site}
        • Source Sites: {', '.join(self.source_sites)} ({len(self.source_sites)} sites)
        • Brain Measures: {', '.join(self.measures_dict.keys())} ({len(self.measures_dict)} measures)
        • Covariates: {', '.join(self.covariate_cols)}
        
        Data Overview:
        • Total Subjects: {df.height:,}
        • Sites in Data: {len(df[self.site_id_col].unique())}
        • Age Range: {df['age'].min():.1f} - {df['age'].max():.1f} years
        • Sex Distribution: {df.filter(pl.col('sex')==0).height} F, {df.filter(pl.col('sex')==1).height} M
        
        Results:
        • Models Trained: {len(self.trained_models)} / {len(self.source_sites) * len(self.measures_dict)}
        • ComBAT Parameters: max_iter={self.combat_params['max_iter']}, tol={self.combat_params['tol']}
        
        Plot Organization:
        Each model generates 1 page with 2x2 layout:
        • Top-left: Pre-Combat Males (age vs measure)
        • Top-right: Post-Combat Males (age vs measure)  
        • Bottom-left: Pre-Combat Females (age vs measure)
        • Bottom-right: Post-Combat Females (age vs measure)
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        fig.suptitle('Harmonization Analysis Summary', fontsize=16, fontweight='bold')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def save_models(self, output_dir: str = "trained_models") -> None:
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving {len(self.trained_models)} trained models to {output_dir}/")
        
        for model_key, model in self.trained_models.items():
            model_file = output_path / f"{model_key}_combat_model.h5"
            model.save_model(str(model_file))
            print(f"  ✓ Saved: {model_file.name}")
        
        print("✅ All models saved successfully!")


def example_usage():
    """
    Example of how to use the BatchHarmonizationAnalysis class.
    """
    # Example configuration
    ref_site = "site_A"
    source_sites = ["site_B", "site_C", "site_D"]
    
    # Example measures dictionary - YOU NEED TO FILL THIS IN
    measures_dict = {
        "cortical_thickness": [
            # Add your cortical thickness column names here
            "lh_bankssts_thickness", "lh_caudalanteriorcingulate_thickness",
            "rh_bankssts_thickness", "rh_caudalanteriorcingulate_thickness"
        ],
        "surface_area": [
            # Add your surface area column names here
            "lh_bankssts_area", "lh_caudalanteriorcingulate_area",
            "rh_bankssts_area", "rh_caudalanteriorcingulate_area"
        ],
        "volume": [
            # Add your volume column names here
            "Left_Hippocampus", "Right_Hippocampus",
            "Left_Amygdala", "Right_Amygdala"
        ]
    }
    
    # Load your data (replace with your actual data loading)
    # df = pl.read_csv("your_data.csv")
    
    # For demonstration, create example data
    from pairwise_combat_polars import create_example_dataframe
    
    print("📝 Creating example dataset...")
    
    # Count total columns needed
    all_measure_cols = []
    for cols in measures_dict.values():
        all_measure_cols.extend(cols)
    
    df_list = []
    for site in [ref_site] + source_sites:
        site_df = create_example_dataframe(
            n_samples_per_site=100,
            n_locations=len(all_measure_cols),  # Create enough columns
            source_site=site,
            target_site="dummy",
            random_seed=42
        ).filter(pl.col("site_id") == site)
        df_list.append(site_df)
    
    df = pl.concat(df_list)
    
    # Rename voxel columns to match measures_dict
    voxel_cols = [f"voxel_{i+1}" for i in range(len(all_measure_cols))]
    rename_dict = {old: new for old, new in zip(voxel_cols, all_measure_cols)}
    df = df.rename(rename_dict)
    
    print(f"✓ Created example dataset with {df.height} subjects across {len([ref_site] + source_sites)} sites")
    
    # Initialize and run analysis
    analysis = BatchHarmonizationAnalysis(
        ref_site=ref_site,
        source_sites=source_sites,
        measures_dict=measures_dict,
        covariate_cols=["age", "sex"],
        combat_params={"max_iter": 20, "tol": 1e-5}  # Faster for demo
    )
    
    # Run the complete analysis
    analysis.run_batch_analysis(
        df=df,
        output_pdf="example_harmonization_analysis.pdf"
    )
    
    # Optionally save models
    analysis.save_models("example_trained_models")


if __name__ == "__main__":
    print("🧠 PairwiseComBAT Batch Harmonization Analysis")
    print("=" * 50)
    
    # Run example
    example_usage()
    
    print("\n" + "=" * 50)
    print("📚 To use with your own data:")
    print("1. Fill in the measures_dict with your actual column names")
    print("2. Load your DataFrame with df = pl.read_csv('your_data.csv')")
    print("3. Update ref_site and source_sites with your site names")
    print("4. Run analysis.run_batch_analysis(df, 'output.pdf')")
