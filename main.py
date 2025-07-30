import numpy as np
from matplotlib import pyplot as plt
from src.pairwise_combat.core import PairwiseComBAT, Site

MIN_AGE = 40
MAX_AGE = 95
N_SAMPLES_PER_SITE = 100

ALPHA = 0.5
BETA = 0.04
SIGMA = 0.5


def demo_comparison():
    """Compare original vs harmonized data with different site effects"""
    print("Comparing Original vs Pairwise-ComBAT Harmonization")
    print("=" * 55)
    
    # Create sites with different levels of batch effects
    sites = [
        ("Reference", Site(gamma=0.0, delta=1.0)),
        ("Mild Effects", Site(gamma=0.3, delta=1.2)), 
        ("Strong Effects", Site(gamma=0.8, delta=1.6)),
        ("Extreme Effects", Site(gamma=1.2, delta=2.0))
    ]
    
    rng = np.random.default_rng(42)
    n_samples = 120
    
    # Generate data for all sites
    site_data = {}
    for name, site in sites:
        ages = rng.integers(low=MIN_AGE, high=MAX_AGE, size=n_samples)
        noise = rng.normal(loc=0, scale=SIGMA, size=n_samples)
        data = site.generate_site_data(ages, noise)
        site_data[name] = (ages, data, site)
    
    # Use reference site for harmonization
    ref_ages, ref_data, _ = site_data["Reference"]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    combat = PairwiseComBAT()
    gof_results = []
    
    for i, (name, (ages, data, site)) in enumerate(site_data.items()):
        if name == "Reference":
            # Reference site - no harmonization needed
            harmonized_data = data
            gof_before = gof_after = 0.0
        else:
            # Harmonize to reference
            harmonized_data = combat.harmonize_to_reference(
                X_ref=ref_ages, Y_ref=ref_data,
                X_moving=ages, Y_moving=data
            )
            gof_before = combat.compute_goodness_of_fit(ref_ages, ref_data, ages, data)
            gof_after = combat.compute_goodness_of_fit(ref_ages, ref_data, ages, harmonized_data)
        
        gof_results.append((name, gof_before, gof_after))
        
        # Plot original vs harmonized
        axes[i].scatter(ref_ages, ref_data, alpha=0.5, label='Reference', color='blue', s=20)
        axes[i].scatter(ages, data, alpha=0.5, label=f'{name} (Original)', color='red', s=20)
        axes[i].scatter(ages, harmonized_data, alpha=0.7, label=f'{name} (Harmonized)', color='green', s=20)
        
        # Add true canonical line
        age_range = np.linspace(MIN_AGE, MAX_AGE, 100)
        canonical_line = ALPHA + BETA * age_range
        axes[i].plot(age_range, canonical_line, 'k--', alpha=0.8, label='True Canonical', linewidth=2)
        
        axes[i].set_xlabel('Age')
        axes[i].set_ylabel('MRI Metric')
        axes[i].set_title(f'{name}\nγ={site.gamma}, δ={site.delta}')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print goodness-of-fit results
    print("\nGoodness-of-Fit Results:")
    print("-" * 40)
    for name, gof_before, gof_after in gof_results:
        if name == "Reference":
            print(f"{name:15}: N/A (reference site)")
        else:
            improvement = ((gof_before - gof_after) / gof_before * 100) if gof_before > 0 else 0
            print(f"{name:15}: {gof_before:.4f} → {gof_after:.4f} ({improvement:+.1f}%)")


def test_different_age_ranges():
    """Test harmonization with different age distributions"""
    print("\nTesting Different Age Range Distributions")
    print("=" * 45)
    
    # Reference site with full age range
    ref_site = Site(gamma=0.0, delta=1.0)
    moving_site = Site(gamma=0.6, delta=1.4)
    
    rng = np.random.default_rng(123)
    
    # Different age distributions
    age_scenarios = [
        ("Full Range", (MIN_AGE, MAX_AGE)),
        ("Young Only", (MIN_AGE, MIN_AGE + 20)),
        ("Old Only", (MAX_AGE - 20, MAX_AGE)),
        ("Middle Age", (MIN_AGE + 15, MAX_AGE - 15))
    ]
    
    # Reference data (always full range)
    ref_ages = rng.integers(low=MIN_AGE, high=MAX_AGE, size=150)
    ref_noise = rng.normal(loc=0, scale=SIGMA, size=150)
    ref_data = ref_site.generate_site_data(ref_ages, ref_noise)
    
    results = []
    
    for scenario, (min_age, max_age) in age_scenarios:
        # Generate moving site data with restricted age range
        moving_ages = rng.integers(low=min_age, high=max_age, size=100)
        moving_noise = rng.normal(loc=0, scale=SIGMA, size=100)
        moving_data = moving_site.generate_site_data(moving_ages, moving_noise)
        
        # Apply harmonization
        combat = PairwiseComBAT()
        harmonized_data = combat.harmonize_to_reference(
            X_ref=ref_ages, Y_ref=ref_data,
            X_moving=moving_ages, Y_moving=moving_data
        )
        
        # Compute metrics
        gof_before = combat.compute_goodness_of_fit(ref_ages, ref_data, moving_ages, moving_data)
        gof_after = combat.compute_goodness_of_fit(ref_ages, ref_data, moving_ages, harmonized_data)
        
        results.append({
            'scenario': scenario,
            'age_range': f"{min_age}-{max_age}",
            'gof_before': gof_before,
            'gof_after': gof_after,
            'improvement': ((gof_before - gof_after) / gof_before * 100) if gof_before > 0 else 0
        })
    
    # Print results
    print(f"{'Scenario':15} {'Age Range':10} {'Before':>8} {'After':>8} {'Improvement':>12}")
    print("-" * 65)
    for result in results:
        print(f"{result['scenario']:15} {result['age_range']:10} "
              f"{result['gof_before']:8.4f} {result['gof_after']:8.4f} "
              f"{result['improvement']:+11.1f}%")


def main():
    """Main function demonstrating various aspects of Pairwise-ComBAT"""
    # Run the basic demo
    from src.pairwise_combat.core import demo_pairwise_combat
    demo_pairwise_combat()
    
    # Run comparison with different batch effects
    demo_comparison()
    
    # Test with different age ranges
    test_different_age_ranges()


if __name__ == "__main__":
    main()
