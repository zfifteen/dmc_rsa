#!/usr/bin/env python3
"""
QMC Directions Demo - Comprehensive demonstration of enhanced QMC capabilities
October 2025

This script demonstrates:
1. Replicated QMC with Cranley-Patterson randomization
2. Sobol with Owen scrambling (recommended default)
3. L2 discrepancy and stratification balance metrics
4. Confidence intervals from independent replicates
5. Comparison of different QMC engines

Run this to see the full power of the enhanced QMC implementation!
"""

import sys
sys.path.append('../scripts')

from qmc_factorization_analysis import QMCFactorization
from qmc_engines import QMCConfig, qmc_points, map_points_to_candidates
import numpy as np
import pandas as pd

def demo_replicated_qmc():
    """Demonstrate replicated QMC with confidence intervals"""
    print("="*80)
    print("DEMO 1: Replicated QMC with Confidence Intervals")
    print("="*80)
    print("\nThis demonstrates Cranley-Patterson randomization: multiple independent")
    print("QMC replicates with different random seeds provide unbiased error bars.")
    
    # Test cases: different semiprimes
    test_cases = [
        (899, "29×31 (balanced)"),
        (93, "3×31 (unbalanced)"),
        (221, "13×17 (medium balanced)")
    ]
    
    for n, description in test_cases:
        print(f"\n{'-'*80}")
        print(f"Testing N={n} ({description})")
        print('-'*80)
        
        results = QMCFactorization.run_replicated_qmc_analysis(
            n=n,
            num_samples=256,  # Power of 2 for Sobol
            num_replicates=16,
            engine_type="sobol",
            scramble=True,
            seed=42
        )
        
        print(f"\nResults from {results['num_replicates']} independent replicates:")
        print(f"  Unique candidates: {results['unique_count']['mean']:.2f} ± {results['unique_count']['std']:.2f}")
        print(f"  95% CI: [{results['unique_count']['ci_lower']:.2f}, {results['unique_count']['ci_upper']:.2f}]")
        print(f"  Effective rate: {results['effective_rate']['mean']:.4f}")
        print(f"  Factors found: {results['num_hits']['mean']:.2f} (avg per replicate)")
        print(f"\nQuality Metrics:")
        print(f"  L2 discrepancy: {results['l2_discrepancy']['mean']:.4f} ± {results['l2_discrepancy']['std']:.4f}")
        print(f"  Stratification balance: {results['stratification_balance']['mean']:.4f} (higher is better)")

def demo_engine_comparison():
    """Compare different QMC engines"""
    print("\n" + "="*80)
    print("DEMO 2: Engine Comparison (Sobol vs Halton)")
    print("="*80)
    print("\nComparing Sobol (with Owen scrambling) vs Halton (with Faure permutations)")
    
    n = 899
    num_samples = 256
    num_replicates = 16
    
    engines = [
        ("sobol", "Sobol + Owen (Recommended)"),
        ("halton", "Halton + Faure")
    ]
    
    results_by_engine = {}
    
    for engine_type, name in engines:
        print(f"\n{'-'*80}")
        print(f"{name}")
        print('-'*80)
        
        results = QMCFactorization.run_replicated_qmc_analysis(
            n=n,
            num_samples=num_samples,
            num_replicates=num_replicates,
            engine_type=engine_type,
            scramble=True,
            seed=42
        )
        
        results_by_engine[engine_type] = results
        
        print(f"  Unique candidates: {results['unique_count']['mean']:.2f} ± {results['unique_count']['std']:.2f}")
        print(f"  L2 discrepancy: {results['l2_discrepancy']['mean']:.4f}")
        print(f"  Stratification balance: {results['stratification_balance']['mean']:.4f}")
    
    # Winner analysis
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    sobol_unique = results_by_engine['sobol']['unique_count']['mean']
    halton_unique = results_by_engine['halton']['unique_count']['mean']
    
    print(f"\nUnique Candidates:")
    print(f"  Sobol:  {sobol_unique:.2f}")
    print(f"  Halton: {halton_unique:.2f}")
    print(f"  Winner: {'Sobol' if sobol_unique > halton_unique else 'Halton'}")
    
    sobol_disc = results_by_engine['sobol']['l2_discrepancy']['mean']
    halton_disc = results_by_engine['halton']['l2_discrepancy']['mean']
    
    print(f"\nL2 Discrepancy (lower is better):")
    print(f"  Sobol:  {sobol_disc:.4f}")
    print(f"  Halton: {halton_disc:.4f}")
    print(f"  Winner: {'Sobol' if sobol_disc < halton_disc else 'Halton'}")
    
    print("\n✓ Sobol with Owen scrambling is the recommended default for most applications.")

def demo_scrambling_effect():
    """Demonstrate the effect of Owen scrambling"""
    print("\n" + "="*80)
    print("DEMO 3: Effect of Owen Scrambling")
    print("="*80)
    print("\nComparing scrambled vs unscrambled Sobol sequences")
    
    n = 899
    num_samples = 256
    num_replicates = 16
    
    for scramble in [False, True]:
        name = "WITH Owen scrambling" if scramble else "WITHOUT scrambling"
        print(f"\n{'-'*80}")
        print(f"Sobol {name}")
        print('-'*80)
        
        results = QMCFactorization.run_replicated_qmc_analysis(
            n=n,
            num_samples=num_samples,
            num_replicates=num_replicates,
            engine_type="sobol",
            scramble=scramble,
            seed=42
        )
        
        print(f"  Unique candidates: {results['unique_count']['mean']:.2f} ± {results['unique_count']['std']:.2f}")
        print(f"  Variance: {results['unique_count']['std']**2:.4f}")
        print(f"  L2 discrepancy: {results['l2_discrepancy']['mean']:.4f}")
        print(f"  Stratification balance: {results['stratification_balance']['mean']:.4f}")
    
    print("\n✓ Owen scrambling provides randomization for unbiased variance estimation")
    print("  while preserving low-discrepancy properties.")

def demo_statistical_significance():
    """Demonstrate statistical significance testing"""
    print("\n" + "="*80)
    print("DEMO 4: Statistical Significance (QMC vs MC)")
    print("="*80)
    print("\nRunning multiple trials to establish statistical significance")
    
    n = 899
    num_samples = 256
    num_trials = 50  # Moderate number for demo
    
    print(f"\nTest setup:")
    print(f"  N = {n}")
    print(f"  Samples per trial = {num_samples}")
    print(f"  Number of trials = {num_trials}")
    print(f"\nRunning analysis...")
    
    df = QMCFactorization.run_statistical_analysis(
        n, num_samples=num_samples, num_trials=num_trials, include_enhanced=True
    )
    
    mc_baseline = df[df['method'] == 'MC']['unique_count_mean'].values[0]
    mc_ci_lower = df[df['method'] == 'MC']['unique_count_ci_lower'].values[0]
    mc_ci_upper = df[df['method'] == 'MC']['unique_count_ci_upper'].values[0]
    
    print(f"\nMC Baseline:")
    print(f"  Mean: {mc_baseline:.2f}")
    print(f"  95% CI: [{mc_ci_lower:.2f}, {mc_ci_upper:.2f}]")
    
    print(f"\nComparison with other methods:")
    print('-'*80)
    for _, row in df.iterrows():
        if row['method'] == 'MC':
            continue
        
        improvement = row['unique_count_mean'] / mc_baseline
        ci_lower = row['unique_count_ci_lower']
        ci_upper = row['unique_count_ci_upper']
        
        # Check if CI overlaps with MC baseline CI
        overlaps = not (ci_lower > mc_ci_upper or ci_upper < mc_ci_lower)
        significance = "Not significant" if overlaps else "Significant"
        
        print(f"\n{row['method']:20s}:")
        print(f"  Mean: {row['unique_count_mean']:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Improvement: {improvement:.3f}× vs MC")
        print(f"  Statistical significance: {significance}")

def demo_usage_recommendations():
    """Provide usage recommendations"""
    print("\n" + "="*80)
    print("USAGE RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on theory and experimental validation:

1. DEFAULT ENGINE: Use Sobol with Owen scrambling
   - Best multidimensional stratification
   - Owen scrambling provides unbiased variance estimation
   - Supported by scipy.stats.qmc.Sobol(scramble=True)
   
2. REPLICATES: Use 8-32 independent replicates
   - Provides confidence intervals via Cranley-Patterson randomization
   - Each replicate uses different random seed
   - More replicates = tighter confidence intervals
   
3. SAMPLE SIZE: Use powers of 2 for Sobol
   - n = 256, 512, 1024, 2048, etc.
   - Maintains balance properties of Sobol sequences
   - scipy will warn if not power of 2
   
4. DIMENSIONS: Keep dimensions modest (≤ 8-12)
   - QMC advantage diminishes in high dimensions
   - Use coordinate weights if some dimensions matter more
   - Current implementation uses 2D (window position + residue class)
   
5. MAPPING: Use smooth transformations
   - Avoid hard discontinuities in candidate mapping
   - Soft edges and bounded adjustments preserve low discrepancy
   - map_points_to_candidates implements this
   
6. METRICS: Report both point-set and candidate quality
   - L2 discrepancy: measures uniformity in [0,1)^d
   - Stratification balance: measures bin distribution
   - Unique candidates: actual performance metric
   - Confidence intervals: statistical rigor

Example code:

    from qmc_factorization_analysis import QMCFactorization
    
    results = QMCFactorization.run_replicated_qmc_analysis(
        n=899,
        num_samples=256,      # Power of 2
        num_replicates=16,    # For confidence intervals
        engine_type="sobol",  # Recommended
        scramble=True,        # Owen scrambling
        seed=42               # Reproducibility
    )
    
    print(f"Mean unique candidates: {results['unique_count']['mean']:.2f}")
    print(f"95% CI: [{results['unique_count']['ci_lower']:.2f}, "
          f"{results['unique_count']['ci_upper']:.2f}]")
""")

def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("QMC DIRECTIONS - COMPREHENSIVE DEMONSTRATION")
    print("Enhanced QMC for RSA Factorization")
    print("October 2025")
    print("="*80)
    
    demo_replicated_qmc()
    demo_engine_comparison()
    demo_scrambling_effect()
    demo_statistical_significance()
    demo_usage_recommendations()
    
    print("\n" + "="*80)
    print("✓ All demos completed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("- Run these methods on your specific RSA numbers")
    print("- Adjust num_replicates based on desired CI width")
    print("- Use enhanced metrics for paper/report")
    print("- Extend to ECM σ sampling and GNFS parameter sweeps")
    print("="*80)

if __name__ == "__main__":
    main()
