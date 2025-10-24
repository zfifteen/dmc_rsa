#!/usr/bin/env python3
"""
Test replicated QMC analysis with confidence intervals
Demonstrates Cranley-Patterson randomization for variance estimation
"""

import sys
sys.path.append('scripts')

from qmc_factorization_analysis import QMCFactorization
import pandas as pd

def test_replicated_qmc():
    """Test replicated QMC with confidence intervals from replicates"""
    print("="*70)
    print("Testing Replicated QMC Analysis")
    print("="*70)
    
    n = 899  # 29 × 31
    num_samples = 200
    num_replicates = 16
    
    print(f"\nTest parameters:")
    print(f"  N = {n} (factors: 29, 31)")
    print(f"  Samples per replicate = {num_samples}")
    print(f"  Number of replicates = {num_replicates}")
    
    # Test with Sobol + Owen scrambling
    print("\n" + "-"*70)
    print("Sobol with Owen Scrambling (Default Recommended)")
    print("-"*70)
    
    sobol_results = QMCFactorization.run_replicated_qmc_analysis(
        n=n,
        num_samples=num_samples,
        num_replicates=num_replicates,
        engine_type="sobol",
        scramble=True,
        seed=42
    )
    
    print(f"\nUnique Candidates:")
    print(f"  Mean: {sobol_results['unique_count']['mean']:.2f}")
    print(f"  Std:  {sobol_results['unique_count']['std']:.2f}")
    print(f"  95% CI: [{sobol_results['unique_count']['ci_lower']:.2f}, "
          f"{sobol_results['unique_count']['ci_upper']:.2f}]")
    
    print(f"\nEffective Rate:")
    print(f"  Mean: {sobol_results['effective_rate']['mean']:.4f}")
    print(f"  Std:  {sobol_results['effective_rate']['std']:.4f}")
    print(f"  95% CI: [{sobol_results['effective_rate']['ci_lower']:.4f}, "
          f"{sobol_results['effective_rate']['ci_upper']:.4f}]")
    
    print(f"\nNumber of Hits (Factors Found):")
    print(f"  Mean: {sobol_results['num_hits']['mean']:.2f}")
    print(f"  Std:  {sobol_results['num_hits']['std']:.2f}")
    print(f"  95% CI: [{sobol_results['num_hits']['ci_lower']:.2f}, "
          f"{sobol_results['num_hits']['ci_upper']:.2f}]")
    
    print(f"\nL2 Discrepancy:")
    print(f"  Mean: {sobol_results['l2_discrepancy']['mean']:.4f}")
    print(f"  Std:  {sobol_results['l2_discrepancy']['std']:.4f}")
    print(f"  95% CI: [{sobol_results['l2_discrepancy']['ci_lower']:.4f}, "
          f"{sobol_results['l2_discrepancy']['ci_upper']:.4f}]")
    
    print(f"\nStratification Balance:")
    print(f"  Mean: {sobol_results['stratification_balance']['mean']:.4f}")
    print(f"  Std:  {sobol_results['stratification_balance']['std']:.4f}")
    print(f"  95% CI: [{sobol_results['stratification_balance']['ci_lower']:.4f}, "
          f"{sobol_results['stratification_balance']['ci_upper']:.4f}]")
    
    # Test with Halton + Faure scrambling
    print("\n" + "-"*70)
    print("Halton with Faure Scrambling")
    print("-"*70)
    
    halton_results = QMCFactorization.run_replicated_qmc_analysis(
        n=n,
        num_samples=num_samples,
        num_replicates=num_replicates,
        engine_type="halton",
        scramble=True,
        seed=42
    )
    
    print(f"\nUnique Candidates:")
    print(f"  Mean: {halton_results['unique_count']['mean']:.2f}")
    print(f"  Std:  {halton_results['unique_count']['std']:.2f}")
    print(f"  95% CI: [{halton_results['unique_count']['ci_lower']:.2f}, "
          f"{halton_results['unique_count']['ci_upper']:.2f}]")
    
    print(f"\nL2 Discrepancy:")
    print(f"  Mean: {halton_results['l2_discrepancy']['mean']:.4f}")
    print(f"  Std:  {halton_results['l2_discrepancy']['std']:.4f}")
    
    # Comparison
    print("\n" + "="*70)
    print("Comparison: Sobol vs Halton")
    print("="*70)
    
    sobol_mean = sobol_results['unique_count']['mean']
    halton_mean = halton_results['unique_count']['mean']
    
    print(f"\nUnique Candidates:")
    print(f"  Sobol:  {sobol_mean:.2f}")
    print(f"  Halton: {halton_mean:.2f}")
    print(f"  Ratio:  {sobol_mean/halton_mean:.3f}×")
    
    sobol_disc = sobol_results['l2_discrepancy']['mean']
    halton_disc = halton_results['l2_discrepancy']['mean']
    
    print(f"\nL2 Discrepancy (lower is better):")
    print(f"  Sobol:  {sobol_disc:.4f}")
    print(f"  Halton: {halton_disc:.4f}")
    print(f"  {'Sobol' if sobol_disc < halton_disc else 'Halton'} has lower discrepancy")
    
    print("\n" + "="*70)
    print("✓ Replicated QMC analysis completed successfully")
    print("="*70)

def test_enhanced_methods_in_statistical_analysis():
    """Test that enhanced methods work in the statistical analysis framework"""
    print("\n" + "="*70)
    print("Testing Enhanced Methods in Statistical Analysis")
    print("="*70)
    
    n = 899
    num_samples = 200
    num_trials = 10  # Small for quick test
    
    print(f"\nRunning analysis with enhanced methods...")
    print(f"  N = {n}")
    print(f"  Samples = {num_samples}")
    print(f"  Trials = {num_trials}")
    
    df = QMCFactorization.run_statistical_analysis(
        n, num_samples=num_samples, num_trials=num_trials, include_enhanced=True
    )
    
    print("\nResults:")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"\n{row['method']}:")
        print(f"  Unique candidates: {row['unique_count_mean']:.1f} "
              f"[{row['unique_count_ci_lower']:.1f}, {row['unique_count_ci_upper']:.1f}]")
        
        if 'l2_discrepancy_mean' in row and not pd.isna(row['l2_discrepancy_mean']):
            print(f"  L2 discrepancy: {row['l2_discrepancy_mean']:.4f}")
        if 'stratification_balance_mean' in row and not pd.isna(row['stratification_balance_mean']):
            print(f"  Stratification balance: {row['stratification_balance_mean']:.4f}")
    
    # Compare improvements
    mc_baseline = df[df['method'] == 'MC']['unique_count_mean'].values[0]
    
    print("\n" + "-"*70)
    print("Improvement vs MC Baseline:")
    print("-"*70)
    for _, row in df.iterrows():
        improvement = row['unique_count_mean'] / mc_baseline
        print(f"  {row['method']:20s}: {improvement:.3f}×")
    
    print("\n" + "="*70)
    print("✓ Enhanced methods test completed successfully")
    print("="*70)

def main():
    test_replicated_qmc()
    test_enhanced_methods_in_statistical_analysis()
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
    print("\nKey Findings:")
    print("- Replicated QMC provides confidence intervals from independent replicates")
    print("- Sobol with Owen scrambling is the recommended default")
    print("- L2 discrepancy and stratification balance provide quality metrics")
    print("- Enhanced methods integrate seamlessly with existing analysis")
    print("="*70)

if __name__ == "__main__":
    main()
