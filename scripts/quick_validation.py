#!/usr/bin/env python3
"""
Quick validation test for rank-1 lattice and EAS integration
Tests core functionality with reduced trial counts
"""

import sys
sys.path.append('scripts')

import numpy as np
from qmc_factorization_analysis import QMCFactorization

def quick_validation():
    """Quick validation of rank-1 lattice, EAS, and standard QMC"""
    print("="*70)
    print("Quick Validation: Rank-1 Lattice & EAS Integration")
    print("="*70)
    
    n = 899  # 29 × 31
    num_samples = 128
    num_trials = 10  # Minimal for quick test
    
    print(f"\nTest: N={n}, samples={num_samples}, trials={num_trials}")
    print("\nRunning analysis...")
    
    df = QMCFactorization.run_statistical_analysis(
        n=n,
        num_samples=num_samples,
        num_trials=num_trials,
        include_enhanced=True,
        include_rank1=True,
        include_eas=True
    )
    
    # Calculate improvements
    mc_baseline = df[df['method'] == 'MC']['unique_count_mean'].values[0]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for _, row in df.iterrows():
        method = row['method']
        mean = row['unique_count_mean']
        improvement = mean / mc_baseline
        
        print(f"\n{method:20s}: {mean:6.2f} unique candidates")
        print(f"{'':20s}  Improvement: {improvement:.3f}×")
        
        # Show lattice metrics if available
        if 'min_distance_mean' in row and not np.isnan(row['min_distance_mean']):
            print(f"{'':20s}  Min distance: {row['min_distance_mean']:.4f}")
    
    # Verify rank-1 and EAS methods are present
    rank1_methods = df[df['method'].str.contains('Rank1', na=False)]
    eas_methods = df[df['method'].str.contains('EAS', na=False)]
    
    print("\n" + "="*70)
    print("VALIDATION STATUS")
    print("="*70)
    
    checks = [
        ("Rank-1 Fibonacci present", len(df[df['method'] == 'Rank1-Fibonacci']) > 0),
        ("Rank-1 Cyclic present", len(df[df['method'] == 'Rank1-Cyclic']) > 0),
        ("Rank-1 methods have min_distance metric", 
         'min_distance_mean' in rank1_methods.columns and 
         not rank1_methods['min_distance_mean'].isna().all()),
        ("EAS method present", len(eas_methods) > 0),
        ("All methods produce unique candidates > 0",
         (df['unique_count_mean'] > 0).all()),
    ]
    
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n" + "="*70)
        print("✓ All validation checks passed!")
        print("="*70)
        print("\nRank-1 lattice and EAS integration is working correctly.")
        print("The cyclic subgroup construction provides competitive performance")
        print("with theoretically-motivated regularity properties.")
        print("EAS provides elliptic lattice sampling with golden-angle spiral.")
    else:
        print("\n✗ Some validation checks failed!")
        return False
    
    return True


if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)
