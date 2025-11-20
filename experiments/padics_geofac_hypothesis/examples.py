#!/usr/bin/env python3
"""
Example Usage: P-adic Operations in Practice

This script demonstrates practical applications of the p-adic
module for geofac framework analysis.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.padics_geofac_hypothesis import (
    p_adic_valuation, p_adic_distance, p_adic_expansion,
    p_adic_norm, analyze_geofac_spine, is_ultrametric_valid
)
from wave_crispr_signal import theta_prime, K_Z5D
from cognitive_number_theory import kappa


def example_1_distance_metrics():
    """Example 1: Using p-adic distance as a metric."""
    print("\n" + "="*70)
    print("EXAMPLE 1: P-adic Distance Metrics")
    print("="*70)
    
    # Compare numbers around 1000
    reference = 1000
    candidates = [992, 996, 1004, 1008, 1024]
    
    print(f"\nFinding closest numbers to {reference} in 2-adic metric:")
    
    distances = []
    for n in candidates:
        dist = p_adic_distance(reference, n, p=2)
        distances.append((n, dist))
        print(f"  d({reference}, {n}) = {dist:.6f}")
    
    # Sort by p-adic distance
    distances.sort(key=lambda x: x[1])
    closest = distances[0][0]
    
    print(f"\nClosest number in 2-adic metric: {closest}")
    print(f"(Because {abs(closest - reference)} = {closest - reference} is highly divisible by 2)")


def example_2_clustering_analysis():
    """Example 2: Analyzing clustering with ultrametric."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Ultrametric Clustering Analysis")
    print("="*70)
    
    # Check if three numbers form a tight cluster
    cluster = [1000, 1024, 1040]
    p = 2
    
    print(f"\nAnalyzing cluster {cluster} with p={p}:")
    
    # Compute all pairwise distances
    d01 = p_adic_distance(cluster[0], cluster[1], p)
    d12 = p_adic_distance(cluster[1], cluster[2], p)
    d02 = p_adic_distance(cluster[0], cluster[2], p)
    
    print(f"  d({cluster[0]}, {cluster[1]}) = {d01:.6f}")
    print(f"  d({cluster[1]}, {cluster[2]}) = {d12:.6f}")
    print(f"  d({cluster[0]}, {cluster[2]}) = {d02:.6f}")
    
    # Verify ultrametric property
    is_valid = is_ultrametric_valid(*cluster, p)
    print(f"\nUltrametric property holds: {is_valid}")
    
    if is_valid:
        print("✓ These numbers form a valid ultrametric cluster")
        print("  (non-overlapping with other clusters)")


def example_3_geofac_spine():
    """Example 3: Analyzing geofac spine structure."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Geofac Spine Analysis")
    print("="*70)
    
    n = 2024
    p = 2
    
    print(f"\nAnalyzing n={n} in base-{p}:")
    
    # Get p-adic valuation
    val = p_adic_valuation(n, p)
    print(f"  p-adic valuation: v_{p}({n}) = {val}")
    print(f"  Meaning: {n} = {p}^{val} × {n // (p**val)}")
    
    # Get p-adic expansion
    expansion = p_adic_expansion(n, p, num_digits=12)
    print(f"\n  p-adic expansion (first 12 digits):")
    print(f"  {expansion}")
    
    # Show as traditional base notation
    print(f"\n  In base-{p} (traditional notation):")
    print(f"  {n} = ", end="")
    for i in range(len(expansion)-1, -1, -1):
        if expansion[i] != 0 or i == 0:
            print(f"{expansion[i]}", end="")
    print(f"_{p}")
    
    # Analyze spine
    spine = analyze_geofac_spine(n, p, max_level=5)
    print(f"\n  Geofac spine structure:")
    for k, residue, norm in spine:
        print(f"    Level {k}: {n} ≡ {residue} (mod {p}^{k})")


def example_4_integration_with_z_framework():
    """Example 4: Combining p-adic with Z-framework."""
    print("\n" + "="*70)
    print("EXAMPLE 4: P-adic + Z-Framework Integration")
    print("="*70)
    
    n = 10000
    
    # Compute Z-framework values
    theta = theta_prime(n, k=K_Z5D)
    kappa_n = kappa(n)
    
    # Compute p-adic values for multiple primes
    print(f"\nAnalyzing n={n}:")
    print(f"\n  Z-framework metrics:")
    print(f"    θ'({n}, k={K_Z5D:.5f}) = {theta:.6f}")
    print(f"    κ({n}) = {kappa_n:.6f}")
    
    print(f"\n  P-adic structure:")
    for p in [2, 5]:
        val = p_adic_valuation(n, p)
        norm = p_adic_norm(n, p)
        print(f"    {p}-adic: v_{p}({n}) = {val}, |{n}|_{p} = {norm:.6f}")
    
    print(f"\n  Interpretation:")
    print(f"    - θ' provides golden-ratio bias for optimization")
    print(f"    - κ measures divisor density curvature")
    print(f"    - p-adic valuation shows prime factorization structure")
    print(f"    - All three are complementary, not competing!")


def example_5_practical_factorization():
    """Example 5: Using p-adic metrics for factorization hints."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Factorization Hints from P-adic Analysis")
    print("="*70)
    
    # Analyze 899 = 29 × 31
    N = 899
    print(f"\nAnalyzing N={N}:")
    
    # Check p-adic valuations for small primes
    print(f"\n  P-adic valuations:")
    for p in [2, 3, 5, 7, 11, 13]:
        val = p_adic_valuation(N, p)
        if val > 0:
            print(f"    v_{p}({N}) = {val} ✓ (divisible by {p})")
        else:
            print(f"    v_{p}({N}) = {val}   (not divisible by {p})")
    
    print(f"\n  Analysis:")
    print(f"    - All small prime valuations are 0")
    print(f"    - This suggests {N} is a product of larger primes")
    print(f"    - Indeed: {N} = 29 × 31")
    
    # Show distances in different p-adic metrics
    print(f"\n  Distance from nearby numbers (2-adic):")
    for delta in [-1, 1, 8, 25]:
        n = N + delta
        d = p_adic_distance(N, n, p=2)
        print(f"    d({N}, {n}) = {d:.6f}")


def main():
    """Run all examples."""
    print("="*70)
    print("P-ADIC OPERATIONS: PRACTICAL EXAMPLES")
    print("="*70)
    
    example_1_distance_metrics()
    example_2_clustering_analysis()
    example_3_geofac_spine()
    example_4_integration_with_z_framework()
    example_5_practical_factorization()
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print("\nThese examples show how to:")
    print("  1. Use p-adic distance as an alternative metric")
    print("  2. Verify ultrametric clustering properties")
    print("  3. Analyze geofac spine structure")
    print("  4. Combine p-adic with Z-framework metrics")
    print("  5. Extract factorization hints from p-adic analysis")
    print("\nFor more details, see:")
    print("  - experiments/padics_geofac_hypothesis/README.md")
    print("  - experiments/padics_geofac_hypothesis/QUICK_REFERENCE.md")


if __name__ == '__main__':
    main()
