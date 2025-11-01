#!/usr/bin/env python3
"""
Example usage of Elliptic Adaptive Search (EAS) for RSA factorization

This demonstrates the key features of EAS:
1. Elliptic lattice point generation with golden-angle spiral
2. Adaptive window sizing based on bit length
3. Prime density modeling near √N
4. Performance characteristics across different bit sizes
"""

import sys
sys.path.append('scripts')

import numpy as np
from eas_factorize import (
    EllipticAdaptiveSearch, EASConfig, factorize_eas, benchmark_eas
)


def example_1_basic_usage():
    """Example 1: Basic factorization with default settings"""
    print("="*70)
    print("Example 1: Basic Factorization")
    print("="*70)
    
    # Factor a known semiprime
    n = 899  # 29 × 31
    print(f"\nFactoring N = {n}")
    
    result = factorize_eas(n, verbose=True)
    
    if result.success:
        print(f"\n✓ Successfully factored!")
        print(f"  Factors: {result.factor_p} × {result.factor_q} = {n}")
    else:
        print(f"\n✗ Failed to factor")
    
    print(f"\nStatistics:")
    print(f"  Candidates checked: {result.candidates_checked}")
    print(f"  Search reduction: {result.search_reduction:.1f}×")
    print(f"  Time elapsed: {result.time_elapsed:.4f}s")


def example_2_custom_configuration():
    """Example 2: Using custom EAS configuration"""
    print("\n" + "="*70)
    print("Example 2: Custom Configuration")
    print("="*70)
    
    # Create custom configuration
    config = EASConfig(
        max_samples=500,          # Limit search space
        adaptive_window=True,      # Enable adaptive window sizing
        base_radius_factor=0.15,  # Wider search radius
        elliptic_eccentricity=0.7  # More circular ellipses
    )
    
    print(f"\nConfiguration:")
    print(f"  Max samples: {config.max_samples}")
    print(f"  Adaptive window: {config.adaptive_window}")
    print(f"  Base radius factor: {config.base_radius_factor}")
    print(f"  Elliptic eccentricity: {config.elliptic_eccentricity}")
    
    # Factor with custom config
    n = 1147  # 31 × 37
    print(f"\nFactoring N = {n}")
    
    result = factorize_eas(n, config=config, verbose=True)


def example_3_performance_analysis():
    """Example 3: Analyze performance across bit sizes"""
    print("\n" + "="*70)
    print("Example 3: Performance Analysis Across Bit Sizes")
    print("="*70)
    
    # Test on small bit sizes for demonstration
    bit_sizes = [16, 20, 24]
    trials = 5
    
    print(f"\nTesting {trials} trials per bit size...")
    print(f"Bit sizes: {bit_sizes}")
    
    results = benchmark_eas(
        bit_sizes=bit_sizes,
        trials_per_size=trials,
        verbose=True
    )
    
    # Summarize results
    print("\n" + "="*70)
    print("Performance Summary")
    print("="*70)
    print(f"{'Bits':<8} {'Success Rate':<15} {'Avg Checks':<15} {'Search Reduction':<20}")
    print("-" * 70)
    
    for bits in bit_sizes:
        r = results[bits]
        print(f"{bits:<8} {r['success_rate']*100:>5.1f}%         "
              f"{r['avg_checks']:>8.0f}        {r['avg_reduction']:>10.0f}×")


def example_4_comparing_adaptive_window():
    """Example 4: Compare adaptive vs fixed window sizing"""
    print("\n" + "="*70)
    print("Example 4: Adaptive vs Fixed Window Sizing")
    print("="*70)
    
    test_cases = [
        (143, "11 × 13", 8),   # 8-bit
        (899, "29 × 31", 10),  # 10-bit
        (2491, "47 × 53", 12), # 12-bit
    ]
    
    for n, factors, bits in test_cases:
        print(f"\n{'-'*70}")
        print(f"N = {n} ({factors}, {bits} bits)")
        print(f"{'-'*70}")
        
        # Test with adaptive window
        config_adaptive = EASConfig(adaptive_window=True)
        result_adaptive = factorize_eas(n, config=config_adaptive, verbose=False)
        
        # Test with fixed window
        config_fixed = EASConfig(adaptive_window=False, base_radius_factor=0.1)
        result_fixed = factorize_eas(n, config=config_fixed, verbose=False)
        
        print(f"\nAdaptive Window:")
        print(f"  Success: {result_adaptive.success}")
        print(f"  Checks: {result_adaptive.candidates_checked}")
        print(f"  Reduction: {result_adaptive.search_reduction:.1f}×")
        
        print(f"\nFixed Window:")
        print(f"  Success: {result_fixed.success}")
        print(f"  Checks: {result_fixed.candidates_checked}")
        print(f"  Reduction: {result_fixed.search_reduction:.1f}×")


def example_5_elliptic_lattice_visualization():
    """Example 5: Visualize elliptic lattice point generation"""
    print("\n" + "="*70)
    print("Example 5: Elliptic Lattice Point Generation")
    print("="*70)
    
    # Create EAS instance
    eas = EllipticAdaptiveSearch()
    
    # Generate points
    sqrt_n = 100.0
    radius = 10.0
    n_points = 50
    
    print(f"\nGenerating elliptic lattice points:")
    print(f"  Center (√N): {sqrt_n}")
    print(f"  Radius: {radius}")
    print(f"  Number of points: {n_points}")
    
    candidates = eas._generate_elliptic_lattice_points(n_points, sqrt_n, radius)
    
    print(f"\nGenerated {len(candidates)} total candidates")
    print(f"Unique candidates: {len(np.unique(candidates))}")
    print(f"Range: [{np.min(candidates)}, {np.max(candidates)}]")
    
    # Show distribution statistics
    distances = np.abs(candidates - sqrt_n)
    print(f"\nDistance from √N:")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Median: {np.median(distances):.2f}")
    print(f"  Max: {np.max(distances):.2f}")
    
    # Show golden angle property
    golden_angle_deg = np.degrees(eas.config.golden_angle)
    print(f"\nGolden angle: {golden_angle_deg:.2f}° (~137.5°)")
    print(f"This provides optimal angular distribution without radial alignment")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Elliptic Adaptive Search (EAS) - Usage Examples")
    print("="*70)
    print("\nThese examples demonstrate key features of EAS for RSA factorization.")
    print("EAS uses elliptic lattice sampling with golden-angle spiral to")
    print("efficiently explore factor space near √N.\n")
    
    examples = [
        example_1_basic_usage,
        example_2_custom_configuration,
        example_3_performance_analysis,
        example_4_comparing_adaptive_window,
        example_5_elliptic_lattice_visualization,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n✗ Example failed with error: {e}")
    
    print("\n" + "="*70)
    print("Examples Complete")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. EAS performs best on smaller factors (16-40 bits)")
    print("2. Adaptive window sizing improves success rates")
    print("3. Elliptic + golden-angle provides efficient geometric sampling")
    print("4. Search space reduction can be orders of magnitude")
    print("\nFor production use, consider:")
    print("- Hybrid approaches combining EAS with ECM or QS")
    print("- Tuning parameters based on target bit size")
    print("- Using as a candidate generator for other methods")


if __name__ == "__main__":
    main()
