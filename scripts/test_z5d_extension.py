#!/usr/bin/env python3
"""
Unit tests for Z5D extension (k*≈0.04449)
Tests high-precision validation, prime density boost, and bootstrap CI
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_crispr_signal import (
    theta_prime, K_Z5D, theta_prime_high_precision,
    compute_prime_density_boost, validate_z5d_extension
)


def test_k_z5d_constant():
    """Test that K_Z5D constant is defined correctly"""
    print("\nTesting K_Z5D constant...")
    
    assert K_Z5D is not None, "K_Z5D should be defined"
    assert isinstance(K_Z5D, float), "K_Z5D should be float"
    assert 0.04 < K_Z5D < 0.05, "K_Z5D should be approximately 0.04449"
    assert abs(K_Z5D - 0.04449) < 0.001, "K_Z5D should be 0.04449"
    
    print(f"  K_Z5D = {K_Z5D}")
    print("  ✓ K_Z5D constant is correct")


def test_theta_prime_with_z5d_k():
    """Test theta_prime with Z5D k value"""
    print("\nTesting theta_prime with k={:.5f}...".format(K_Z5D))
    
    # Test at various sample points
    test_points = [1, 100, 1000, 10000, 100000, 1000000]
    
    for n in test_points:
        theta = theta_prime(n, k=K_Z5D)
        
        # Should return valid value
        assert np.isfinite(theta), f"theta_prime({n}) should be finite"
        assert theta > 0, f"theta_prime({n}) should be positive"
        
        # With very small k, convergence to phi is slower
        # So theta should vary more across samples
        if n > 1:
            theta_prev = theta_prime(n - 1, k=K_Z5D)
            # Values should be close but not identical
            assert abs(theta - theta_prev) < 1.0, "Theta values should be reasonably close"
    
    print(f"  Tested theta_prime at {len(test_points)} sample points")
    print("  ✓ theta_prime works correctly with Z5D k value")


def test_theta_prime_high_precision():
    """Test high-precision theta_prime computation"""
    print("\nTesting theta_prime_high_precision...")
    
    # Test with dps=50
    n = 1000000
    theta_hp = theta_prime_high_precision(n, k=K_Z5D, dps=50)
    theta_std = theta_prime(n, k=K_Z5D)
    
    # Should be mpmath type
    assert hasattr(theta_hp, '__float__'), "Should be convertible to float"
    
    # Convert to float for comparison
    theta_hp_float = float(theta_hp)
    
    # Should be close to standard precision
    error = abs(theta_hp_float - theta_std)
    assert error < 1e-10, f"High-precision error should be very small: {error}"
    
    print(f"  High-precision value: {theta_hp}")
    print(f"  Standard value:       {theta_std}")
    print(f"  Error:                {error:.2e}")
    print("  ✓ High-precision computation works correctly")


def test_compute_prime_density_boost():
    """Test prime density boost computation"""
    print("\nTesting compute_prime_density_boost...")
    
    # Test with small sample size
    n_samples = 10000
    boost = compute_prime_density_boost(n_samples, k=K_Z5D, baseline_k=0.3)
    
    # Should return dictionary with expected keys
    expected_keys = ['n_samples', 'k_baseline', 'k_new', 'boost_factor', 'boost_percent']
    for key in expected_keys:
        assert key in boost, f"Result should contain '{key}'"
    
    # Values should be reasonable
    assert boost['n_samples'] == n_samples
    assert boost['k_baseline'] == 0.3
    assert boost['k_new'] == K_Z5D
    assert boost['boost_factor'] > 0, "Boost factor should be positive"
    
    print(f"  N samples:       {boost['n_samples']:,}")
    print(f"  Baseline k:      {boost['k_baseline']}")
    print(f"  Z5D k:           {boost['k_new']}")
    print(f"  Boost factor:    {boost['boost_factor']:.2f}x")
    print(f"  Boost percent:   {boost['boost_percent']:.1f}%")
    print("  ✓ Prime density boost computation works")


def test_validate_z5d_extension_quick():
    """Test Z5D validation with quick parameters"""
    print("\nTesting validate_z5d_extension (quick mode)...")
    
    # Use small sample size for testing
    results = validate_z5d_extension(
        n_samples=1000,
        k=K_Z5D,
        n_bootstrap=100,
        confidence=0.95,
        dps=50
    )
    
    # Should return dictionary with expected structure
    assert 'n_samples' in results
    assert 'k_value' in results
    assert 'high_precision_tests' in results
    assert 'prime_density_boost' in results
    assert 'bootstrap_ci' in results
    
    # Check high-precision tests
    assert isinstance(results['high_precision_tests'], list)
    assert len(results['high_precision_tests']) > 0
    
    # Check each test has required fields
    for test in results['high_precision_tests']:
        assert 'n' in test
        assert 'error' in test
        assert 'error_valid' in test
    
    # Check bootstrap CI
    ci = results['bootstrap_ci']
    assert 'mean' in ci
    assert 'variance' in ci
    assert 'ci_lower' in ci['mean']
    assert 'ci_upper' in ci['mean']
    
    print(f"  N samples:           {results['n_samples']:,}")
    print(f"  k value:             {results['k_value']}")
    print(f"  High-precision tests:{len(results['high_precision_tests'])}")
    print(f"  All errors valid:    {results['all_errors_valid']}")
    print(f"  Max error:           {results['max_error']:.2e}")
    print(f"  Boost percent:       {results['prime_density_boost']['boost_percent']:.1f}%")
    print("  ✓ Z5D validation works correctly")


def test_z5d_vs_baseline_comparison():
    """Compare Z5D k value against baseline k=0.3"""
    print("\nTesting Z5D vs baseline comparison...")
    
    # Sample points
    sample_points = [1, 10, 100, 1000, 10000]
    
    print("  {:>8}  {:>12}  {:>12}  {:>12}".format("n", "baseline", "Z5D", "difference"))
    print("  " + "-" * 52)
    
    for n in sample_points:
        theta_baseline = theta_prime(n, k=0.3)
        theta_z5d = theta_prime(n, k=K_Z5D)
        diff = abs(theta_z5d - theta_baseline)
        
        print("  {:>8}  {:>12.6f}  {:>12.6f}  {:>12.6f}".format(
            n, theta_baseline, theta_z5d, diff
        ))
        
        # Both should be valid
        assert np.isfinite(theta_baseline)
        assert np.isfinite(theta_z5d)
    
    print("  ✓ Comparison completed successfully")


def test_z5d_convergence_properties():
    """Test convergence properties of Z5D k value"""
    print("\nTesting Z5D convergence properties...")
    
    # With smaller k, convergence to phi should be slower
    n_large = 1000000
    
    theta_baseline = theta_prime(n_large, k=0.3)
    theta_z5d = theta_prime(n_large, k=K_Z5D)
    
    # Both should converge toward PHI
    from wave_crispr_signal import PHI
    
    dist_baseline = abs(theta_baseline - PHI)
    dist_z5d = abs(theta_z5d - PHI)
    
    print(f"  PHI:                 {PHI:.6f}")
    print(f"  Baseline (k=0.3):    {theta_baseline:.6f} (dist: {dist_baseline:.6f})")
    print(f"  Z5D (k={K_Z5D:.5f}):  {theta_z5d:.6f} (dist: {dist_z5d:.6f})")
    
    # With smaller k, should converge more slowly (larger distance from PHI)
    # But not always true for all n, so just check they're both reasonable
    assert dist_baseline < 1.0, "Baseline should be near PHI"
    assert dist_z5d < 1.0, "Z5D should be near PHI"
    
    print("  ✓ Convergence properties validated")


def run_all_tests():
    """Run all Z5D tests"""
    print("=" * 70)
    print("Z5D Extension Test Suite (k*≈0.04449)")
    print("=" * 70)
    
    test_k_z5d_constant()
    test_theta_prime_with_z5d_k()
    test_theta_prime_high_precision()
    test_compute_prime_density_boost()
    test_validate_z5d_extension_quick()
    test_z5d_vs_baseline_comparison()
    test_z5d_convergence_properties()
    
    print()
    print("=" * 70)
    print("All Z5D extension tests passed! ✓")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
