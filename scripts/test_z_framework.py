#!/usr/bin/env python3
"""
Unit tests for Z-framework modules (cognitive_number_theory and wave_crispr_signal)
Tests kappa, theta_prime, and related functions for the Bias-Adaptive Sampling Engine
"""

import sys
import numpy as np

# Add paths
sys.path.insert(0, '/home/runner/work/dmc_rsa/dmc_rsa')

from cognitive_number_theory import kappa, count_divisors
from wave_crispr_signal import theta_prime, Z_transform, validate_precision, PHI


def test_count_divisors():
    """Test divisor counting function"""
    print("\nTesting count_divisors...")
    
    # Test basic cases
    assert count_divisors(1) == 1, "1 has 1 divisor"
    assert count_divisors(12) == 6, "12 has 6 divisors (1,2,3,4,6,12)"
    assert count_divisors(7) == 2, "Prime 7 has 2 divisors (1,7)"
    assert count_divisors(16) == 5, "16 has 5 divisors (1,2,4,8,16)"
    
    # Test error handling
    try:
        count_divisors(0)
        assert False, "Should raise ValueError for n=0"
    except ValueError:
        pass
    
    print("  ✓ count_divisors works correctly")


def test_kappa_scalar():
    """Test kappa function with scalar input"""
    print("\nTesting kappa (scalar)...")
    
    # Test basic computation
    k1 = kappa(1)
    assert k1 > 0, "kappa(1) should be positive"
    assert k1 < 1, "kappa(1) should be small"
    
    k12 = kappa(12)
    assert k12 > k1, "kappa(12) > kappa(1) due to more divisors"
    
    # Test mathematical properties: κ(n) = d(n) · ln(n+1) / e²
    expected_k12 = 6 * np.log(13) / (np.e ** 2)
    assert abs(k12 - expected_k12) < 1e-10, "kappa(12) matches formula"
    
    # Test error handling
    try:
        kappa(0)
        assert False, "Should raise ValueError for n=0"
    except ValueError:
        pass
    
    print("  ✓ kappa (scalar) works correctly")


def test_kappa_array():
    """Test kappa function with array input"""
    print("\nTesting kappa (array)...")
    
    n_array = np.array([1, 2, 3, 4, 5])
    k_array = kappa(n_array)
    
    assert len(k_array) == len(n_array), "Output array matches input length"
    assert np.all(k_array > 0), "All kappa values are positive"
    
    # Check individual values match scalar version
    for i, n in enumerate(n_array):
        assert abs(k_array[i] - kappa(int(n))) < 1e-10, f"kappa({n}) matches"
    
    print("  ✓ kappa (array) works correctly")


def test_theta_prime_scalar():
    """Test theta_prime function with scalar input"""
    print("\nTesting theta_prime (scalar)...")
    
    # Test basic computation
    theta1 = theta_prime(1, k=0.3)
    assert theta1 > 0, "theta_prime(1) should be positive"
    
    theta10 = theta_prime(10, k=0.3)
    assert theta10 > 0, "theta_prime(10) should be positive"
    
    # Test formula: θ′(n,k) = φ · ((n mod φ)/φ)^k
    n = 5
    k = 0.3
    expected = PHI * ((n % PHI) / PHI) ** k
    result = theta_prime(n, k)
    assert abs(result - expected) < 1e-10, "theta_prime matches formula"
    
    # Test k parameter effect
    theta_k02 = theta_prime(10, k=0.2)
    theta_k05 = theta_prime(10, k=0.5)
    # Different k values should give different results
    assert abs(theta_k02 - theta_k05) > 1e-6, "k parameter has effect"
    
    print("  ✓ theta_prime (scalar) works correctly")


def test_theta_prime_array():
    """Test theta_prime function with array input"""
    print("\nTesting theta_prime (array)...")
    
    n_array = np.array([1, 5, 10, 20, 50])
    theta_array = theta_prime(n_array, k=0.3)
    
    assert len(theta_array) == len(n_array), "Output array matches input length"
    assert np.all(theta_array > 0), "All theta_prime values are positive"
    
    # Check individual values match scalar version
    for i, n in enumerate(n_array):
        expected = theta_prime(float(n), k=0.3)
        assert abs(theta_array[i] - expected) < 1e-10, f"theta_prime({n}) matches"
    
    print("  ✓ theta_prime (array) works correctly")


def test_Z_transform_scalar():
    """Test Z_transform function with scalar input"""
    print("\nTesting Z_transform (scalar)...")
    
    # Test basic computation: Z = sample_index * (delta_sample / c_lattice)
    sample_idx = 100
    delta = 0.01
    delta_max = 1.0
    
    z_value = Z_transform(sample_idx, delta, delta_max)
    expected = sample_idx * (delta / PHI)
    
    assert abs(z_value - expected) < 1e-10, "Z_transform matches formula"
    assert z_value > 0, "Z value should be positive"
    
    # Test with custom c_lattice
    c_custom = 2.0
    z_custom = Z_transform(sample_idx, delta, delta_max, c_lattice=c_custom)
    expected_custom = sample_idx * (delta / c_custom)
    assert abs(z_custom - expected_custom) < 1e-10, "Z_transform works with custom c"
    
    print("  ✓ Z_transform (scalar) works correctly")


def test_Z_transform_array():
    """Test Z_transform function with array input"""
    print("\nTesting Z_transform (array)...")
    
    indices = np.array([10, 20, 30, 40])
    delta = 0.01
    delta_max = 1.0
    
    z_array = Z_transform(indices, delta, delta_max)
    
    assert len(z_array) == len(indices), "Output array matches input length"
    assert np.all(z_array > 0), "All Z values are positive"
    
    # Check monotonicity (should increase with index)
    assert np.all(np.diff(z_array) > 0), "Z values increase with index"
    
    # Check individual values
    for i, idx in enumerate(indices):
        expected = Z_transform(int(idx), delta, delta_max)
        assert abs(z_array[i] - expected) < 1e-10, f"Z_transform({idx}) matches"
    
    print("  ✓ Z_transform (array) works correctly")


def test_validate_precision():
    """Test precision validation function"""
    print("\nTesting validate_precision...")
    
    # Test valid values
    assert validate_precision(1.0), "1.0 is valid"
    assert validate_precision(1.0 + 1e-13), "Value within precision is valid"
    assert validate_precision(PHI), "Golden ratio is valid"
    
    # Test invalid values
    assert not validate_precision(float('nan')), "NaN is invalid"
    assert not validate_precision(float('inf')), "Inf is invalid"
    
    print("  ✓ validate_precision works correctly")


def test_integration_bias_adaptive():
    """Integration test: Combining kappa and theta_prime for bias-adaptive sampling"""
    print("\nTesting integration (bias-adaptive sampling)...")
    
    # Simulate sample indices
    n_samples = 100
    sample_indices = np.arange(1, n_samples + 1)
    
    # Compute Z-framework features
    kappa_vals = kappa(sample_indices)
    theta_vals = theta_prime(sample_indices, k=0.3)
    
    # Compute bias weights: sample_weight = 1 / (1 + κ_bias · Δ_discrepancy)
    # For testing, use simplified form
    delta_disc = 0.01
    weights = 1.0 / (1.0 + kappa_vals * delta_disc)
    
    assert len(weights) == n_samples, "Weights computed for all samples"
    assert np.all(weights > 0), "All weights are positive"
    assert np.all(weights <= 1), "Weights are bounded by 1"
    
    # Apply theta bias
    biased_samples = sample_indices * (1.0 + 0.1 * np.sin(theta_vals))
    assert len(biased_samples) == n_samples, "Biased samples computed"
    
    print("  ✓ Integration test passed")


def test_golden_ratio_properties():
    """Test golden ratio φ properties"""
    print("\nTesting golden ratio properties...")
    
    # φ = (1 + √5) / 2 ≈ 1.618
    assert abs(PHI - 1.618033988749895) < 1e-10, "PHI has correct value"
    
    # φ² = φ + 1 (golden ratio property)
    assert abs(PHI**2 - (PHI + 1)) < 1e-10, "Golden ratio property φ² = φ + 1"
    
    # 1/φ = φ - 1
    assert abs(1/PHI - (PHI - 1)) < 1e-10, "Golden ratio property 1/φ = φ - 1"
    
    print("  ✓ Golden ratio properties verified")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Z-Framework Test Suite")
    print("=" * 60)
    
    test_count_divisors()
    test_kappa_scalar()
    test_kappa_array()
    test_theta_prime_scalar()
    test_theta_prime_array()
    test_Z_transform_scalar()
    test_Z_transform_array()
    test_validate_precision()
    test_golden_ratio_properties()
    test_integration_bias_adaptive()
    
    print("=" * 60)
    print("All Z-framework tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
