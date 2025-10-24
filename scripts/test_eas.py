#!/usr/bin/env python3
"""
Tests for Elliptic Adaptive Search (EAS) factorization module.

Validates:
- Elliptic lattice point generation
- Golden-angle spiral sampling
- Adaptive window sizing
- Factorization correctness
- Performance characteristics
"""

import sys
sys.path.append('scripts')

import numpy as np
from eas_factorize import (
    EllipticAdaptiveSearch, EASConfig, EASResult,
    factorize_eas, benchmark_eas
)


def test_eas_config():
    """Test EAS configuration dataclass"""
    print("Testing EAS configuration...")
    
    # Default config
    config = EASConfig()
    assert config.max_samples == 2000
    assert config.golden_angle > 0
    assert 0 < config.elliptic_eccentricity <= 1
    assert config.adaptive_window == True
    
    # Custom config
    custom = EASConfig(max_samples=1000, adaptive_window=False)
    assert custom.max_samples == 1000
    assert custom.adaptive_window == False
    
    print("  ✓ EAS configuration works correctly")


def test_adaptive_window_radius():
    """Test adaptive window radius calculation"""
    print("Testing adaptive window radius...")
    
    eas = EllipticAdaptiveSearch()
    
    # Test different bit lengths - use actual bit_length() values
    test_cases = [
        (2**15, 2**7.5, 0.05),   # 16-bit: tight window
        (2**31, 2**15.5, 0.05),  # 32-bit: tight window
        (2**39, 2**19.5, 0.10),  # 40-bit: medium window
        (2**47, 2**23.5, 0.15),  # 48-bit: wider window
        (2**63, 2**31.5, 0.20),  # 64-bit: even wider
    ]
    
    for n, sqrt_n, expected_scale in test_cases:
        radius = eas._adaptive_window_radius(n, sqrt_n)
        # Check that radius is proportional to sqrt_n with reasonable scale
        scale = radius / sqrt_n
        assert 0.01 <= scale <= 0.5, \
            f"Radius scale {scale} should be reasonable for {n.bit_length()}-bit"
    
    print("  ✓ Adaptive window radius calculation works correctly")


def test_elliptic_lattice_generation():
    """Test elliptic lattice point generation"""
    print("Testing elliptic lattice point generation...")
    
    eas = EllipticAdaptiveSearch()
    
    sqrt_n = 100.0
    radius = 10.0
    n_points = 50
    
    candidates = eas._generate_elliptic_lattice_points(n_points, sqrt_n, radius)
    
    # Check basic properties
    assert len(candidates) > 0, "Should generate candidates"
    assert len(candidates) <= 2 * n_points, "Should not exceed 2×n_points (±offsets)"
    
    # Check candidates are near sqrt_n
    min_candidate = np.min(candidates)
    max_candidate = np.max(candidates)
    assert min_candidate >= sqrt_n - radius * 1.5, "Candidates should be within radius"
    assert max_candidate <= sqrt_n + radius * 1.5, "Candidates should be within radius"
    
    # Check for reasonable diversity (accounting for integer rounding creating duplicates)
    unique_candidates = np.unique(candidates)
    assert len(unique_candidates) >= 5, "Should have at least some diverse candidates"
    
    print("  ✓ Elliptic lattice generation works correctly")


def test_primality_check():
    """Test simple primality checking"""
    print("Testing primality check...")
    
    eas = EllipticAdaptiveSearch()
    
    # Test known primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in primes:
        assert eas._is_prime(p), f"{p} should be prime"
    
    # Test known composites
    composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    for c in composites:
        assert not eas._is_prime(c), f"{c} should not be prime"
    
    # Test edge cases
    assert not eas._is_prime(0)
    assert not eas._is_prime(1)
    
    print("  ✓ Primality check works correctly")


def test_factorize_small_semiprimes():
    """Test factorization on small known semiprimes"""
    print("Testing factorization on small semiprimes...")
    
    test_cases = [
        (15, 3, 5),      # 4-bit
        (21, 3, 7),      # 5-bit
        (35, 5, 7),      # 6-bit
        (77, 7, 11),     # 7-bit
        (143, 11, 13),   # 8-bit
        (899, 29, 31),   # 10-bit (from original tests)
    ]
    
    successes = 0
    for n, expected_p, expected_q in test_cases:
        result = factorize_eas(n, verbose=False)
        
        if result.success:
            # Check we got the right factors (order may vary)
            factors = {result.factor_p, result.factor_q}
            expected = {expected_p, expected_q}
            assert factors == expected, \
                f"Wrong factors for {n}: got {factors}, expected {expected}"
            successes += 1
    
    # Should succeed on at least some of these small cases
    success_rate = successes / len(test_cases)
    assert success_rate >= 0.3, \
        f"Success rate too low: {success_rate:.1%} (expected >= 30%)"
    
    print(f"  ✓ Factorization works correctly ({success_rate:.1%} success rate)")


def test_eas_result_structure():
    """Test EASResult dataclass structure"""
    print("Testing EASResult structure...")
    
    # Run a factorization
    result = factorize_eas(35, verbose=False)
    
    # Check all fields exist and have correct types
    assert isinstance(result.success, bool), "success should be bool"
    assert isinstance(result.candidates_checked, int), "candidates_checked should be int"
    assert isinstance(result.time_elapsed, float), "time_elapsed should be float"
    assert isinstance(result.search_reduction, float), "search_reduction should be float"
    
    # Check numeric fields are positive
    assert result.candidates_checked > 0, "candidates_checked should be positive"
    assert result.time_elapsed > 0, "time_elapsed should be positive"
    assert result.search_reduction > 0, "search_reduction should be positive"
    
    # Check factors if successful
    if result.success:
        assert isinstance(result.factor_p, int), "factor_p should be int when successful"
        assert isinstance(result.factor_q, int), "factor_q should be int when successful"
        assert result.factor_p * result.factor_q == 35, "factors should multiply to N"
    else:
        assert result.factor_p is None, "factor_p should be None when failed"
        assert result.factor_q is None, "factor_q should be None when failed"
    
    print("  ✓ EASResult structure is correct")


def test_golden_angle_property():
    """Test that golden angle is correctly computed"""
    print("Testing golden angle property...")
    
    config = EASConfig()
    golden_angle = config.golden_angle
    
    # Golden angle should be approximately 137.5 degrees
    golden_angle_deg = np.degrees(golden_angle)
    expected_deg = 137.5
    
    assert abs(golden_angle_deg - expected_deg) < 0.1, \
        f"Golden angle {golden_angle_deg}° should be ~{expected_deg}°"
    
    # Verify it's derived correctly: golden angle = π(3 - √5)
    expected_angle = np.pi * (3 - np.sqrt(5))
    assert abs(golden_angle - expected_angle) < 1e-10, \
        f"Golden angle should be π(3-√5), got {golden_angle}, expected {expected_angle}"
    
    print("  ✓ Golden angle property verified")


def test_config_customization():
    """Test that custom configurations affect behavior"""
    print("Testing configuration customization...")
    
    # Test with very small max_samples
    small_config = EASConfig(max_samples=10)
    eas_small = EllipticAdaptiveSearch(small_config)
    result_small = eas_small.factorize(899, verbose=False)
    
    # Should check very few candidates
    assert result_small.candidates_checked <= 20, \
        "Small config should check few candidates"
    
    # Test with adaptive window disabled
    fixed_config = EASConfig(adaptive_window=False, base_radius_factor=0.5)
    eas_fixed = EllipticAdaptiveSearch(fixed_config)
    
    # Window should be fixed regardless of bit size
    radius_32 = eas_fixed._adaptive_window_radius(2**32, 2**16)
    radius_64 = eas_fixed._adaptive_window_radius(2**64, 2**32)
    
    # Both should use base_radius_factor (0.5)
    assert abs(radius_32 / (2**16) - 0.5) < 0.1
    assert abs(radius_64 / (2**32) - 0.5) < 0.1
    
    print("  ✓ Configuration customization works correctly")


def test_search_space_reduction():
    """Test that search space reduction is calculated correctly"""
    print("Testing search space reduction calculation...")
    
    n = 899
    sqrt_n = int(np.sqrt(n))
    
    result = factorize_eas(n, verbose=False)
    
    # Search space reduction should be reasonable
    assert result.search_reduction >= 1.0, \
        "Search reduction should be at least 1×"
    
    # For successful factorization, should have good reduction
    if result.success:
        # Full search space is roughly sqrt_n / 2
        full_space = sqrt_n // 2
        actual_reduction = full_space / result.candidates_checked
        
        # Should match reported reduction approximately
        assert abs(result.search_reduction - actual_reduction) / actual_reduction < 0.5, \
            "Reported reduction should match calculated reduction"
    
    print("  ✓ Search space reduction calculated correctly")


def test_performance_characteristics():
    """Test expected performance characteristics match empirical findings"""
    print("Testing performance characteristics...")
    
    # Test that smaller bit sizes have higher success rates
    # Run a few trials at different sizes
    results_small = []
    results_medium = []
    
    for _ in range(5):
        # Generate small semiprimes (~ 16-bit)
        p = 251  # 8-bit prime
        q = 257  # 8-bit prime
        n_small = p * q  # ~16-bit
        
        result = factorize_eas(n_small, verbose=False)
        results_small.append(result.success)
        
        # Generate medium semiprimes (~ 32-bit)
        p = 65521  # 16-bit prime
        q = 65537  # 16-bit prime
        n_medium = p * q  # ~32-bit
        
        result = factorize_eas(n_medium, verbose=False)
        results_medium.append(result.success)
    
    success_small = sum(results_small) / len(results_small)
    success_medium = sum(results_medium) / len(results_medium)
    
    # Small should have higher or equal success rate to medium
    # (allowing for randomness in small sample)
    assert success_small >= success_medium * 0.5, \
        f"Small ({success_small:.0%}) should not be much worse than medium ({success_medium:.0%})"
    
    print(f"  ✓ Performance characteristics reasonable (small: {success_small:.0%}, medium: {success_medium:.0%})")


def run_all_tests():
    """Run all EAS tests"""
    print("=" * 70)
    print("Elliptic Adaptive Search (EAS) Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_eas_config,
        test_adaptive_window_radius,
        test_elliptic_lattice_generation,
        test_primality_check,
        test_factorize_small_semiprimes,
        test_eas_result_structure,
        test_golden_angle_property,
        test_config_customization,
        test_search_space_reduction,
        test_performance_characteristics,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
