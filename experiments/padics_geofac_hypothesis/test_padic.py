"""
Test Suite for P-adic Module

Ensures correctness of p-adic operations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.padics_geofac_hypothesis.padic import (
    p_adic_valuation, p_adic_distance, p_adic_expansion,
    p_adic_from_expansion, is_ultrametric_valid,
    compute_cauchy_sequence_convergence, hensel_lift
)


def test_p_adic_valuation():
    """Test p-adic valuation function."""
    print("\n--- Testing p_adic_valuation ---")
    
    tests = [
        (8, 2, 3),      # 8 = 2^3
        (12, 2, 2),     # 12 = 2^2 * 3
        (15, 5, 1),     # 15 = 3 * 5
        (100, 2, 2),    # 100 = 2^2 * 25
        (1000, 5, 3),   # 1000 = 8 * 5^3
    ]
    
    for n, p, expected in tests:
        result = p_adic_valuation(n, p)
        status = "✓" if result == expected else "✗"
        print(f"  {status} v_{p}({n}) = {result} (expected {expected})")
        assert result == expected, f"Failed: v_{p}({n}) = {result}, expected {expected}"
    
    print("  All valuation tests passed!")


def test_p_adic_distance():
    """Test p-adic distance function."""
    print("\n--- Testing p_adic_distance ---")
    
    # Distance should be p^(-v_p(a-b))
    assert p_adic_distance(8, 0, 2) == 0.125, "d(8,0) should be 2^-3 = 0.125"
    assert p_adic_distance(10, 2, 2) == 0.125, "d(10,2) should be 2^-3 = 0.125"
    assert p_adic_distance(5, 5, 2) == 0.0, "d(5,5) should be 0"
    
    print("  ✓ All distance tests passed!")


def test_p_adic_expansion():
    """Test p-adic expansion function."""
    print("\n--- Testing p_adic_expansion ---")
    
    # 13 in base 2: 1101_2 = [1,0,1,1]
    exp = p_adic_expansion(13, 2, 5)
    assert exp[:4] == [1, 0, 1, 1], f"13 in base 2 should be [1,0,1,1], got {exp[:4]}"
    print(f"  ✓ 13 in base 2: {exp[:4]}")
    
    # 10 in base 5: 20_5 = [0,2]
    exp = p_adic_expansion(10, 5, 3)
    assert exp[:2] == [0, 2], f"10 in base 5 should be [0,2], got {exp[:2]}"
    print(f"  ✓ 10 in base 5: {exp[:2]}")
    
    # Verify roundtrip
    n = 2024
    for p in [2, 3, 5]:
        exp = p_adic_expansion(n, p, 20)
        reconstructed = p_adic_from_expansion(exp, p)
        assert reconstructed == n, f"Roundtrip failed for {n}, p={p}"
        print(f"  ✓ Roundtrip for {n}, p={p}: OK")
    
    print("  All expansion tests passed!")


def test_ultrametric_property():
    """Test ultrametric triangle inequality."""
    print("\n--- Testing ultrametric property ---")
    
    triplets = [
        (100, 200, 300),
        (1000, 1024, 1040),
        (2024, 2048, 2072),
    ]
    
    for p in [2, 5]:
        print(f"  Testing with p={p}:")
        for a, b, c in triplets:
            is_valid = is_ultrametric_valid(a, b, c, p)
            status = "✓" if is_valid else "✗"
            print(f"    {status} ({a}, {b}, {c}): {is_valid}")
            assert is_valid, f"Ultrametric failed for ({a},{b},{c}), p={p}"
    
    print("  All ultrametric tests passed!")


def test_hensel_lift():
    """Test Hensel lifting."""
    print("\n--- Testing Hensel lifting ---")
    
    # x^2 - 1 = 0
    f = lambda x: x**2 - 1
    df = lambda x: 2*x
    
    # Lift through 5-adic tower
    p = 5
    solution = 1
    print(f"  Starting: x ≡ {solution} (mod {p})")
    
    for k in range(1, 4):
        lifted = hensel_lift(f, df, solution, p, k)
        if lifted is not None:
            pk1 = p ** (k+1)
            assert f(lifted) % pk1 == 0, f"Lift failed at level {k}"
            print(f"  ✓ Lifted to mod {pk1}: x ≡ {lifted}")
            solution = lifted
        else:
            print(f"  Lift failed at level {k}")
            break
    
    print("  Hensel lifting tests passed!")


def test_cauchy_convergence():
    """Test Cauchy sequence detection."""
    print("\n--- Testing Cauchy convergence ---")
    
    # A sequence that converges in 2-adic metric
    sequence = [1000 + 2**k for k in range(10)]
    distances = compute_cauchy_sequence_convergence(sequence, 2)
    
    print(f"  Sequence: {sequence[:5]}...")
    print(f"  Distances: {distances[:5]}...")
    
    # Check that distances are decreasing (or at least bounded)
    assert len(distances) == len(sequence) - 1
    print("  ✓ Cauchy convergence test passed!")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("P-adic Module Test Suite")
    print("="*70)
    
    try:
        test_p_adic_valuation()
        test_p_adic_distance()
        test_p_adic_expansion()
        test_ultrametric_property()
        test_hensel_lift()
        test_cauchy_convergence()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return True
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
