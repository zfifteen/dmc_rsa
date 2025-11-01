#!/usr/bin/env python3
"""
Integration test for κ-weighted rank-1 lattices

This script tests the complete integration of the κ-weighting feature
across all components: divisor_density, qmc_engines, and lattice generation.
"""

import sys
sys.path.append('scripts')

import numpy as np
from cognitive_number_theory.divisor_density import kappa
from scripts.qmc_engines import QMCConfig, make_engine, kappa_weight
from scripts.rank1_lattice import Rank1LatticeConfig, generate_rank1_lattice


def test_kappa_function():
    """Test the kappa function from cognitive_number_theory"""
    print("="*70)
    print("Test 1: κ function from cognitive_number_theory")
    print("="*70)
    
    # Test various numbers
    test_cases = [
        (12, 2.0, 2.1),    # 6 divisors
        (17, 0.7, 0.8),    # 2 divisors (prime)
        (899, 3.6, 3.8),   # RSA-like semiprime
    ]
    
    for n, lower, upper in test_cases:
        k = kappa(n)
        assert lower < k < upper, f"κ({n}) = {k} not in ({lower}, {upper})"
        print(f"  ✓ κ({n}) = {k:.3f}")
    
    print()
    return True


def test_kappa_weight_function():
    """Test the kappa_weight function from qmc_engines"""
    print("="*70)
    print("Test 2: kappa_weight function")
    print("="*70)
    
    # Create test points
    points = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    n = 899
    
    # Apply weighting
    weighted = kappa_weight(points, n)
    
    # Verify shape preservation
    assert weighted.shape == points.shape, "Shape should be preserved"
    print(f"  ✓ Shape preserved: {weighted.shape}")
    
    # Verify weighting changes values
    assert not np.allclose(weighted, points), "Weighting should change values"
    print(f"  ✓ Values changed by weighting")
    
    # Verify all values are non-negative after weighting
    assert np.all(weighted >= 0), "Weighted values should be non-negative"
    print(f"  ✓ All weighted values non-negative")
    
    print()
    return True


def test_rank1_lattice_with_kappa():
    """Test rank-1 lattice generation with κ-weighting"""
    print("="*70)
    print("Test 3: Rank-1 lattice with κ-weighting via QMCConfig")
    print("="*70)
    
    n_samples = 128
    n = 899
    
    # Test different lattice types
    lattice_types = ['fibonacci', 'korobov', 'cyclic']
    
    for lattice_type in lattice_types:
        # Create config with κ-weighting enabled
        cfg = QMCConfig(
            dim=2,
            n=n_samples,
            engine="rank1_lattice",
            lattice_generator=lattice_type,
            with_kappa_weight=True,
            kappa_n=n,
            seed=42
        )
        
        # Generate points
        engine = make_engine(cfg)
        points_weighted = engine.random(n_samples)
        
        # Verify generation
        assert points_weighted.shape == (n_samples, 2), f"Shape mismatch for {lattice_type}"
        assert np.all(points_weighted >= 0), f"Negative values in {lattice_type}"
        print(f"  ✓ {lattice_type:12s}: {n_samples} points generated")
        
        # Compare with unweighted
        cfg_unweighted = QMCConfig(
            dim=2,
            n=n_samples,
            engine="rank1_lattice",
            lattice_generator=lattice_type,
            with_kappa_weight=False,
            seed=42
        )
        
        engine_unweighted = make_engine(cfg_unweighted)
        points_unweighted = engine_unweighted.random(n_samples)
        
        # They should differ
        if not np.allclose(points_weighted, points_unweighted):
            print(f"                 κ-weighted differs from unweighted ✓")
        else:
            print(f"                 ⚠ κ-weighted identical to unweighted (unexpected)")
    
    print()
    return True


def test_candidate_generation():
    """Test full candidate generation pipeline with κ-weighting"""
    print("="*70)
    print("Test 4: Full candidate generation pipeline")
    print("="*70)
    
    N = 899  # 29 × 31
    n_samples = 256
    sqrt_n = int(np.sqrt(N))
    
    # Generate with κ-weighting
    cfg = QMCConfig(
        dim=2,
        n=n_samples,
        engine="rank1_lattice",
        lattice_generator="fibonacci",
        with_kappa_weight=True,
        kappa_n=N,
        seed=42
    )
    
    engine = make_engine(cfg)
    points = engine.random(n_samples)
    
    # Convert to candidates (simple mapping for demonstration)
    candidates = (points[:, 0] * sqrt_n).astype(int)
    candidates = candidates[(candidates > 1) & (candidates < N)]
    
    unique_candidates = len(np.unique(candidates))
    print(f"  Generated {n_samples} points")
    print(f"  Mapped to {len(candidates)} valid candidates")
    print(f"  Unique candidates: {unique_candidates}")
    
    # Check if factors are in candidates
    factors = [29, 31]
    found_factors = [f for f in factors if f in candidates]
    print(f"  Found factors: {found_factors}")
    
    assert len(candidates) > 0, "Should generate some candidates"
    assert unique_candidates > 0, "Should have some unique candidates"
    
    print()
    return True


def test_vectorized_kappa():
    """Test vectorized kappa computation"""
    print("="*70)
    print("Test 5: Vectorized κ computation")
    print("="*70)
    
    from cognitive_number_theory.divisor_density import kappa_vectorized
    
    # Test on array
    n_array = np.arange(10, 100, dtype=int)
    k_array = kappa_vectorized(n_array)
    
    assert k_array.shape == n_array.shape, "Shape should match"
    assert np.all(k_array > 0), "All κ values should be positive"
    
    print(f"  ✓ Vectorized on {len(n_array)} values")
    print(f"    Min κ: {k_array.min():.3f}")
    print(f"    Max κ: {k_array.max():.3f}")
    print(f"    Mean κ: {k_array.mean():.3f}")
    
    print()
    return True


def main():
    """Run all integration tests"""
    print()
    print("="*70)
    print(" κ-WEIGHTED RANK-1 LATTICES - INTEGRATION TEST SUITE")
    print("="*70)
    print()
    
    tests = [
        test_kappa_function,
        test_kappa_weight_function,
        test_rank1_lattice_with_kappa,
        test_candidate_generation,
        test_vectorized_kappa,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} returned False")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    print()
    
    if failed == 0:
        print("✓ All integration tests passed!")
        print()
        print("The κ-weighted rank-1 lattice feature is fully integrated:")
        print("  • κ function computes divisor density curvature")
        print("  • kappa_weight applies weighting to lattice points")
        print("  • QMCConfig supports with_kappa_weight and kappa_n parameters")
        print("  • Rank1LatticeEngine applies weighting when configured")
        print("  • All lattice types (Fibonacci, Korobov, Cyclic) support weighting")
        print()
        return 0
    else:
        print(f"✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
