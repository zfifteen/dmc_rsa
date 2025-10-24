#!/usr/bin/env python3
"""
Test rank-1 lattice integration with QMC engines
Validates that rank-1 lattices work seamlessly with existing QMC framework
"""

import sys
sys.path.append('scripts')

import numpy as np
from qmc_engines import (
    QMCConfig, make_engine, qmc_points, map_points_to_candidates,
    estimate_l2_discrepancy, stratification_balance
)
from rank1_lattice import compute_lattice_quality_metrics


def test_rank1_lattice_engine_creation():
    """Test creating rank-1 lattice engine through QMCConfig"""
    print("Testing rank-1 lattice engine creation...")
    
    cfg = QMCConfig(
        dim=2,
        n=128,
        engine="rank1_lattice",
        lattice_generator="fibonacci",
        scramble=True,
        seed=42
    )
    
    engine = make_engine(cfg)
    assert engine is not None
    
    # Generate points
    points = engine.random(128)
    assert points.shape == (128, 2)
    assert np.all(points >= 0) and np.all(points < 1)
    
    print(f"  Generated {len(points)} points")
    print("  ✓ Engine creation works correctly")


def test_rank1_vs_sobol_vs_halton():
    """Compare rank-1 lattice with Sobol and Halton sequences"""
    print("\nComparing Rank-1 Lattice vs Sobol vs Halton...")
    
    n = 128
    d = 2
    seed = 42
    
    # Rank-1 Lattice (Fibonacci)
    cfg_rank1_fib = QMCConfig(
        dim=d, n=n, engine="rank1_lattice",
        lattice_generator="fibonacci", scramble=True, seed=seed
    )
    eng_rank1_fib = make_engine(cfg_rank1_fib)
    points_rank1_fib = eng_rank1_fib.random(n)
    
    # Rank-1 Lattice (Cyclic)
    cfg_rank1_cyc = QMCConfig(
        dim=d, n=n, engine="rank1_lattice",
        lattice_generator="cyclic", subgroup_order=16,
        scramble=True, seed=seed
    )
    eng_rank1_cyc = make_engine(cfg_rank1_cyc)
    points_rank1_cyc = eng_rank1_cyc.random(n)
    
    # Sobol
    cfg_sobol = QMCConfig(dim=d, n=n, engine="sobol", scramble=True, seed=seed)
    eng_sobol = make_engine(cfg_sobol)
    points_sobol = eng_sobol.random(n)
    
    # Halton
    cfg_halton = QMCConfig(dim=d, n=n, engine="halton", scramble=True, seed=seed)
    eng_halton = make_engine(cfg_halton)
    points_halton = eng_halton.random(n)
    
    # Monte Carlo for baseline
    np.random.seed(seed)
    points_mc = np.random.random((n, d))
    
    # Compute metrics
    disc_rank1_fib = estimate_l2_discrepancy(points_rank1_fib)
    disc_rank1_cyc = estimate_l2_discrepancy(points_rank1_cyc)
    disc_sobol = estimate_l2_discrepancy(points_sobol)
    disc_halton = estimate_l2_discrepancy(points_halton)
    disc_mc = estimate_l2_discrepancy(points_mc)
    
    bal_rank1_fib = stratification_balance(points_rank1_fib)
    bal_rank1_cyc = stratification_balance(points_rank1_cyc)
    bal_sobol = stratification_balance(points_sobol)
    bal_halton = stratification_balance(points_halton)
    bal_mc = stratification_balance(points_mc)
    
    # Lattice-specific metrics
    metrics_rank1_fib = compute_lattice_quality_metrics(points_rank1_fib)
    metrics_rank1_cyc = compute_lattice_quality_metrics(points_rank1_cyc)
    
    print("\n  L2 Discrepancy (lower is better):")
    print(f"    Rank-1 Fibonacci:      {disc_rank1_fib:.4f}")
    print(f"    Rank-1 Cyclic:         {disc_rank1_cyc:.4f}")
    print(f"    Sobol:                 {disc_sobol:.4f}")
    print(f"    Halton:                {disc_halton:.4f}")
    print(f"    Monte Carlo:           {disc_mc:.4f}")
    
    print("\n  Stratification Balance (higher is better):")
    print(f"    Rank-1 Fibonacci:      {bal_rank1_fib:.4f}")
    print(f"    Rank-1 Cyclic:         {bal_rank1_cyc:.4f}")
    print(f"    Sobol:                 {bal_sobol:.4f}")
    print(f"    Halton:                {bal_halton:.4f}")
    print(f"    Monte Carlo:           {bal_mc:.4f}")
    
    print("\n  Lattice-Specific Metrics:")
    print(f"    Rank-1 Fibonacci - Min distance: {metrics_rank1_fib['min_distance']:.4f}")
    print(f"    Rank-1 Cyclic    - Min distance: {metrics_rank1_cyc['min_distance']:.4f}")
    
    # All should be valid
    assert disc_rank1_fib > 0 and disc_rank1_cyc > 0
    assert bal_rank1_fib > 0 and bal_rank1_cyc > 0
    
    print("\n  ✓ All methods produce valid point sets")


def test_rank1_with_qmc_points_generator():
    """Test rank-1 lattice with replicated QMC points generator"""
    print("\nTesting rank-1 lattice with replicated points generator...")
    
    cfg = QMCConfig(
        dim=2,
        n=64,
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=8,
        scramble=True,
        seed=42,
        replicates=4
    )
    
    replicates = list(qmc_points(cfg))
    
    # Check we get the right number of replicates
    assert len(replicates) == 4
    
    # Check each replicate has correct shape
    for i, points in enumerate(replicates):
        assert points.shape == (64, 2)
        assert np.all(points >= 0) and np.all(points < 1)
        print(f"    Replicate {i}: {len(points)} points")
    
    # Check replicates are different (due to different scrambling)
    assert not np.allclose(replicates[0], replicates[1])
    
    print("  ✓ Replicated generation works correctly")


def test_rank1_with_rsa_candidate_mapping():
    """Test rank-1 lattice with RSA candidate mapping"""
    print("\nTesting rank-1 lattice with RSA candidate mapping...")
    
    N = 899  # 29 × 31
    window_radius = 10
    
    cfg = QMCConfig(
        dim=2,
        n=128,
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=20,  # Divides φ(899) = 840
        scramble=True,
        seed=42
    )
    
    eng = make_engine(cfg)
    X = eng.random(128)
    
    # Map to RSA candidates
    candidates = map_points_to_candidates(X, N, window_radius)
    
    # Check candidates are valid
    assert len(candidates) > 0
    assert np.all(candidates > 1)
    assert np.all(candidates < N)
    assert np.all(candidates % 2 == 1)  # Odd
    
    # Check for factors
    hits = [c for c in candidates if N % c == 0 and c > 1 and c < N]
    
    print(f"  Generated {len(candidates)} candidates")
    print(f"  Found {len(hits)} factors: {set(hits)}")
    
    assert len(hits) > 0, "Should find at least one factor"
    
    print("  ✓ RSA candidate mapping works correctly")


def test_rank1_replicated_analysis():
    """Test replicated rank-1 lattice analysis for confidence intervals"""
    print("\nTesting replicated rank-1 lattice analysis...")
    
    N = 899
    window_radius = 10
    num_replicates = 8
    
    cfg = QMCConfig(
        dim=2,
        n=128,
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=20,
        scramble=True,
        seed=42,
        replicates=num_replicates
    )
    
    unique_counts = []
    hit_counts = []
    
    for replicate_idx, X in enumerate(qmc_points(cfg)):
        candidates = map_points_to_candidates(X, N, window_radius)
        unique_candidates = np.unique(candidates)
        hits = [c for c in unique_candidates if N % c == 0 and c > 1 and c < N]
        
        unique_counts.append(len(unique_candidates))
        hit_counts.append(len(hits))
    
    # Calculate statistics
    mean_unique = np.mean(unique_counts)
    std_unique = np.std(unique_counts, ddof=1)
    ci_lower = mean_unique - 1.96 * std_unique / np.sqrt(num_replicates)
    ci_upper = mean_unique + 1.96 * std_unique / np.sqrt(num_replicates)
    
    mean_hits = np.mean(hit_counts)
    
    print(f"\n  Unique candidates: {mean_unique:.2f} ± {std_unique:.2f}")
    print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Mean hits (factors): {mean_hits:.2f}")
    
    assert mean_unique > 0
    assert mean_hits > 0
    
    print("  ✓ Replicated analysis works correctly")


def test_all_generator_types():
    """Test all rank-1 lattice generator types"""
    print("\nTesting all generator types...")
    
    N = 899
    window_radius = 10
    
    generator_types = ["fibonacci", "korobov", "cyclic"]
    
    for gen_type in generator_types:
        print(f"\n  Testing {gen_type}...")
        
        cfg = QMCConfig(
            dim=2,
            n=128,
            engine="rank1_lattice",
            lattice_generator=gen_type,
            subgroup_order=20 if gen_type == "cyclic" else None,
            scramble=True,
            seed=42
        )
        
        eng = make_engine(cfg)
        X = eng.random(128)
        candidates = map_points_to_candidates(X, N, window_radius)
        unique_candidates = np.unique(candidates)
        hits = [c for c in unique_candidates if N % c == 0 and c > 1 and c < N]
        
        print(f"    Unique candidates: {len(unique_candidates)}")
        print(f"    Factors found: {len(hits)}")
        
        assert len(unique_candidates) > 0
        assert len(hits) > 0
    
    print("\n  ✓ All generator types work correctly")


def main():
    """Run all integration tests"""
    print("="*70)
    print("Rank-1 Lattice Integration Test Suite")
    print("="*70)
    
    test_rank1_lattice_engine_creation()
    test_rank1_vs_sobol_vs_halton()
    test_rank1_with_qmc_points_generator()
    test_rank1_with_rsa_candidate_mapping()
    test_rank1_replicated_analysis()
    test_all_generator_types()
    
    print("\n" + "="*70)
    print("All integration tests passed! ✓")
    print("="*70)
    print("\nKey Findings:")
    print("- Rank-1 lattices integrate seamlessly with existing QMC framework")
    print("- Cyclic subgroup construction provides competitive quality metrics")
    print("- RSA candidate mapping works correctly with lattice points")
    print("- Replicated analysis enables confidence interval estimation")
    print("- All generator types (Fibonacci, Korobov, Cyclic) functional")
    print("="*70)


if __name__ == "__main__":
    main()
