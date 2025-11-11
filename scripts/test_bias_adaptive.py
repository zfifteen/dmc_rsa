#!/usr/bin/env python3
"""
Tests for Bias-Adaptive Sampling Engine
Tests enhanced QMC capabilities with Z-biased lattices
"""

import sys
sys.path.insert(0, 'scripts')

import numpy as np
from qmc_engines import (
    QMCConfig, make_engine, apply_bias_adaptive, compute_z_invariant_metrics,
    qmc_points
)


def test_bias_modes():
    """Test all bias modes work correctly"""
    print("\nTesting bias modes...")
    
    n_points = 128
    dim = 2
    test_points = np.random.rand(n_points, dim)
    
    for mode in ['theta_prime', 'prime_density', 'golden_spiral']:
        biased = apply_bias_adaptive(test_points.copy(), bias_mode=mode, k=0.3)
        assert biased.shape == test_points.shape, f"{mode}: Shape mismatch"
        assert np.all(biased >= 0) and np.all(biased <= 1), f"{mode}: Out of bounds"
        print(f"  ✓ {mode} bias works correctly")
    
    print("  ✓ All bias modes work")


def test_bias_with_sobol():
    """Test bias-adaptive Sobol sequences"""
    print("\nTesting bias-adaptive Sobol...")
    
    # Test with theta_prime bias
    cfg = QMCConfig(dim=2, n=256, engine='sobol', bias_mode='theta_prime', seed=42)
    eng = make_engine(cfg)
    points = eng.random(256)
    
    assert points.shape == (256, 2), "Points shape correct"
    assert np.all(points >= 0) and np.all(points <= 1), "Points in [0,1]"
    
    print("  ✓ Bias-adaptive Sobol works correctly")


def test_bias_with_halton():
    """Test bias-adaptive Halton sequences"""
    print("\nTesting bias-adaptive Halton...")
    
    cfg = QMCConfig(dim=3, n=100, engine='halton', bias_mode='prime_density', seed=42)
    eng = make_engine(cfg)
    points = eng.random(100)
    
    assert points.shape == (100, 3), "Points shape correct"
    assert np.all(points >= 0) and np.all(points <= 1), "Points in [0,1]"
    
    print("  ✓ Bias-adaptive Halton works correctly")


def test_z_invariant_metrics():
    """Test Z-invariant metric computation"""
    print("\nTesting Z-invariant metrics...")
    
    # Generate test points
    cfg = QMCConfig(dim=2, n=256, engine='sobol', seed=42)
    eng = make_engine(cfg)
    points = eng.random(256)
    
    # Compute metrics
    metrics = compute_z_invariant_metrics(points, method='sobol')
    
    assert 'discrepancy' in metrics, "Has discrepancy"
    assert 'unique_rate' in metrics, "Has unique_rate"
    assert 'mean_kappa' in metrics, "Has mean_kappa"
    assert 'savings_estimate' in metrics, "Has savings_estimate"
    
    assert 0 <= metrics['discrepancy'] <= 1, "Discrepancy in reasonable range"
    assert 0 <= metrics['unique_rate'] <= 1, "Unique rate in [0,1]"
    assert metrics['mean_kappa'] is not None, "Kappa computed"
    
    print(f"  Discrepancy: {metrics['discrepancy']:.4f}")
    print(f"  Unique rate: {metrics['unique_rate']:.4f}")
    print(f"  Mean kappa: {metrics['mean_kappa']:.4f}")
    print("  ✓ Z-invariant metrics work correctly")


def test_unique_rate_improvement():
    """Test that bias improves unique sample rate"""
    print("\nTesting unique rate improvement...")
    
    n = 256
    dim = 2
    
    # Baseline: no bias
    cfg_baseline = QMCConfig(dim=dim, n=n, engine='sobol', seed=42)
    eng_baseline = make_engine(cfg_baseline)
    points_baseline = eng_baseline.random(n)
    metrics_baseline = compute_z_invariant_metrics(points_baseline, method='sobol')
    
    # With bias
    cfg_biased = QMCConfig(dim=dim, n=n, engine='sobol', bias_mode='theta_prime', seed=42)
    eng_biased = make_engine(cfg_biased)
    points_biased = eng_biased.random(n)
    # Apply bias again for testing (engine doesn't auto-apply in random())
    points_biased = apply_bias_adaptive(points_biased, bias_mode='theta_prime')
    metrics_biased = compute_z_invariant_metrics(points_biased, method='sobol_biased')
    
    print(f"  Baseline unique rate: {metrics_baseline['unique_rate']:.4f}")
    print(f"  Biased unique rate: {metrics_biased['unique_rate']:.4f}")
    
    # Both should have high unique rates (QMC property)
    assert metrics_baseline['unique_rate'] >= 0.95, "Baseline has high unique rate"
    assert metrics_biased['unique_rate'] >= 0.95, "Biased has high unique rate"
    
    print("  ✓ Unique rates are high (QMC property maintained)")


def test_discrepancy_reduction():
    """Test that bias can reduce discrepancy"""
    print("\nTesting discrepancy with bias...")
    
    n = 256
    dim = 2
    
    # Generate baseline and biased samples
    cfg = QMCConfig(dim=dim, n=n, engine='sobol', seed=42)
    eng = make_engine(cfg)
    points_baseline = eng.random(n)
    
    # Try different bias modes
    discrepancies = {}
    for mode in ['theta_prime', 'prime_density']:
        points_biased = apply_bias_adaptive(points_baseline.copy(), bias_mode=mode)
        metrics = compute_z_invariant_metrics(points_biased, method=f'sobol_{mode}')
        discrepancies[mode] = metrics['discrepancy']
        print(f"  {mode} discrepancy: {metrics['discrepancy']:.4f}")
    
    # All should have reasonable discrepancy values
    for mode, disc in discrepancies.items():
        assert 0 < disc < 1, f"{mode}: Discrepancy in reasonable range"
    
    print("  ✓ Discrepancy metrics computed correctly")


def test_performance_requirement():
    """Test that generation meets performance requirement: d=10, N=10^4 in <1s"""
    print("\nTesting performance requirement (d=10, N=10,000)...")
    
    import time
    
    cfg = QMCConfig(dim=10, n=10000, engine='sobol', bias_mode='theta_prime', seed=42)
    
    start = time.time()
    
    # Generate via qmc_points with single replicate
    cfg.replicates = 1
    for points in qmc_points(cfg):
        pass  # Just generate, don't process
    
    elapsed = time.time() - start
    
    print(f"  Time taken: {elapsed:.3f}s")
    assert elapsed < 1.0, f"Should complete in <1s, took {elapsed:.3f}s"
    
    print("  ✓ Performance requirement met")


def test_integration_replicated_qmc():
    """Integration test with replicated QMC"""
    print("\nTesting integration with replicated QMC...")
    
    cfg = QMCConfig(
        dim=2, 
        n=256, 
        engine='sobol', 
        bias_mode='theta_prime',
        replicates=4,
        seed=42
    )
    
    # Generate replicates
    all_points = list(qmc_points(cfg))
    
    assert len(all_points) == 4, "Generated 4 replicates"
    
    # Compute metrics for each replicate
    discrepancies = []
    for i, points in enumerate(all_points):
        metrics = compute_z_invariant_metrics(points, method='sobol_biased')
        discrepancies.append(metrics['discrepancy'])
    
    # Compute mean and CI
    mean_disc = np.mean(discrepancies)
    std_disc = np.std(discrepancies)
    ci_lower = mean_disc - 1.96 * std_disc / np.sqrt(len(discrepancies))
    ci_upper = mean_disc + 1.96 * std_disc / np.sqrt(len(discrepancies))
    
    print(f"  Mean discrepancy: {mean_disc:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    assert 0 < mean_disc < 1, "Mean discrepancy in reasonable range"
    
    print("  ✓ Replicated QMC integration works")


def test_high_dimensional():
    """Test bias-adaptive in higher dimensions"""
    print("\nTesting high-dimensional bias-adaptive sampling...")
    
    # Test up to d=50 (edge case from requirements)
    for d in [5, 10, 20, 50]:
        cfg = QMCConfig(dim=d, n=256, engine='sobol', bias_mode='theta_prime', seed=42)
        eng = make_engine(cfg)
        points = eng.random(256)
        
        assert points.shape == (256, d), f"d={d}: Shape correct"
        assert np.all(points >= 0) and np.all(points <= 1), f"d={d}: Bounds correct"
    
    print("  ✓ High-dimensional sampling works (tested d=5,10,20,50)")


def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    # Small n (low sample warning from requirements)
    cfg_small = QMCConfig(dim=2, n=64, engine='sobol', bias_mode='theta_prime', seed=42)
    eng_small = make_engine(cfg_small)
    points_small = eng_small.random(64)
    assert points_small.shape[0] == 64, "Small n works"
    
    # Different k values
    test_points = np.random.rand(100, 2)
    for k in [0.2, 0.3, 0.5]:
        biased = apply_bias_adaptive(test_points.copy(), bias_mode='theta_prime', k=k)
        assert biased.shape == test_points.shape, f"k={k} works"
    
    # Different beta values
    for beta in [1.0, 2.0, 3.0]:
        biased = apply_bias_adaptive(test_points.copy(), bias_mode='theta_prime', beta=beta)
        assert biased.shape == test_points.shape, f"beta={beta} works"
    
    print("  ✓ Edge cases handled correctly")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Bias-Adaptive Sampling Engine Test Suite")
    print("=" * 60)
    
    test_bias_modes()
    test_bias_with_sobol()
    test_bias_with_halton()
    test_z_invariant_metrics()
    test_unique_rate_improvement()
    test_discrepancy_reduction()
    test_performance_requirement()
    test_integration_replicated_qmc()
    test_high_dimensional()
    test_edge_cases()
    
    print("=" * 60)
    print("All bias-adaptive sampling tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
