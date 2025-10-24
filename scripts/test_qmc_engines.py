#!/usr/bin/env python3
"""
Tests for qmc_engines module
"""

import sys
sys.path.append('scripts')

import numpy as np
from qmc_engines import (
    QMCConfig, make_engine, qmc_points, map_points_to_candidates,
    estimate_l2_discrepancy, stratification_balance
)

def test_qmc_config():
    """Test QMCConfig dataclass"""
    print("Testing QMCConfig...")
    
    # Default config
    cfg = QMCConfig(dim=2, n=100)
    assert cfg.dim == 2
    assert cfg.n == 100
    assert cfg.engine == "sobol"
    assert cfg.scramble == True
    assert cfg.replicates == 8
    
    # Custom config
    cfg2 = QMCConfig(dim=3, n=50, engine="halton", scramble=False, seed=42, replicates=16)
    assert cfg2.dim == 3
    assert cfg2.n == 50
    assert cfg2.engine == "halton"
    assert cfg2.scramble == False
    assert cfg2.seed == 42
    assert cfg2.replicates == 16
    
    print("  ✓ QMCConfig works correctly")

def test_make_engine():
    """Test engine creation"""
    print("Testing make_engine...")
    
    # Sobol engine
    cfg_sobol = QMCConfig(dim=2, n=100, engine="sobol", scramble=True, seed=42)
    engine = make_engine(cfg_sobol)
    assert engine is not None
    points = engine.random(10)
    assert points.shape == (10, 2)
    assert np.all(points >= 0) and np.all(points <= 1)
    
    # Halton engine
    cfg_halton = QMCConfig(dim=3, n=100, engine="halton", scramble=True, seed=42)
    engine = make_engine(cfg_halton)
    assert engine is not None
    points = engine.random(10)
    assert points.shape == (10, 3)
    assert np.all(points >= 0) and np.all(points <= 1)
    
    # Invalid engine
    try:
        cfg_invalid = QMCConfig(dim=2, n=100, engine="invalid")
        make_engine(cfg_invalid)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported engine" in str(e)
    
    print("  ✓ make_engine works correctly")

def test_qmc_points():
    """Test replicated QMC point generation"""
    print("Testing qmc_points...")
    
    cfg = QMCConfig(dim=2, n=50, engine="sobol", scramble=True, seed=42, replicates=4)
    
    replicates = list(qmc_points(cfg))
    
    # Check we get the right number of replicates
    assert len(replicates) == 4
    
    # Check each replicate has correct shape
    for points in replicates:
        assert points.shape == (50, 2)
        assert np.all(points >= 0) and np.all(points <= 1)
    
    # Check replicates are different (due to different seeds)
    assert not np.allclose(replicates[0], replicates[1])
    assert not np.allclose(replicates[0], replicates[2])
    
    print("  ✓ qmc_points generates correct replicates")

def test_map_points_to_candidates():
    """Test candidate mapping"""
    print("Testing map_points_to_candidates...")
    
    # Create simple test points
    N = 899  # 29 × 31
    window_radius = 10
    
    # Generate some QMC points
    cfg = QMCConfig(dim=2, n=20, engine="sobol", scramble=True, seed=42)
    eng = make_engine(cfg)
    X = eng.random(20)
    
    # Map to candidates
    candidates = map_points_to_candidates(X, N, window_radius)
    
    # Check candidates are in valid range
    assert np.all(candidates > 1)
    assert np.all(candidates < N)
    
    # Check candidates are odd
    assert np.all(candidates % 2 == 1)
    
    # Check candidates have correct residues (1, 3, 7, 9 mod 10)
    residues = candidates % 10
    valid_residues = {1, 3, 7, 9}
    for r in residues:
        assert r in valid_residues, f"Invalid residue: {r}"
    
    print(f"  ✓ Generated {len(candidates)} valid candidates from {len(X)} points")

def test_estimate_l2_discrepancy():
    """Test L2 discrepancy estimation"""
    print("Testing estimate_l2_discrepancy...")
    
    # Generate QMC and MC points
    cfg_qmc = QMCConfig(dim=2, n=100, engine="sobol", scramble=True, seed=42)
    eng = make_engine(cfg_qmc)
    qmc_pts = eng.random(100)
    
    np.random.seed(42)
    mc_pts = np.random.random((100, 2))
    
    # QMC should have lower discrepancy than MC (usually)
    disc_qmc = estimate_l2_discrepancy(qmc_pts)
    disc_mc = estimate_l2_discrepancy(mc_pts)
    
    assert disc_qmc > 0
    assert disc_mc > 0
    
    print(f"  L2 discrepancy: QMC={disc_qmc:.4f}, MC={disc_mc:.4f}")
    print("  ✓ L2 discrepancy estimation works")

def test_stratification_balance():
    """Test stratification balance metric"""
    print("Testing stratification_balance...")
    
    # Generate QMC points
    cfg = QMCConfig(dim=2, n=100, engine="sobol", scramble=True, seed=42)
    eng = make_engine(cfg)
    qmc_pts = eng.random(100)
    
    balance = stratification_balance(qmc_pts, n_bins=10)
    
    assert 0 <= balance <= 1, f"Balance should be in [0,1], got {balance}"
    
    # QMC should have good balance (typically > 0.8)
    print(f"  Stratification balance: {balance:.4f}")
    print("  ✓ Stratification balance metric works")

def test_sobol_power_of_two_validation():
    """Test Sobol power-of-two validation"""
    print("Testing Sobol power-of-two validation...")
    
    from qmc_engines import validate_sobol_sample_size
    
    # Test power of 2 - should pass without warning
    assert validate_sobol_sample_size(256) == 256
    assert validate_sobol_sample_size(128) == 128
    
    # Test non-power of 2 with auto_round - should round up
    assert validate_sobol_sample_size(200, auto_round=True) == 256
    assert validate_sobol_sample_size(100, auto_round=True) == 128
    
    # Test non-power of 2 without auto_round - should raise
    try:
        validate_sobol_sample_size(200, auto_round=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "power of 2" in str(e)
    
    # Test QMCConfig with auto_round_sobol
    cfg_auto = QMCConfig(dim=2, n=200, engine="sobol", auto_round_sobol=True)
    eng = make_engine(cfg_auto)  # Should warn but not fail
    assert eng is not None
    
    # Test QMCConfig without auto_round_sobol
    cfg_no_auto = QMCConfig(dim=2, n=200, engine="sobol", auto_round_sobol=False)
    try:
        make_engine(cfg_no_auto)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "power of 2" in str(e)
    
    print("  ✓ Sobol power-of-two validation works correctly")

def test_integration_with_rsa():
    """Integration test: generate candidates for RSA factorization"""
    print("Testing integration with RSA factorization...")
    
    N = 899  # 29 × 31
    window_radius = 10
    
    cfg = QMCConfig(dim=2, n=100, engine="sobol", scramble=True, seed=42, replicates=3)
    
    all_candidates = []
    for replicate_idx, X in enumerate(qmc_points(cfg)):
        candidates = map_points_to_candidates(X, N, window_radius)
        all_candidates.append(candidates)
        
        # Check for factors
        hits = [c for c in candidates if N % c == 0 and c > 1 and c < N]
        print(f"  Replicate {replicate_idx}: {len(candidates)} candidates, {len(hits)} hits")
        
        if hits:
            print(f"    Found factors: {hits}")
    
    # Check that replicates produce different candidate sets
    assert not np.array_equal(all_candidates[0], all_candidates[1])
    
    print("  ✓ Integration test passed")

def main():
    """Run all tests"""
    print("="*60)
    print("QMC Engines Test Suite")
    print("="*60)
    
    test_qmc_config()
    test_make_engine()
    test_qmc_points()
    test_map_points_to_candidates()
    test_estimate_l2_discrepancy()
    test_stratification_balance()
    test_sobol_power_of_two_validation()
    test_integration_with_rsa()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == "__main__":
    main()
