#!/usr/bin/env python3
"""
Test Suite for Auto-Scaling subgroup_order Feature
Tests the auto-derivation of subgroup_order based on geometric parameters
"""

import sys
import os
import warnings
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rank1_lattice import (
    _derive_subgroup_order,
    Rank1LatticeConfig,
    generate_rank1_lattice
)
from qmc_engines import QMCConfig, make_engine


def test_derive_subgroup_order_formula():
    """Test the auto-scaling formula with various inputs"""
    print("Testing _derive_subgroup_order formula...")
    
    # Test case from issue: n=144, dim=2, cone_height=1.2, spiral_depth=3
    # Expected: m = floor(1.8 * sqrt(144/2) * 1.2 * (1 + 3/4))
    #         = floor(1.8 * 8.485 * 1.2 * 1.75)
    #         = floor(32.08) = 32
    m1 = _derive_subgroup_order(n=144, dim=2, cone_height=1.2, spiral_depth=3)
    assert m1 == 32, f"Expected m=32 for n=144, dim=2, got {m1}"
    print(f"  ✓ n=144, dim=2, cone_height=1.2, spiral_depth=3 -> m={m1}")
    
    # Test large n case
    m2 = _derive_subgroup_order(n=10000, dim=3, cone_height=1.2, spiral_depth=3)
    assert m2 >= 90, f"Expected m>=90 for n=10000, dim=3, got {m2}"
    print(f"  ✓ n=10000, dim=3 -> m={m2} (>= 90)")
    
    # Test small n case
    m3 = _derive_subgroup_order(n=100, dim=1, cone_height=1.2, spiral_depth=3)
    assert m3 >= 10, f"Expected m>=10 for n=100, dim=1, got {m3}"
    print(f"  ✓ n=100, dim=1 -> m={m3} (>= 10)")
    
    # Test bounds enforcement: m should be at least 4
    m4 = _derive_subgroup_order(n=10, dim=10, cone_height=0.1, spiral_depth=0)
    assert m4 >= 4, f"Expected m>=4 for very small inputs, got {m4}"
    print(f"  ✓ Small inputs enforce minimum m=4: m={m4}")
    
    # Test bounds enforcement: m should not exceed n
    m5 = _derive_subgroup_order(n=50, dim=1, cone_height=10.0, spiral_depth=20)
    assert m5 <= 50, f"Expected m<=50 (not exceeding n), got {m5}"
    print(f"  ✓ Large parameters enforce maximum m<=n: m={m5}")
    
    print("  ✓ All formula tests passed")


def test_environment_override():
    """Test FORCE_SUBGROUP_ORDER environment variable override"""
    print("\nTesting FORCE_SUBGROUP_ORDER environment variable...")
    
    # Set environment variable
    os.environ['FORCE_SUBGROUP_ORDER'] = '99'
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = _derive_subgroup_order(n=144, dim=2, cone_height=1.2, spiral_depth=3)
        
        assert m == 99, f"Expected m=99 from environment override, got {m}"
        assert len(w) > 0, "Expected warning about forced value"
        assert "FORCE_SUBGROUP_ORDER" in str(w[0].message)
        print(f"  ✓ Environment override works: m={m}")
        print(f"  ✓ Warning issued: {w[0].message}")
    
    # Clean up
    del os.environ['FORCE_SUBGROUP_ORDER']
    print("  ✓ Environment override test passed")


def test_auto_derivation_no_manual_setting():
    """Test that subgroup_order is auto-derived when not manually set"""
    print("\nTesting auto-derivation (no manual subgroup_order)...")
    
    # Test with cyclic generator
    cfg1 = Rank1LatticeConfig(
        n=144,
        d=2,
        generator_type='cyclic',
        cone_height=1.2,
        spiral_depth=3
    )
    points1 = generate_rank1_lattice(cfg1)
    assert points1.shape == (144, 2), f"Expected shape (144, 2), got {points1.shape}"
    print(f"  ✓ Cyclic generator with auto-derivation: shape={points1.shape}")
    
    # Test with elliptic_cyclic generator
    cfg2 = Rank1LatticeConfig(
        n=144,
        d=2,
        generator_type='elliptic_cyclic',
        cone_height=1.2,
        spiral_depth=3
    )
    points2 = generate_rank1_lattice(cfg2)
    assert points2.shape == (144, 2), f"Expected shape (144, 2), got {points2.shape}"
    print(f"  ✓ Elliptic cyclic with auto-derivation: shape={points2.shape}")
    
    print("  ✓ Auto-derivation tests passed")


def test_deprecation_warning_manual_setting():
    """Test that deprecation warning is issued when subgroup_order is manually set"""
    print("\nTesting deprecation warning for manual subgroup_order...")
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        cfg = Rank1LatticeConfig(
            n=144,
            d=2,
            generator_type='cyclic',
            subgroup_order=12
        )
        points = generate_rank1_lattice(cfg)
        
        assert len(w) > 0, "Expected deprecation warning"
        assert issubclass(w[0].category, DeprecationWarning), "Expected DeprecationWarning"
        assert "deprecated" in str(w[0].message).lower()
        print(f"  ✓ Deprecation warning issued")
        print(f"  ✓ Warning message: {w[0].message}")
    
    print("  ✓ Deprecation warning test passed")


def test_qmc_engine_integration():
    """Test that auto-scaling works through QMC engine interface"""
    print("\nTesting QMC engine integration...")
    
    # Test with rank1_lattice engine
    cfg1 = QMCConfig(
        n=144,
        dim=2,
        engine='rank1_lattice',
        lattice_generator='cyclic',
        cone_height=1.2,
        spiral_depth=3
    )
    engine1 = make_engine(cfg1)
    points1 = engine1.random()
    assert points1.shape == (144, 2), f"Expected shape (144, 2), got {points1.shape}"
    print(f"  ✓ rank1_lattice engine: shape={points1.shape}")
    
    # Test with elliptic_cyclic engine
    cfg2 = QMCConfig(
        n=144,
        dim=2,
        engine='elliptic_cyclic',
        cone_height=2.0,
        spiral_depth=5
    )
    engine2 = make_engine(cfg2)
    points2 = engine2.random()
    assert points2.shape == (144, 2), f"Expected shape (144, 2), got {points2.shape}"
    print(f"  ✓ elliptic_cyclic engine: shape={points2.shape}")
    
    print("  ✓ QMC engine integration tests passed")


def test_parameter_effects():
    """Test that changing geometric parameters affects the derived subgroup_order"""
    print("\nTesting parameter effects on derived subgroup_order...")
    
    base_n, base_dim = 1000, 2
    base_cone, base_spiral = 1.2, 3
    
    # Baseline
    m_base = _derive_subgroup_order(base_n, base_dim, base_cone, base_spiral)
    print(f"  Baseline: n={base_n}, dim={base_dim}, cone_height={base_cone}, spiral_depth={base_spiral} -> m={m_base}")
    
    # Increase cone_height should increase m
    m_higher_cone = _derive_subgroup_order(base_n, base_dim, base_cone * 2, base_spiral)
    assert m_higher_cone > m_base, "Higher cone_height should increase m"
    print(f"  ✓ Higher cone_height ({base_cone * 2}) -> m={m_higher_cone} (increased)")
    
    # Increase spiral_depth should increase m
    m_higher_spiral = _derive_subgroup_order(base_n, base_dim, base_cone, base_spiral * 2)
    assert m_higher_spiral > m_base, "Higher spiral_depth should increase m"
    print(f"  ✓ Higher spiral_depth ({base_spiral * 2}) -> m={m_higher_spiral} (increased)")
    
    # Increase dim should decrease m (for same n)
    m_higher_dim = _derive_subgroup_order(base_n, base_dim * 2, base_cone, base_spiral)
    assert m_higher_dim < m_base, "Higher dim should decrease m"
    print(f"  ✓ Higher dim ({base_dim * 2}) -> m={m_higher_dim} (decreased)")
    
    # Increase n should increase m
    m_higher_n = _derive_subgroup_order(base_n * 4, base_dim, base_cone, base_spiral)
    assert m_higher_n > m_base, "Higher n should increase m"
    print(f"  ✓ Higher n ({base_n * 4}) -> m={m_higher_n} (increased)")
    
    print("  ✓ Parameter effects tests passed")


def test_default_parameters():
    """Test that default cone_height and spiral_depth values are used correctly"""
    print("\nTesting default parameter values...")
    
    # Create config without specifying geometric parameters
    cfg = Rank1LatticeConfig(n=144, d=2, generator_type='cyclic')
    
    # Check defaults
    assert cfg.cone_height == 1.2, f"Expected default cone_height=1.2, got {cfg.cone_height}"
    assert cfg.spiral_depth == 3, f"Expected default spiral_depth=3, got {cfg.spiral_depth}"
    print(f"  ✓ Default cone_height={cfg.cone_height}")
    print(f"  ✓ Default spiral_depth={cfg.spiral_depth}")
    
    # Generate points to ensure it works
    points = generate_rank1_lattice(cfg)
    assert points.shape == (144, 2)
    print(f"  ✓ Points generated with defaults: shape={points.shape}")
    
    print("  ✓ Default parameters test passed")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Auto-Scaling subgroup_order Test Suite")
    print("=" * 70)
    
    try:
        test_derive_subgroup_order_formula()
        test_environment_override()
        test_auto_derivation_no_manual_setting()
        test_deprecation_warning_manual_setting()
        test_qmc_engine_integration()
        test_parameter_effects()
        test_default_parameters()
        
        print("\n" + "=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print("\nKey Validations:")
        print("- Auto-scaling formula works correctly")
        print("- Environment variable override functional")
        print("- Auto-derivation works without manual settings")
        print("- Deprecation warnings issued for manual settings")
        print("- QMC engine integration seamless")
        print("- Geometric parameters affect derived values correctly")
        print("- Default parameters work as expected")
        print("=" * 70)
        
        return 0
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
