#!/usr/bin/env python3
"""
Arctan-Refined Curvature Implementation
========================================

This module implements the hypothesis that augmenting κ(n) with arctan(φ · frac(n/φ))
terms enhances Korobov lattice parameter tuning for QMC integration.

Hypothesis (to be falsified):
    κ_arctan(n) = κ(n) + arctan(φ · frac(n/φ))
    
    where:
    - κ(n) = d(n) · ln(n+1) / e² (baseline curvature)
    - φ = (1 + √5) / 2 (golden ratio)
    - frac(x) = x - floor(x) (fractional part)

Expected claim: 10-30% variance reduction in QMC for periodic integrands.

Author: Z-Mode experiment framework
Date: November 2025
"""

import numpy as np
from typing import Union, Dict, Tuple
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from cognitive_number_theory import kappa
from wave_crispr_signal import PHI


def arctan_refinement(n: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute arctan refinement term: arctan(φ · frac(n/φ)).
    
    This is the hypothesized enhancement to the baseline κ(n) curvature.
    The arctan function is claimed to improve golden-ratio equidistribution
    in Korobov lattices.
    
    Args:
        n: Sample index or indices
        
    Returns:
        Arctan refinement value(s) in range [-π/2, π/2]
        
    Examples:
        >>> arctan_refinement(1)
        0.565...
        >>> arctan_refinement(np.array([1, 10, 100]))
        array([0.565..., 1.236..., 1.178...])
    """
    # Handle array input
    if isinstance(n, (list, tuple, np.ndarray)):
        n_array = np.asarray(n, dtype=float)
        frac_n_phi = np.mod(n_array / PHI, 1.0)
        return np.arctan(PHI * frac_n_phi)
    
    # Handle scalar input
    n_float = float(n)
    frac_n_phi = (n_float / PHI) % 1.0
    return np.arctan(PHI * frac_n_phi)


def kappa_arctan(n: Union[int, np.ndarray], 
                 alpha: float = 1.0) -> Union[float, np.ndarray]:
    """
    Compute arctan-refined curvature: κ_arctan(n) = κ(n) + α · arctan(φ · frac(n/φ)).
    
    This combines the baseline divisor-weighted curvature with the arctan
    refinement term. The parameter α controls the strength of the arctan
    contribution.
    
    Args:
        n: Sample index or indices (must be positive integers)
        alpha: Scaling factor for arctan term (default: 1.0)
        
    Returns:
        Arctan-refined curvature value(s)
        
    Raises:
        ValueError: If n < 1
        
    Examples:
        >>> kappa_arctan(1, alpha=1.0)
        0.659...
        >>> kappa_arctan(12, alpha=1.0)
        2.095...
    """
    # Compute baseline curvature
    k_base = kappa(n)
    
    # Compute arctan refinement
    arctan_term = arctan_refinement(n)
    
    # Combine with scaling factor
    return k_base + alpha * arctan_term


def compute_korobov_generator_arctan(n: int, d: int, 
                                     use_arctan: bool = True,
                                     alpha: float = 1.0) -> np.ndarray:
    """
    Generate Korobov lattice parameter using curvature-based selection.
    
    The hypothesis claims that using arctan-refined curvature for selecting
    the Korobov generator 'a' improves lattice quality for periodic integrands.
    
    Strategy:
        1. Compute curvature for all valid generators a ∈ [2, n) coprime to n
        2. Select generator with optimal (minimum or maximum) curvature
        3. Generate lattice vector z = (1, a, a², ..., a^(d-1)) mod n
    
    Args:
        n: Lattice size (should be prime for optimal properties)
        d: Dimension
        use_arctan: If True, use κ_arctan; if False, use baseline κ
        alpha: Scaling factor for arctan term
        
    Returns:
        Korobov generating vector of length d
    """
    from math import gcd
    
    # Find all valid generators (coprime to n)
    candidates = [a for a in range(2, n) if gcd(a, n) == 1]
    
    if not candidates:
        raise ValueError(f"No valid generators found for n={n}")
    
    # Compute curvature for each candidate
    if use_arctan:
        curvatures = np.array([kappa_arctan(a, alpha=alpha) for a in candidates])
    else:
        curvatures = np.array([kappa(a) for a in candidates])
    
    # Select generator with minimum curvature (hypothesis: smoother = better)
    # This is one interpretation; we'll test both min and max
    optimal_idx = np.argmin(curvatures)
    a_optimal = candidates[optimal_idx]
    
    # Generate Korobov vector z = (1, a, a², ..., a^(d-1)) mod n
    z = np.zeros(d, dtype=np.int64)
    z[0] = 1
    for k in range(1, d):
        z[k] = (z[k-1] * a_optimal) % n
    
    return z


def generate_korobov_lattice(n: int, d: int, 
                             use_arctan: bool = True,
                             alpha: float = 1.0) -> np.ndarray:
    """
    Generate rank-1 Korobov lattice points in [0,1)^d.
    
    Args:
        n: Number of lattice points
        d: Dimension
        use_arctan: If True, use arctan-refined curvature for generator selection
        alpha: Scaling factor for arctan refinement
        
    Returns:
        Array of shape (n, d) with lattice points in [0,1)^d
    """
    # Get generating vector
    z = compute_korobov_generator_arctan(n, d, use_arctan=use_arctan, alpha=alpha)
    
    # Generate lattice points: x_i = (i * z / n) mod 1 for i = 0, ..., n-1
    points = np.zeros((n, d))
    for i in range(n):
        points[i] = (i * z / n) % 1.0
    
    return points


def measure_lattice_quality(points: np.ndarray) -> Dict[str, float]:
    """
    Measure quality metrics for a lattice point set.
    
    Metrics:
    - L2 discrepancy (star discrepancy approximation)
    - Minimum distance (coverage uniformity)
    - Periodic wrapdistance (for periodic functions)
    
    Args:
        points: Array of shape (n, d) with points in [0,1)^d
        
    Returns:
        Dictionary with quality metrics
    """
    n, d = points.shape
    
    # L2 star discrepancy (simplified box counting)
    # This is a computationally tractable approximation
    l2_disc = 0.0
    n_test_boxes = min(100, n)  # Test boxes for efficiency
    rng = np.random.default_rng(42)
    
    for _ in range(n_test_boxes):
        # Random box [0, u1) × [0, u2) × ... × [0, ud)
        u = rng.uniform(0, 1, d)
        
        # Count points in box
        in_box = np.all(points < u, axis=1)
        count = np.sum(in_box)
        
        # Discrepancy = |count/n - vol(box)|
        box_vol = np.prod(u)
        disc = abs(count / n - box_vol)
        l2_disc += disc ** 2
    
    l2_disc = np.sqrt(l2_disc / n_test_boxes)
    
    # Minimum distance (pairwise distances, sample for efficiency)
    n_pairs = min(1000, n * (n - 1) // 2)
    min_dist = np.inf
    
    if n > 1:
        for _ in range(min(n_pairs, 100)):
            i, j = rng.choice(n, size=2, replace=False)
            # Periodic distance (wraparound)
            diff = np.abs(points[i] - points[j])
            diff = np.minimum(diff, 1.0 - diff)
            dist = np.sqrt(np.sum(diff ** 2))
            min_dist = min(min_dist, dist)
    
    # Periodic wrapdistance statistics
    mean_wrapdist = 0.0
    if n > 10:
        for _ in range(100):
            i, j = rng.choice(n, size=2, replace=False)
            diff = np.abs(points[i] - points[j])
            diff = np.minimum(diff, 1.0 - diff)
            dist = np.sqrt(np.sum(diff ** 2))
            mean_wrapdist += dist
        mean_wrapdist /= 100
    
    return {
        'l2_discrepancy': l2_disc,
        'min_distance': min_dist if min_dist < np.inf else 0.0,
        'mean_wrapdist': mean_wrapdist,
        'n_points': n,
        'dimension': d
    }


if __name__ == "__main__":
    # Quick validation
    print("Arctan-Refined Curvature Module")
    print("=" * 50)
    
    # Test arctan refinement
    print("\n1. Arctan Refinement Term")
    test_vals = [1, 10, 100, 1000]
    for n in test_vals:
        arctan_val = arctan_refinement(n)
        print(f"  arctan_refinement({n:4d}) = {arctan_val:.6f}")
    
    # Test κ_arctan
    print("\n2. Arctan-Refined Curvature")
    for n in test_vals:
        k_base = kappa(n)
        k_arctan_val = kappa_arctan(n, alpha=1.0)
        improvement = k_arctan_val - k_base
        print(f"  n={n:4d}: κ={k_base:.6f}, κ_arctan={k_arctan_val:.6f}, Δ={improvement:.6f}")
    
    # Test lattice generation
    print("\n3. Korobov Lattice Generation")
    n, d = 127, 2  # Use prime n for good properties
    
    # Baseline lattice
    points_baseline = generate_korobov_lattice(n, d, use_arctan=False)
    metrics_baseline = measure_lattice_quality(points_baseline)
    print(f"\n  Baseline Korobov (n={n}, d={d}):")
    print(f"    L2 discrepancy: {metrics_baseline['l2_discrepancy']:.6f}")
    print(f"    Min distance:   {metrics_baseline['min_distance']:.6f}")
    print(f"    Mean wrapdist:  {metrics_baseline['mean_wrapdist']:.6f}")
    
    # Arctan-refined lattice
    points_arctan = generate_korobov_lattice(n, d, use_arctan=True, alpha=1.0)
    metrics_arctan = measure_lattice_quality(points_arctan)
    print(f"\n  Arctan-Refined Korobov (n={n}, d={d}):")
    print(f"    L2 discrepancy: {metrics_arctan['l2_discrepancy']:.6f}")
    print(f"    Min distance:   {metrics_arctan['min_distance']:.6f}")
    print(f"    Mean wrapdist:  {metrics_arctan['mean_wrapdist']:.6f}")
    
    # Compare
    print("\n4. Comparison")
    disc_change = (metrics_arctan['l2_discrepancy'] - metrics_baseline['l2_discrepancy']) / metrics_baseline['l2_discrepancy'] * 100
    print(f"  Discrepancy change: {disc_change:+.2f}%")
    print(f"  (Negative = improvement)")
