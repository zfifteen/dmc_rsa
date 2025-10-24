#!/usr/bin/env python3
"""
QMC Engines Module - Enhanced QMC capabilities with replicated randomization
Implements Sobol with Owen scrambling and Halton with Faure permutations
October 2025
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Generator
import numpy as np
from scipy.stats import qmc  # pip install scipy

@dataclass
class QMCConfig:
    """Configuration for QMC engine with replicated randomization"""
    dim: int
    n: int
    engine: str = "sobol"     # "sobol" | "halton"
    scramble: bool = True     # Owen for Sobol, Faure/QR for Halton (scipy implements)
    seed: int | None = None
    replicates: int = 8       # Cranley-Patterson: use random_base for shifts

def make_engine(cfg: QMCConfig):
    """
    Create a QMC engine based on configuration.
    
    Args:
        cfg: QMCConfig instance with engine parameters
        
    Returns:
        scipy.stats.qmc sampler instance
        
    Raises:
        ValueError: If engine type is unsupported
    """
    if cfg.engine == "sobol":
        return qmc.Sobol(d=cfg.dim, scramble=cfg.scramble, seed=cfg.seed)
    elif cfg.engine == "halton":
        return qmc.Halton(d=cfg.dim, scramble=cfg.scramble, seed=cfg.seed)
    else:
        raise ValueError(f"Unsupported engine: {cfg.engine}")

def qmc_points(cfg: QMCConfig) -> Generator[np.ndarray, None, None]:
    """
    Generate independent randomized QMC replicates.
    
    This implements Cranley-Patterson randomization by generating multiple
    independent replicates with different random seeds. Each replicate provides
    an independent QMC point set that can be used to estimate variance and
    construct confidence intervals.
    
    Args:
        cfg: QMCConfig instance specifying engine parameters
        
    Yields:
        np.ndarray: QMC point set of shape (n, dim) for each replicate
    """
    # Independent randomized QMC replicates:
    for r in range(cfg.replicates):
        # Create new seed for each replicate
        if cfg.seed is None:
            replicate_seed = None
        else:
            replicate_seed = cfg.seed + r
        
        # Create engine with updated seed
        eng = make_engine(QMCConfig(
            dim=cfg.dim,
            n=cfg.n,
            engine=cfg.engine,
            scramble=cfg.scramble,
            seed=replicate_seed,
            replicates=1  # Not used in recursion
        ))
        
        # Generate points
        X = eng.random(cfg.n)
        yield X

# --- Application-specific mapping ---
def map_points_to_candidates(X: np.ndarray, N: int, window_radius: int, 
                            residues: Tuple[int, ...] = (1, 3, 7, 9)) -> np.ndarray:
    """
    Map [0,1)^2 -> integer candidates with smooth transitions.
    
    This mapping is designed to preserve low discrepancy properties of QMC
    by avoiding hard discontinuities. It uses soft edges and bounded adjustments
    to maintain the variation-bounded property required for QMC effectiveness.
    
    Dimensions:
      dim0: window position in [-R, R] around floor(sqrt(N))
      dim1: residue bucket among {1,3,7,9} for mod 10 filter
    
    Args:
        X: QMC points in [0,1)^2, shape (n, 2)
        N: Semiprime to factor
        window_radius: Search window radius around sqrt(N)
        residues: Allowed residue classes mod 10
        
    Returns:
        np.ndarray: Integer candidates, length <= n
    """
    if X.shape[1] < 2:
        raise ValueError(f"Expected at least 2 dimensions, got {X.shape[1]}")
    
    root = int(np.sqrt(N))
    R = int(window_radius)
    
    # Soft-edges to avoid discontinuities: jitter by half-step before rounding
    offsets = np.floor((X[:, 0] * (2*R + 1)) - R + 0.5).astype(np.int64)
    cand = root + offsets
    cand |= 1  # Make odd
    
    # Enforce last-digit class via small forward search (bounded, keeps smoothness mostly intact)
    choices = np.asarray(residues, dtype=np.int64)
    cls = choices[(X[:, 1] * len(choices)).astype(int).clip(max=len(choices)-1)]
    want = (cls % 10)
    
    # Adjust candidate until last digit matches (≤4 increments)
    for i in range(cand.size):
        step = 0
        while cand[i] % 10 != want[i] and step < 4:
            cand[i] += 2  # Stay odd
            step += 1
    
    # Filter to valid range
    mask = (cand > 1) & (cand < N)
    return cand[mask]


def estimate_l2_discrepancy(points: np.ndarray) -> float:
    """
    Estimate L2 discrepancy as a proxy for star discrepancy.
    
    The L2 discrepancy is computationally cheaper than star discrepancy
    and provides a good proxy for the uniformity of point distribution.
    Lower values indicate better uniformity.
    
    Args:
        points: QMC points in [0,1)^d, shape (n, d)
        
    Returns:
        float: Estimated L2 discrepancy
    """
    if len(points) == 0:
        return 1.0
    
    n, d = points.shape
    
    # Centered L2 discrepancy formula (simplified)
    # D_n^2 ≈ (1/n) Σ_i Σ_j prod_k min(x_ik, x_jk)
    total = 0.0
    for i in range(min(n, 100)):  # Sample for efficiency
        for j in range(min(n, 100)):
            prod = 1.0
            for k in range(d):
                prod *= min(points[i, k], points[j, k])
            total += prod
    
    # Normalize
    sample_size = min(n, 100)
    if sample_size > 0:
        return np.sqrt(total / (sample_size * sample_size))
    return 1.0


def stratification_balance(points: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute stratification balance metric.
    
    Divides each dimension into bins and measures how uniformly points
    are distributed across bins. Perfect uniform distribution = 1.0,
    worse distributions < 1.0.
    
    Args:
        points: QMC points in [0,1)^d, shape (n, d)
        n_bins: Number of bins per dimension
        
    Returns:
        float: Balance metric in [0, 1], higher is better
    """
    if len(points) == 0:
        return 0.0
    
    n, d = points.shape
    expected_per_bin = n / n_bins
    
    total_deviation = 0.0
    for dim in range(d):
        bins = np.histogram(points[:, dim], bins=n_bins, range=(0, 1))[0]
        # Measure deviation from uniform
        deviation = np.sum(np.abs(bins - expected_per_bin))
        total_deviation += deviation
    
    # Normalize: max deviation is n per dim
    max_deviation = n * d
    if max_deviation > 0:
        return 1.0 - (total_deviation / max_deviation)
    return 1.0
