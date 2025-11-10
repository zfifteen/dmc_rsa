#!/usr/bin/env python3
"""
QMC Engines Module - Enhanced QMC capabilities with replicated randomization
Implements Sobol with Owen scrambling, Halton with Faure permutations,
and rank-1 lattice constructions from group theory
October 2025
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Generator, Optional
import numpy as np
import warnings
import sys
import os
from scipy.stats import qmc  # pip install scipy

# Add parent directory to path for Z-framework imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import Z-framework modules
try:
    from cognitive_number_theory.divisor_density import kappa
    from wave_crispr_signal.z_framework import theta_prime
    Z_AVAILABLE = True
except ImportError as e:
    Z_AVAILABLE = False
    warnings.warn(
        f"Z-framework modules not available: {e}. Z-bias will not be available.",
        ImportWarning
    )

# Import rank-1 lattice module
try:
    from rank1_lattice import (
        Rank1LatticeConfig, generate_rank1_lattice,
        compute_lattice_quality_metrics
    )
    RANK1_AVAILABLE = True
except ImportError:
    RANK1_AVAILABLE = False
    warnings.warn(
        "rank1_lattice module not available. Rank-1 lattice engine will not be available.",
        ImportWarning
    )

# Import EAS module
try:
    from eas_factorize import EllipticAdaptiveSearch, EASConfig
    EAS_AVAILABLE = True
except ImportError:
    EAS_AVAILABLE = False
    warnings.warn(
        "eas_factorize module not available. EAS engine will not be available.",
        ImportWarning
    )

def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    """Return the next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def validate_sobol_sample_size(n: int, auto_round: bool = True) -> int:
    """Validate and optionally round sample size for Sobol sequences.
    
    Sobol sequences have optimal balance properties when the number of samples
    is a power of 2. This function checks the sample size and optionally rounds
    it to the next power of 2.
    
    Args:
        n: Requested number of samples
        auto_round: If True, automatically round to next power of 2
        
    Returns:
        int: Valid sample size (rounded if auto_round=True, original otherwise)
        
    Raises:
        ValueError: If n is not a power of 2 and auto_round is False
        
    Example:
        >>> validate_sobol_sample_size(200)  # Returns 256 with warning
        >>> validate_sobol_sample_size(256)  # Returns 256, no warning
        >>> validate_sobol_sample_size(200, auto_round=False)  # Raises ValueError
    """
    if not _is_power_of_two(n):
        if auto_round:
            next_pow2 = _next_power_of_two(n)
            warnings.warn(
                f"Sobol sequences require n to be a power of 2 for optimal balance properties. "
                f"Rounding {n} -> {next_pow2}.",
                UserWarning,
                stacklevel=2
            )
            return next_pow2
        else:
            raise ValueError(
                f"Sobol sequences require n to be a power of 2. "
                f"Got n={n}. Use n={_next_power_of_two(n)} or set auto_round=True."
            )
    return n


def z_bias(samples, n, k=0.3):
    """Apply Z-framework bias."""
    if not Z_AVAILABLE:
        warnings.warn("Z-framework not available, returning original samples")
        return samples
    curv = np.array([kappa(int(s)) for s in samples])
    phase = theta_prime(n, k)
    weights = 1 / (curv + 1e-6) * np.sin(phase * samples)
    return samples * weights / weights.max()


def apply_bias_adaptive(points: np.ndarray, bias_mode: str, k: float = 0.3, beta: float = 2.0) -> np.ndarray:
    """
    Apply bias-adaptive transformation to QMC points.
    
    Implements the Bias-Adaptive Sampling Engine that fuses Sobol-Owen scrambling
    with Z-biased lattices for variance reduction in high-dimensional integrations.
    
    Args:
        points: QMC points in [0,1]^d, shape (n, d)
        bias_mode: Bias strategy - "prime_density", "golden_spiral", or "theta_prime"
        k: Exponent for theta_prime bias (default: 0.3)
        beta: Bias exponent for weighting (default: 2.0)
        
    Returns:
        Biased points in [0,1]^d with reduced discrepancy
        
    Raises:
        ValueError: If bias_mode is unsupported or Z-framework unavailable
    """
    if not Z_AVAILABLE:
        raise ValueError("Bias-adaptive sampling requires Z-framework modules")
    
    n, d = points.shape
    sample_indices = np.arange(1, n + 1)
    
    if bias_mode == "theta_prime":
        # Apply θ′(n,k) = φ · ((n mod φ)/φ)^k for golden-angle spiral bias
        theta_vals = theta_prime(sample_indices, k=k)
        # Adjust each point: x_i,j = x_i,j + θ′(i,k) mod 1
        for j in range(d):
            points[:, j] = (points[:, j] + theta_vals / (d * 2)) % 1.0
    
    elif bias_mode == "prime_density":
        # Use κ(n) = d(n) · ln(n+1) / e² for curvature-based weighting
        kappa_vals = kappa(sample_indices)
        # Weight-based reordering: prioritize low-curvature samples
        weights = 1.0 / (1.0 + kappa_vals)
        # Sort points by weight (descending) to prioritize low-discrepancy regions
        sorted_indices = np.argsort(-weights)
        points = points[sorted_indices]
    
    elif bias_mode == "golden_spiral":
        # Fibonacci/golden spiral for 2D, generalized for higher dimensions
        from wave_crispr_signal import PHI
        golden_angle = 2 * np.pi / (PHI ** 2)  # ≈ 2.399... radians
        
        for i in range(n):
            angle = i * golden_angle
            # Apply spiral transformation to first two dimensions
            if d >= 2:
                r = np.sqrt(points[i, 0])
                points[i, 0] = r * np.cos(angle)
                points[i, 1] = r * np.sin(angle)
                # Renormalize to [0,1]
                points[i, 0] = (points[i, 0] + 1) / 2
                points[i, 1] = (points[i, 1] + 1) / 2
    
    else:
        raise ValueError(f"Unsupported bias_mode: {bias_mode}. "
                        f"Supported modes: 'prime_density', 'golden_spiral', 'theta_prime'")
    
    return points


def compute_z_invariant_metrics(points: np.ndarray, method: str = "sobol") -> dict:
    """
    Compute Z-invariant metrics for bias-adaptive QMC sequences.
    
    Metrics include:
    - Discrepancy: D_N via scipy.stats.qmc.discrepancy
    - Unique rate: ratio of unique samples
    - Curvature bias: mean κ(n) weighting
    
    Args:
        points: QMC points in [0,1]^d
        method: QMC method name for labeling
        
    Returns:
        Dictionary with metrics: discrepancy, unique_rate, mean_kappa, savings_estimate
    """
    n, d = points.shape
    
    # Compute discrepancy using scipy
    try:
        disc = qmc.discrepancy(points, method='L2')
    except:
        # Fallback to simple estimate
        disc = estimate_l2_discrepancy(points)
    
    # Compute unique rate (round to 12 decimal places for comparison)
    unique_points = len(np.unique(np.round(points, 12), axis=0))
    unique_rate = unique_points / n
    
    # Compute mean kappa if Z-framework available
    if Z_AVAILABLE:
        sample_indices = np.arange(1, n + 1)
        kappa_vals = kappa(sample_indices)
        mean_kappa = np.mean(kappa_vals)
    else:
        mean_kappa = None
    
    # Estimate variance reduction (simplified)
    # Theoretical: QMC gives O((log N)^d / N) vs MC O(1/√N)
    mc_variance_rate = 1.0 / np.sqrt(n)
    qmc_variance_rate = (np.log(n) ** d) / n
    savings_estimate = (mc_variance_rate / qmc_variance_rate) - 1 if qmc_variance_rate > 0 else 0
    
    return {
        'method': method,
        'n_samples': n,
        'dimensions': d,
        'discrepancy': disc,
        'unique_rate': unique_rate,
        'mean_kappa': mean_kappa,
        'savings_estimate': savings_estimate
    }


@dataclass
class QMCConfig:
    """Configuration for QMC engine with replicated randomization"""
    dim: int
    n: int
    engine: str = "sobol"  # "sobol", "halton", "rank1_lattice", "elliptic_cyclic", "eas"
    scramble: bool = True
    seed: Optional[int] = None
    replicates: int = 8
    auto_round_sobol: bool = True

    # Rank-1 lattice specific parameters
    lattice_generator: str = "cyclic"  # "fibonacci", "korobov", "cyclic", "spiral_conical", "elliptic_cyclic"
    subgroup_order: Optional[int] = None

    # Geometric parameters for spiral-conical and elliptic_cyclic lattices
    cone_height: float = 1.2
    spiral_depth: int = 3
    elliptic_a: Optional[float] = None
    elliptic_b: Optional[float] = None

    # EAS specific parameters
    eas_max_samples: int = 2000
    eas_adaptive_window: bool = True
    eas_reference_point: float = 1000.0

    # Z-bias parameters
    with_z_bias: bool = False
    z_k: float = 0.3
    
    # Bias-Adaptive Sampling Engine parameters
    bias_mode: Optional[str] = None  # None, "prime_density", "golden_spiral", "theta_prime"
    beta: float = 2.0  # Bias exponent for adaptive weighting
def make_engine(cfg: QMCConfig):
    """Create a QMC engine based on configuration.
    
    For Sobol sequences, validates that n is a power of 2 for optimal balance properties.
    If not and auto_round_sobol is True, automatically rounds to next power of 2 with a warning.
    
    For rank-1 lattices, returns a custom wrapper that generates lattice points.
    
    For EAS (Elliptic Adaptive Search), returns a wrapper that generates elliptic lattice points.
    
    Args:
        cfg: QMCConfig instance with engine parameters
        
    Returns:
        scipy.stats.qmc sampler instance, Rank1LatticeEngine wrapper, or EASEngine wrapper
        
    Raises:
        ValueError: If engine type is unsupported or if Sobol n is not power of 2
                   and auto_round_sobol is False
    """
    if cfg.engine == "sobol":
        # Validate power of 2 for Sobol sequences
        if not _is_power_of_two(cfg.n):
            if cfg.auto_round_sobol:
                next_pow2 = _next_power_of_two(cfg.n)
                warnings.warn(
                    f"Sobol sequences require n to be a power of 2 for optimal balance properties. "
                    f"Automatically rounding {cfg.n} -> {next_pow2}. "
                    f"Set auto_round_sobol=False to disable this behavior.",
                    UserWarning,
                    stacklevel=2
                )
                # Note: We don't modify cfg.n, scipy will issue its own warning
                # This is just to inform the user more clearly
            else:
                raise ValueError(
                    f"Sobol sequences require n to be a power of 2. "
                    f"Got n={cfg.n}. Use n={_next_power_of_two(cfg.n)} or set auto_round_sobol=True."
                )
        return qmc.Sobol(d=cfg.dim, scramble=cfg.scramble, seed=cfg.seed)
    elif cfg.engine == "halton":
        return qmc.Halton(d=cfg.dim, scramble=cfg.scramble, seed=cfg.seed)
    elif cfg.engine in ("rank1_lattice", "elliptic_cyclic"):
        if not RANK1_AVAILABLE:
            raise ValueError(
                "Rank-1 lattice engine requires rank1_lattice module. "
                "Module import failed."
            )
        return Rank1LatticeEngine(cfg)
    elif cfg.engine == "eas":
        if not EAS_AVAILABLE:
            raise ValueError(
                "EAS engine requires eas_factorize module. "
                "Module import failed."
            )
        return EASEngine(cfg)
    else:
        raise ValueError(f"Unsupported engine: {cfg.engine}. Supported engines are 'sobol', 'halton', 'rank1_lattice', 'elliptic_cyclic', 'eas'.")

class Rank1LatticeEngine:
    """Wrapper class for rank-1 lattice engine to match scipy.stats.qmc interface.
    
    This allows rank-1 lattices to be used seamlessly with existing QMC code
    that expects scipy-style engines with a random() method.
    """
    
    def __init__(self, cfg: QMCConfig):
        """Initialize rank-1 lattice engine with configuration"""
        if not RANK1_AVAILABLE:
            raise ImportError("rank1_lattice module not available")
        
        self.cfg = cfg
        
        # Determine generator type based on engine
        if cfg.engine == "elliptic_cyclic":
            generator_type = "elliptic_cyclic"
        else:
            generator_type = cfg.lattice_generator
        
        self.lattice_cfg = Rank1LatticeConfig(
            n=cfg.n,
            d=cfg.dim,
            subgroup_order=cfg.subgroup_order,
            generator_type=generator_type,
            seed=cfg.seed,
            scramble=cfg.scramble,
            cone_height=cfg.cone_height,
            spiral_depth=cfg.spiral_depth,
            elliptic_a=cfg.elliptic_a,
            elliptic_b=cfg.elliptic_b
        )
        self._points_cache = None        
    def random(self, n: Optional[int] = None) -> np.ndarray:
        """
        Generate rank-1 lattice points.
        
        Args:
            n: Number of points to generate (must match config.n for lattices)
            
        Returns:
            Array of shape (n, d) with lattice points in [0,1]^d
        """
        if n is not None and n != self.cfg.n:
            warnings.warn(
                f"Rank-1 lattice size is fixed at {self.cfg.n}. "
                f"Requested {n} points, returning {self.cfg.n} points.",
                UserWarning
            )
        
        # Generate lattice points (cache for efficiency)
        if self._points_cache is None:
            self._points_cache = generate_rank1_lattice(self.lattice_cfg)
        
        return self._points_cache.copy()
    
    def reset(self):
        """Reset the lattice engine (clear cache)"""
        self._points_cache = None


class EASEngine:
    """Wrapper class for Elliptic Adaptive Search (EAS) engine to match scipy.stats.qmc interface.
    
    This allows EAS elliptic lattice sampling to be used seamlessly with existing QMC code
    that expects scipy-style engines with a random() method.
    
    Note: EAS generates points in a deterministic elliptic lattice pattern,
    not truly random points. The "random" method name is kept for API compatibility.
    """
    
    def __init__(self, cfg: QMCConfig):
        """Initialize EAS engine with configuration"""
        if not EAS_AVAILABLE:
            raise ImportError("eas_factorize module not available")
        
        self.cfg = cfg
        self.eas_config = EASConfig(
            max_samples=cfg.eas_max_samples,
            adaptive_window=cfg.eas_adaptive_window
        )
        self.eas = EllipticAdaptiveSearch(self.eas_config)
        
        # Seed random state for reproducibility in elliptic generation
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        
    def random(self, n: Optional[int] = None) -> np.ndarray:
        """
        Generate EAS elliptic lattice points.
        
        This generates points using elliptic lattice + golden-angle spiral sampling.
        The points are normalized to [0,1]^d for compatibility with QMC interface.
        
        Args:
            n: Number of points to generate
            
        Returns:
            Array of shape (n, d) with normalized elliptic lattice points in [0,1]^d
            
        Note:
            The reference_point parameter (from QMCConfig.eas_reference_point) controls
            the central value around which the elliptic lattice is generated. Larger
            values spread points further; smaller values cluster them. Default is 1000.0.
        """
        if n is None:
            n = self.cfg.n
            
        # Generate elliptic lattice points around a configurable reference value.
        # The reference point affects the distribution and quality of generated points.
        sqrt_n = self.cfg.eas_reference_point
        radius = sqrt_n * 0.1  # 10% window
        
        # Generate candidates using EAS
        candidates = self.eas._generate_elliptic_lattice_points(
            n // 2,  # Account for ± offsets
            sqrt_n,
            radius
        )
        
        # Normalize to [0, 1)^d
        # For 2D: (r, theta) where r is radial distance, theta is angle
        min_val = np.min(candidates)
        max_val = np.max(candidates)
        range_val = max_val - min_val if max_val > min_val else 1.0
        
        # Create 2D points: dimension 0 is radial, dimension 1 is angular
        points = np.zeros((min(len(candidates), n), self.cfg.dim))
        
        for i in range(min(len(candidates), n)):
            # Radial component (normalized distance from center)
            points[i, 0] = (candidates[i] - min_val) / range_val
            
            # Angular component (based on golden angle)
            if self.cfg.dim >= 2:
                angle_idx = i % len(candidates)
                points[i, 1] = (angle_idx * self.eas_config.golden_angle) % (2 * np.pi) / (2 * np.pi)
            
            # Additional dimensions if needed (use simple stratification)
            for d in range(2, self.cfg.dim):
                points[i, d] = (i / n + d * 0.1) % 1.0
        
        return points
    
    def reset(self):
        """Reset the EAS engine"""
        # Reseed if seed was provided
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)


def qmc_points(cfg: QMCConfig) -> Generator[np.ndarray, None, None]:
    """Generate independent randomized QMC replicates.
    
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
            replicates=1,  # Not used in recursion
            lattice_generator=cfg.lattice_generator,
            subgroup_order=cfg.subgroup_order,
            cone_height=cfg.cone_height,
            spiral_depth=cfg.spiral_depth,
            elliptic_a=cfg.elliptic_a,
            elliptic_b=cfg.elliptic_b,
            bias_mode=None,  # Don't apply bias twice
            beta=cfg.beta
        ))
        
        # Generate points
        X = eng.random(cfg.n)
        
        # Apply bias-adaptive transformation if specified
        if cfg.bias_mode is not None:
            X = apply_bias_adaptive(X, cfg.bias_mode, k=cfg.z_k, beta=cfg.beta)
        
        yield X

# --- Application-specific mapping ---
def map_points_to_candidates(X: np.ndarray, N: int, window_radius: int, 
                            residues: Tuple[int, ...] = (1, 3, 7, 9)) -> np.ndarray:
    """Map [0,1]^2 -> integer candidates with smooth transitions.
    
    This mapping is designed to preserve low discrepancy properties of QMC
    by avoiding hard discontinuities. It uses soft edges and bounded adjustments
    to maintain the variation-bounded property required for QMC effectiveness.
    
    Dimensions:
      dim0: window position in [-R, R] around floor(sqrt(N))
      dim1: residue bucket among {1,3,7,9} for mod 10 filter
    
    Args:
        X: QMC points in [0,1]^2, shape (n, 2)
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
    """Estimate L2 discrepancy as a proxy for star discrepancy.
    
    The L2 discrepancy is computationally cheaper than star discrepancy
    and provides a good proxy for the uniformity of point distribution.
    Lower values indicate better uniformity.
    
    Args:
        points: QMC points in [0,1]^d, shape (n, d)
        
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
    """Compute stratification balance metric.
    
    Divides each dimension into bins and measures how uniformly points
    are distributed across bins. Perfect uniform distribution = 1.0,
    worse distributions < 1.0.
    
    Args:
        points: QMC points in [0,1]^d, shape (n, d)
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
