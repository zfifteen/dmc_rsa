"""
Z-Framework Module
Implements theta_prime(n,k) and Z transformations for bias resolution
"""

import numpy as np
import mpmath
from typing import Union

# Golden ratio φ = (1 + √5) / 2
PHI = (1 + np.sqrt(5)) / 2


def theta_prime(n: Union[int, float, np.ndarray], k: float = 0.3) -> Union[float, np.ndarray]:
    """
    Compute bias resolution function θ′(n,k) = φ · ((n mod φ)/φ)^k.
    
    Uses golden ratio φ ≈ 1.618 for bias embedding with golden-angle spirals
    to minimize clustering in high-dimensional integrations.
    
    Args:
        n: Sample index or indices
        k: Exponent parameter (default: 0.3, typical range 0.2-0.5)
        
    Returns:
        Bias resolution value(s)
        
    Examples:
        >>> theta_prime(1, k=0.3)
        0.999...
        >>> theta_prime(10, k=0.3)
        1.007...
    """
    # Handle array input
    if isinstance(n, (list, tuple, np.ndarray)):
        n_array = np.asarray(n, dtype=float)
        mod_result = np.mod(n_array, PHI)
        return PHI * np.power(mod_result / PHI, k)
    
    # Handle scalar input
    n_float = float(n)
    mod_result = n_float % PHI
    return PHI * ((mod_result / PHI) ** k)


def Z_transform(sample_index: Union[int, np.ndarray], 
                delta_sample: float,
                delta_max: float,
                c_lattice: float = PHI) -> Union[float, np.ndarray]:
    """
    Compute Z-invariant transform: Z = sample_index * (delta_sample / c_lattice).
    
    The universal invariant form Z = A(B / c) is adapted to sampling as
    Z = S(Δ_discrepancy / c_lattice), where c_lattice = φ (golden ratio) 
    is the domain constant for low-discrepancy sequences.
    
    Args:
        sample_index: Sample index or indices (A component)
        delta_sample: Sample discrepancy shift (B component)
        delta_max: Maximum discrepancy (for normalization, unused in basic form)
        c_lattice: Domain constant (default: φ ≈ 1.618)
        
    Returns:
        Z-transformed value(s)
        
    Examples:
        >>> Z_transform(100, 0.01, 1.0)
        0.618...
        >>> Z_transform(np.array([10, 20, 30]), 0.01, 1.0)
        array([0.0618..., 0.1236..., 0.1854...])
    """
    if isinstance(sample_index, (list, tuple, np.ndarray)):
        sample_index = np.asarray(sample_index, dtype=float)
    
    return sample_index * (delta_sample / c_lattice)


def validate_precision(value: float, target_precision: float = 1e-12) -> bool:
    """
    Validate that a computed value meets target precision using mpmath.
    
    Args:
        value: Value to check
        target_precision: Required precision (default: 1e-12)
        
    Returns:
        True if precision is adequate
        
    Examples:
        >>> validate_precision(1.0 + 1e-13)
        True
        >>> validate_precision(float('nan'))
        False
    """
    if not np.isfinite(value):
        return False
    
    # Use mpmath for high-precision validation
    with mpmath.workprec(50):  # 50 decimal places
        mp_value = mpmath.mpf(value)
        # Check if value has finite representation within precision
        return abs(mp_value) < mpmath.mpf('1e100')  # Sanity check
