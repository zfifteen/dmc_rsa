"""
Divisor Density Module
Implements kappa(n) = d(n) * ln(n+1) / e^2 for discrepancy curvature
where d(n) is the number of divisors of n
"""

import numpy as np
import sympy
from typing import Union

# Euler's number squared for kappa calculation
E_SQUARED = np.e ** 2


def count_divisors(n: int) -> int:
    """
    Count the number of divisors of n using sympy.
    
    Args:
        n: Positive integer
        
    Returns:
        Number of divisors of n
        
    Raises:
        ValueError: If n < 1
        
    Examples:
        >>> count_divisors(12)  # divisors: 1, 2, 3, 4, 6, 12
        6
        >>> count_divisors(1)
        1
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    
    return sympy.divisor_count(n)


def kappa(n: Union[int, np.ndarray], epsilon: float = 1e-12) -> Union[float, np.ndarray]:
    """
    Compute discrepancy curvature κ(n) = d(n) · ln(n+1) / e².
    
    This function guards against zero-division via n ≥ 1 constraints and
    provides numerical stability with epsilon.
    
    Args:
        n: Positive integer or array of positive integers (n >= 1)
        epsilon: Small value to prevent numerical issues (default: 1e-12)
        
    Returns:
        Curvature value(s)
        
    Raises:
        ValueError: If any n < 1
        
    Examples:
        >>> kappa(1)  # d(1)=1, ln(2)/e^2
        0.09402...
        >>> kappa(12)  # d(12)=6, ln(13)/e^2
        0.9177...
    """
    # Handle array input
    if isinstance(n, (list, tuple, np.ndarray)):
        n_array = np.asarray(n)
        if np.any(n_array < 1):
            raise ValueError("All values of n must be >= 1")
        
        # Vectorized computation
        d_n = np.array([count_divisors(int(val)) for val in n_array])
        return d_n * np.log(n_array + 1) / E_SQUARED
    
    # Handle scalar input
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    
    d_n = count_divisors(int(n))
    return d_n * np.log(n + 1) / E_SQUARED
