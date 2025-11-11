"""
Wave CRISPR Signal Module
Implements Z-framework transformations and theta functions
Extended with Z5D support for k*≈0.04449
"""

from .z_framework import (
    theta_prime, Z_transform, validate_precision, PHI, K_Z5D,
    theta_prime_high_precision, compute_prime_density_boost, validate_z5d_extension
)

__all__ = [
    'theta_prime', 'Z_transform', 'validate_precision', 'PHI', 'K_Z5D',
    'theta_prime_high_precision', 'compute_prime_density_boost', 'validate_z5d_extension'
]
