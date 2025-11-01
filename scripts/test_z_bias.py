#!/usr/bin/env python3
"""
Unit and integration tests for Z-framework bias in QMC engines.
"""

import numpy as np
import pytest
from qmc_engines import z_bias, Z_AVAILABLE

def test_z_bias_availability():
    """Test that Z_AVAILABLE is set correctly."""
    # This will pass if imports succeeded
    assert isinstance(Z_AVAILABLE, bool)

@pytest.mark.skipif(not Z_AVAILABLE, reason="Z-framework not available")
def test_z_bias_basic():
    """Unit test for z_bias function."""
    samples = np.array([1.0, 2.0, 3.0])
    n = 899  # Small test semiprime
    k = 0.3
    biased = z_bias(samples, n, k)
    assert len(biased) == len(samples)
    assert isinstance(biased, np.ndarray)

@pytest.mark.skipif(not Z_AVAILABLE, reason="Z-framework not available")
def test_z_bias_rsa100():
    """Integration test on RSA-100 subset."""
    # Load RSA-100
    with open('../rsa100.txt', 'r') as f:
        n = int(f.read().strip())
    
    samples = np.random.random(100)
    biased = z_bias(samples, n)
    # Check that bias is applied (weights differ)
    assert not np.allclose(biased, samples)

if __name__ == "__main__":
    pytest.main([__file__])