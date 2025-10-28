#!/usr/bin/env python3
"""
Tests for fermat_qmc_bias module

Validates biased QMC Fermat factorization implementations including:
- All sampler types
- Correctness of factorization
- Sampler recommendations
- Performance characteristics
"""

import sys
sys.path.append('scripts')

import numpy as np
from fermat_qmc_bias import (
    FermatConfig, SamplerType, fermat_factor, make_sampler,
    generate_semiprime, recommend_sampler, is_square, fermat_trial,
    SequentialSampler, UniformRandomSampler, UniformGoldenSampler,
    BiasedRandomSampler, BiasedGoldenSampler, FarBiasedGoldenSampler,
    HybridSampler, DualMixtureSampler
)


def test_is_square():
    """Test square detection"""
    print("Testing is_square...")
    
    assert is_square(0) == True
    assert is_square(1) == True
    assert is_square(4) == True
    assert is_square(9) == True
    assert is_square(16) == True
    assert is_square(100) == True
    
    assert is_square(2) == False
    assert is_square(3) == False
    assert is_square(5) == False
    assert is_square(99) == False
    assert is_square(-1) == False
    
    print("  ✓ is_square works correctly")


def test_fermat_trial():
    """Test single Fermat trial"""
    print("Testing fermat_trial...")
    
    # N = 899 = 29 * 31
    N = 899
    sqrtN = 30  # floor(sqrt(899))
    
    # k=0: a=30, a^2-N = 900-899 = 1 = 1^2 ✓
    result = fermat_trial(N, sqrtN, 0)
    assert result == (29, 31), f"Expected (29, 31), got {result}"
    
    # k=1: a=31, a^2-N = 961-899 = 62 (not square)
    result = fermat_trial(N, sqrtN, 1)
    assert result is None
    
    # Large N: 899 = 29 * 31, also works with offset
    result = fermat_trial(899, 29, 1)  # a=30, matches above
    assert result == (29, 31)
    
    print("  ✓ fermat_trial works correctly")


def test_generate_semiprime():
    """Test semiprime generation"""
    print("Testing generate_semiprime...")
    
    N, p, q = generate_semiprime(bit_length=60, max_delta_exp=20, seed=42)
    
    # Verify factorization
    assert p * q == N, f"p*q != N: {p}*{q} = {p*q} != {N}"
    
    # Verify bit length is approximately correct
    assert 50 <= N.bit_length() <= 70, f"Unexpected bit length: {N.bit_length()}"
    
    # Verify p and q are primes (sympy guarantees this from nextprime)
    import sympy as sp
    assert sp.isprime(p), f"p={p} is not prime"
    assert sp.isprime(q), f"q={q} is not prime"
    
    print(f"  Generated: N={N} ({N.bit_length()} bits) = {p} * {q}")
    print("  ✓ generate_semiprime works correctly")


def test_sequential_sampler():
    """Test sequential sampler"""
    print("Testing SequentialSampler...")
    
    sampler = SequentialSampler(window_size=1000, seed=42)
    
    # Sequential should return 0, 1, 2, 3, ...
    for i in range(10):
        assert sampler.sample(i) == i
    
    print("  ✓ SequentialSampler works correctly")


def test_uniform_random_sampler():
    """Test uniform random sampler"""
    print("Testing UniformRandomSampler...")
    
    sampler = UniformRandomSampler(window_size=1000, seed=42)
    
    # Generate samples
    samples = [sampler.sample(i) for i in range(100)]
    
    # Check range
    assert all(0 <= s < 1000 for s in samples)
    
    # Check some variability (not all the same)
    assert len(set(samples)) > 10
    
    print(f"  Generated {len(set(samples))} unique samples from 100 trials")
    print("  ✓ UniformRandomSampler works correctly")


def test_uniform_golden_sampler():
    """Test uniform golden ratio sampler (QMC)"""
    print("Testing UniformGoldenSampler...")
    
    sampler = UniformGoldenSampler(window_size=1000, seed=42)
    
    # Generate samples
    samples = [sampler.sample(i) for i in range(100)]
    
    # Check range
    assert all(0 <= s < 1000 for s in samples)
    
    # Check low discrepancy - should have better coverage than random
    assert len(set(samples)) > 80  # Golden ratio should avoid duplicates better
    
    print(f"  Generated {len(set(samples))} unique samples from 100 trials")
    print("  ✓ UniformGoldenSampler works correctly")


def test_biased_golden_sampler():
    """Test biased golden sampler"""
    print("Testing BiasedGoldenSampler...")
    
    sampler = BiasedGoldenSampler(window_size=1000, beta=2.0, seed=42)
    
    # Generate samples
    samples = [sampler.sample(i) for i in range(100)]
    
    # Check range
    assert all(0 <= s < 1000 for s in samples)
    
    # Check bias toward small values
    median = np.median(samples)
    assert median < 500, f"Expected median < 500 with beta=2.0, got {median}"
    
    print(f"  Median sample: {median} (biased toward 0)")
    print("  ✓ BiasedGoldenSampler works correctly")


def test_far_biased_golden_sampler():
    """Test far-biased sampler"""
    print("Testing FarBiasedGoldenSampler...")
    
    sampler = FarBiasedGoldenSampler(window_size=1000, beta=2.5, seed=42)
    
    # Generate samples
    samples = [sampler.sample(i) for i in range(100)]
    
    # Check range
    assert all(0 <= s < 1000 for s in samples)
    
    # Check bias toward large values
    median = np.median(samples)
    assert median > 500, f"Expected median > 500 with far-bias, got {median}"
    
    print(f"  Median sample: {median} (biased toward window_size)")
    print("  ✓ FarBiasedGoldenSampler works correctly")


def test_hybrid_sampler():
    """Test hybrid sampler"""
    print("Testing HybridSampler...")
    
    sampler = HybridSampler(window_size=1000, prefix_ratio=0.1, beta=2.0, seed=42)
    
    # First 100 samples (10% of 1000) should be sequential
    for i in range(100):
        assert sampler.sample(i) == i, f"Expected {i}, got {sampler.sample(i)}"
    
    # Remaining samples should be biased
    later_samples = [sampler.sample(i) for i in range(100, 200)]
    assert len(set(later_samples)) > 50  # Should have variety
    
    print("  ✓ HybridSampler works correctly")


def test_dual_mixture_sampler():
    """Test dual mixture sampler"""
    print("Testing DualMixtureSampler...")
    
    sampler = DualMixtureSampler(window_size=1000, far_ratio=0.75,
                                 beta_near=2.0, beta_far=2.5, seed=42)
    
    # Generate samples
    samples = [sampler.sample(i) for i in range(200)]
    
    # Check range
    assert all(0 <= s < 1000 for s in samples)
    
    # Should have mix of small and large values
    median = np.median(samples)
    print(f"  Median sample: {median} (should be mixed)")
    
    print("  ✓ DualMixtureSampler works correctly")


def test_fermat_factor_simple():
    """Test basic Fermat factorization"""
    print("Testing fermat_factor with simple case...")
    
    # N = 899 = 29 * 31 (very close factors)
    cfg = FermatConfig(
        N=899,
        max_trials=100,
        sampler_type=SamplerType.SEQUENTIAL,
        seed=42
    )
    
    result = fermat_factor(cfg)
    
    assert result['success'] == True
    assert result['factors'] == (29, 31)
    assert result['trials'] <= 10  # Should find quickly with sequential
    
    print(f"  Factored 899 in {result['trials']} trials")
    print(f"  Factors: {result['factors']}")
    print("  ✓ fermat_factor works correctly")


def test_fermat_factor_perfect_square():
    """Test perfect square detection"""
    print("Testing fermat_factor with perfect square...")
    
    cfg = FermatConfig(
        N=900,  # 30^2
        max_trials=100,
        sampler_type=SamplerType.SEQUENTIAL,
        seed=42
    )
    
    result = fermat_factor(cfg)
    
    assert result['success'] == True
    assert result['factors'] == (30, 30)
    assert result['trials'] == 0  # Should detect immediately
    
    print(f"  Detected perfect square: {result['factors']}")
    print("  ✓ Perfect square detection works")


def test_fermat_factor_all_samplers():
    """Test all sampler types can factor"""
    print("Testing all sampler types...")
    
    N = 899  # 29 * 31
    
    for sampler_type in SamplerType:
        cfg = FermatConfig(
            N=N,
            max_trials=10000,
            sampler_type=sampler_type,
            seed=42
        )
        
        result = fermat_factor(cfg)
        
        assert result['success'] == True, f"{sampler_type.value} failed to factor {N}"
        assert result['factors'] == (29, 31)
        
        print(f"  {sampler_type.value}: {result['trials']} trials")
    
    print("  ✓ All samplers work correctly")


def test_recommend_sampler():
    """Test sampler recommendation"""
    print("Testing recommend_sampler...")
    
    # Test with known close factors
    rec = recommend_sampler(N=899, p=29, q=31, window_size=100000)
    assert rec['sampler_type'] == SamplerType.HYBRID
    print(f"  Close factors → {rec['sampler_type'].value}: {rec['reason']}")
    
    # Test with known distant factors
    rec = recommend_sampler(N=2**40, p=2**20, q=2**20+2**22, window_size=100000)
    assert rec['sampler_type'] == SamplerType.DUAL_MIXTURE
    print(f"  Distant factors → {rec['sampler_type'].value}: {rec['reason']}")
    
    # Test unknown with large window
    rec = recommend_sampler(N=2**60, window_size=100000)
    assert rec['sampler_type'] == SamplerType.BIASED_GOLDEN
    print(f"  Unknown, large window → {rec['sampler_type'].value}: {rec['reason']}")
    
    # Test unknown with small window
    rec = recommend_sampler(N=2**60, window_size=10000)
    assert rec['sampler_type'] == SamplerType.UNIFORM_GOLDEN
    print(f"  Unknown, small window → {rec['sampler_type'].value}: {rec['reason']}")
    
    print("  ✓ recommend_sampler works correctly")


def test_comparative_performance():
    """Test comparative performance of samplers"""
    print("Testing comparative performance...")
    
    # Generate a test semiprime with moderate gap
    N, p, q = generate_semiprime(bit_length=60, max_delta_exp=20, seed=42)
    
    print(f"  Test semiprime: N={N} ({N.bit_length()} bits)")
    print(f"  Factors: p={p}, q={q}")
    print(f"  Gap: Δ={abs(q-p)}")
    
    # Test different samplers
    samplers = [
        SamplerType.SEQUENTIAL,
        SamplerType.UNIFORM_GOLDEN,
        SamplerType.BIASED_GOLDEN,
    ]
    
    results = {}
    for sampler_type in samplers:
        cfg = FermatConfig(
            N=N,
            max_trials=100000,
            sampler_type=sampler_type,
            beta=2.0,
            seed=42
        )
        
        result = fermat_factor(cfg)
        results[sampler_type.value] = result
        
        if result['success']:
            print(f"  {sampler_type.value}: {result['trials']} trials")
        else:
            print(f"  {sampler_type.value}: FAILED")
    
    print("  ✓ Comparative test complete")


def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("Fermat QMC Bias Test Suite")
    print("=" * 60)
    
    test_is_square()
    test_fermat_trial()
    test_generate_semiprime()
    test_sequential_sampler()
    test_uniform_random_sampler()
    test_uniform_golden_sampler()
    test_biased_golden_sampler()
    test_far_biased_golden_sampler()
    test_hybrid_sampler()
    test_dual_mixture_sampler()
    test_fermat_factor_simple()
    test_fermat_factor_perfect_square()
    test_fermat_factor_all_samplers()
    test_recommend_sampler()
    test_comparative_performance()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
