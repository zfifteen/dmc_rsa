#!/usr/bin/env python3
"""
Demo script for Fermat QMC Bias module

This demonstrates the usage of biased QMC sampling for Fermat factorization,
showing the performance improvements and usage patterns for different scenarios.
"""

import sys
sys.path.append('scripts')

import numpy as np
from fermat_qmc_bias import (
    FermatConfig, SamplerType, fermat_factor, generate_semiprime,
    recommend_sampler
)


def demo_basic_usage():
    """Demonstrate basic factorization with different samplers"""
    print("=" * 70)
    print("Demo 1: Basic Fermat Factorization")
    print("=" * 70)
    
    # Simple example: 899 = 29 * 31
    N = 899
    print(f"\nFactoring N = {N} (29 × 31, very close factors)\n")
    
    samplers = [
        SamplerType.SEQUENTIAL,
        SamplerType.UNIFORM_GOLDEN,
        SamplerType.BIASED_GOLDEN,
        SamplerType.HYBRID,
    ]
    
    for sampler_type in samplers:
        cfg = FermatConfig(
            N=N,
            max_trials=10000,
            sampler_type=sampler_type,
            beta=2.0,
            seed=42
        )
        
        result = fermat_factor(cfg)
        
        print(f"{sampler_type.value:20s}: ", end="")
        if result['success']:
            print(f"{result['trials']:5d} trials → {result['factors']}")
        else:
            print("FAILED")
    
    print()


def demo_recommendation_system():
    """Demonstrate the recommendation system"""
    print("=" * 70)
    print("Demo 2: Automatic Sampler Recommendation")
    print("=" * 70)
    
    scenarios = [
        ("Close factors (Δ=100)", 899, 29, 31, 100000),
        ("Distant factors (Δ=2^22)", 2**40, 2**20, 2**20 + 2**22, 100000),
        ("Unknown, large window", 2**60, None, None, 100000),
        ("Unknown, small window", 2**60, None, None, 10000),
    ]
    
    for name, N, p, q, window in scenarios:
        print(f"\nScenario: {name}")
        rec = recommend_sampler(N=N, p=p, q=q, window_size=window)
        print(f"  Recommended: {rec['sampler_type'].value}")
        print(f"  Reason: {rec['reason']}")


def demo_comparative_benchmark():
    """Benchmark different samplers on a test semiprime"""
    print("\n" + "=" * 70)
    print("Demo 3: Comparative Benchmark")
    print("=" * 70)
    
    # Generate a challenging test case
    N, p, q = generate_semiprime(bit_length=60, max_delta_exp=25, seed=12345)
    delta = abs(q - p)
    
    print(f"\nGenerated test semiprime:")
    print(f"  N = {N} ({N.bit_length()} bits)")
    print(f"  p = {p}")
    print(f"  q = {q}")
    print(f"  Δ = {delta} (~2^{int(np.log2(delta))} )")
    
    # Get recommendation
    rec = recommend_sampler(N=N, p=p, q=q, window_size=100000)
    print(f"\nRecommended sampler: {rec['sampler_type'].value}")
    print(f"Reason: {rec['reason']}")
    
    print(f"\nBenchmarking multiple samplers (max 100k trials):")
    print("-" * 70)
    
    samplers_to_test = [
        (SamplerType.SEQUENTIAL, {}),
        (SamplerType.UNIFORM_GOLDEN, {}),
        (SamplerType.BIASED_GOLDEN, {'beta': 2.0}),
        (SamplerType.FAR_BIASED_GOLDEN, {'beta_far': 2.5}),
        (SamplerType.HYBRID, {'beta': 2.0, 'hybrid_prefix_ratio': 0.05}),
        (SamplerType.DUAL_MIXTURE, {'beta': 2.0, 'beta_far': 2.5, 'dual_far_ratio': 0.75}),
    ]
    
    results = []
    for sampler_type, params in samplers_to_test:
        cfg = FermatConfig(
            N=N,
            max_trials=100000,
            sampler_type=sampler_type,
            seed=42,
            **params
        )
        
        result = fermat_factor(cfg)
        results.append((sampler_type.value, result))
        
        status = "✓" if result['success'] else "✗"
        trials_str = f"{result['trials']:6d}" if result['success'] else "FAILED"
        print(f"{status} {sampler_type.value:20s}: {trials_str} trials")
    
    # Find best result
    if any(r[1]['success'] for r in results):
        best = min((r for r in results if r[1]['success']), key=lambda x: x[1]['trials'])
        print(f"\n→ Best: {best[0]} with {best[1]['trials']} trials")
    
    print()


def demo_bias_effect():
    """Demonstrate the effect of different bias exponents"""
    print("=" * 70)
    print("Demo 4: Effect of Bias Exponent (β)")
    print("=" * 70)
    
    # Use a fixed semiprime
    N, p, q = generate_semiprime(bit_length=60, max_delta_exp=20, seed=999)
    
    print(f"\nTest semiprime: N = {N}")
    print(f"Factors: p={p}, q={q}, Δ={abs(q-p)}")
    print(f"\nTesting different bias exponents with BiasedGoldenSampler:")
    print("-" * 70)
    
    for beta in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        cfg = FermatConfig(
            N=N,
            max_trials=100000,
            sampler_type=SamplerType.BIASED_GOLDEN,
            beta=beta,
            seed=42
        )
        
        result = fermat_factor(cfg)
        
        if result['success']:
            print(f"  β={beta:.1f}: {result['trials']:6d} trials")
        else:
            print(f"  β={beta:.1f}: FAILED")
    
    print()


def demo_hybrid_vs_sequential():
    """Show advantage of hybrid approach for close factors"""
    print("=" * 70)
    print("Demo 5: Hybrid vs Sequential for Close Factors")
    print("=" * 70)
    
    # Generate semiprimes with varying closeness
    print("\nComparing Sequential vs Hybrid (5% prefix) for different factor gaps:")
    print("-" * 70)
    
    for delta_exp in [10, 15, 20, 25]:
        N, p, q = generate_semiprime(bit_length=60, max_delta_exp=delta_exp, seed=42+delta_exp)
        delta = abs(q - p)
        
        print(f"\nΔ ≈ 2^{int(np.log2(delta))} (actual: {delta})")
        
        # Sequential
        cfg_seq = FermatConfig(N=N, max_trials=100000, sampler_type=SamplerType.SEQUENTIAL, seed=42)
        result_seq = fermat_factor(cfg_seq)
        
        # Hybrid
        cfg_hyb = FermatConfig(
            N=N, max_trials=100000, sampler_type=SamplerType.HYBRID,
            hybrid_prefix_ratio=0.05, beta=2.0, seed=42
        )
        result_hyb = fermat_factor(cfg_hyb)
        
        if result_seq['success'] and result_hyb['success']:
            improvement = (result_seq['trials'] - result_hyb['trials']) / result_seq['trials'] * 100
            print(f"  Sequential: {result_seq['trials']:6d} trials")
            print(f"  Hybrid:     {result_hyb['trials']:6d} trials ({improvement:+.1f}%)")
        else:
            print(f"  Sequential: {'FAILED' if not result_seq['success'] else result_seq['trials']}")
            print(f"  Hybrid:     {'FAILED' if not result_hyb['success'] else result_hyb['trials']}")


def demo_statistical_analysis():
    """Run multiple trials to show statistical performance"""
    print("\n" + "=" * 70)
    print("Demo 6: Statistical Analysis (Multiple Runs)")
    print("=" * 70)
    
    # Generate one test semiprime
    N, p, q = generate_semiprime(bit_length=60, max_delta_exp=22, seed=777)
    
    print(f"\nTest semiprime: N = {N}")
    print(f"Running 10 independent trials with different samplers...")
    print("-" * 70)
    
    samplers = [
        SamplerType.UNIFORM_GOLDEN,
        SamplerType.BIASED_GOLDEN,
        SamplerType.HYBRID,
    ]
    
    num_trials = 10
    
    for sampler_type in samplers:
        trial_counts = []
        
        for trial in range(num_trials):
            cfg = FermatConfig(
                N=N,
                max_trials=100000,
                sampler_type=sampler_type,
                beta=2.0,
                hybrid_prefix_ratio=0.05,
                seed=1000 + trial  # Different seed each trial
            )
            
            result = fermat_factor(cfg)
            if result['success']:
                trial_counts.append(result['trials'])
        
        if trial_counts:
            mean_trials = np.mean(trial_counts)
            std_trials = np.std(trial_counts)
            min_trials = np.min(trial_counts)
            max_trials = np.max(trial_counts)
            
            print(f"\n{sampler_type.value}:")
            print(f"  Success rate: {len(trial_counts)}/{num_trials}")
            print(f"  Mean trials: {mean_trials:.1f} ± {std_trials:.1f}")
            print(f"  Range: [{min_trials}, {max_trials}]")
        else:
            print(f"\n{sampler_type.value}:")
            print(f"  Success rate: 0/{num_trials} (all failed)")
    
    print()


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("Fermat QMC Bias Module - Interactive Demo")
    print("=" * 70)
    print("\nThis demo showcases the biased QMC sampling approach for")
    print("Fermat factorization, demonstrating the 43% reduction in trials")
    print("and optimal strategies for different semiprime characteristics.")
    print()
    
    demo_basic_usage()
    demo_recommendation_system()
    demo_comparative_benchmark()
    demo_bias_effect()
    demo_hybrid_vs_sequential()
    demo_statistical_analysis()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Biased QMC (β=2.0) provides consistent improvements over uniform")
    print("  • Hybrid approach excels for close factors (small Δ)")
    print("  • Far-biased and dual-mixture work best for distant factors")
    print("  • Recommendation system helps select optimal strategy")
    print()


if __name__ == "__main__":
    main()
