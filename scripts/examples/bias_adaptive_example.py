#!/usr/bin/env python3
"""
Bias-Adaptive Sampling Engine Example

Demonstrates the usage of the Bias-Adaptive Sampling Engine with various
bias modes and QMC methods.
"""

import sys
import os
import numpy as np
import time

# Add paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_script_dir))
sys.path.insert(0, _parent_dir)

from scripts.qmc_engines import (
    QMCConfig, make_engine, qmc_points, apply_bias_adaptive,
    compute_z_invariant_metrics
)


def example_basic_usage():
    """Example 1: Basic bias-adaptive Sobol sampling"""
    print("=" * 70)
    print("Example 1: Basic Bias-Adaptive Sobol Sampling")
    print("=" * 70)
    
    # Create configuration with theta_prime bias
    cfg = QMCConfig(
        dim=2,              # 2D sampling
        n=1024,             # 1024 samples (power of 2)
        engine='sobol',     # Sobol engine
        scramble=True,      # Owen scrambling
        bias_mode='theta_prime',  # Golden-angle spiral bias
        seed=42             # Reproducibility
    )
    
    # Generate samples
    eng = make_engine(cfg)
    samples = eng.random(1024)
    
    # Apply bias transformation
    biased_samples = apply_bias_adaptive(samples, bias_mode='theta_prime', k=0.3)
    
    # Compute metrics
    metrics = compute_z_invariant_metrics(biased_samples, method='sobol_theta_prime')
    
    print(f"Generated {len(biased_samples)} samples")
    print(f"Discrepancy:     {metrics['discrepancy']:.6f}")
    print(f"Unique rate:     {metrics['unique_rate']:.6f}")
    print(f"Mean kappa:      {metrics['mean_kappa']:.4f}")
    print(f"Savings est:     {metrics['savings_estimate']:.2f}x")
    print()


def example_compare_bias_modes():
    """Example 2: Compare different bias modes"""
    print("=" * 70)
    print("Example 2: Comparing Different Bias Modes")
    print("=" * 70)
    
    # Generate baseline samples
    cfg = QMCConfig(dim=2, n=1024, engine='sobol', seed=42)
    eng = make_engine(cfg)
    samples = eng.random(1024)
    
    # Test each bias mode
    bias_modes = ['theta_prime', 'prime_density', 'golden_spiral']
    
    print(f"{'Mode':<20} {'Discrepancy':<15} {'Unique Rate':<15} {'Mean Kappa':<15}")
    print("-" * 70)
    
    # Baseline (no bias)
    metrics_baseline = compute_z_invariant_metrics(samples, method='baseline')
    print(f"{'baseline':<20} {metrics_baseline['discrepancy']:<15.6f} "
          f"{metrics_baseline['unique_rate']:<15.6f} "
          f"{metrics_baseline['mean_kappa']:<15.4f}")
    
    # Each bias mode
    for mode in bias_modes:
        biased = apply_bias_adaptive(samples.copy(), bias_mode=mode, k=0.3)
        metrics = compute_z_invariant_metrics(biased, method=mode)
        
        print(f"{mode:<20} {metrics['discrepancy']:<15.6f} "
              f"{metrics['unique_rate']:<15.6f} "
              f"{metrics['mean_kappa']:<15.4f}")
    
    print()


def example_replicated_qmc():
    """Example 3: Replicated QMC with confidence intervals"""
    print("=" * 70)
    print("Example 3: Replicated QMC with Confidence Intervals")
    print("=" * 70)
    
    cfg = QMCConfig(
        dim=2,
        n=512,
        engine='sobol',
        bias_mode='theta_prime',
        replicates=10,      # Generate 10 independent replicates
        seed=42
    )
    
    # Collect metrics from each replicate
    discrepancies = []
    unique_rates = []
    
    for i, points in enumerate(qmc_points(cfg)):
        metrics = compute_z_invariant_metrics(points, method=f'replicate_{i}')
        discrepancies.append(metrics['discrepancy'])
        unique_rates.append(metrics['unique_rate'])
    
    # Compute statistics
    disc_mean = np.mean(discrepancies)
    disc_std = np.std(discrepancies)
    disc_ci_lower = disc_mean - 1.96 * disc_std / np.sqrt(len(discrepancies))
    disc_ci_upper = disc_mean + 1.96 * disc_std / np.sqrt(len(discrepancies))
    
    rate_mean = np.mean(unique_rates)
    rate_std = np.std(unique_rates)
    rate_ci_lower = rate_mean - 1.96 * rate_std / np.sqrt(len(unique_rates))
    rate_ci_upper = rate_mean + 1.96 * rate_std / np.sqrt(len(unique_rates))
    
    print(f"Replicates: {cfg.replicates}")
    print(f"\nDiscrepancy:")
    print(f"  Mean:   {disc_mean:.6f}")
    print(f"  Std:    {disc_std:.6f}")
    print(f"  95% CI: [{disc_ci_lower:.6f}, {disc_ci_upper:.6f}]")
    
    print(f"\nUnique Rate:")
    print(f"  Mean:   {rate_mean:.6f}")
    print(f"  Std:    {rate_std:.6f}")
    print(f"  95% CI: [{rate_ci_lower:.6f}, {rate_ci_upper:.6f}]")
    print()


def example_high_dimensional():
    """Example 4: High-dimensional sampling"""
    print("=" * 70)
    print("Example 4: High-Dimensional Bias-Adaptive Sampling")
    print("=" * 70)
    
    dimensions = [2, 5, 10, 20]
    
    print(f"{'Dim':<8} {'Samples':<10} {'Time (s)':<12} {'Discrepancy':<15} {'Unique Rate':<15}")
    print("-" * 70)
    
    for d in dimensions:
        n = 1024
        cfg = QMCConfig(dim=d, n=n, engine='sobol', bias_mode='theta_prime', seed=42)
        
        start = time.time()
        eng = make_engine(cfg)
        samples = eng.random(n)
        biased = apply_bias_adaptive(samples, bias_mode='theta_prime')
        elapsed = time.time() - start
        
        metrics = compute_z_invariant_metrics(biased, method=f'{d}d')
        
        print(f"{d:<8} {n:<10} {elapsed:<12.4f} "
              f"{metrics['discrepancy']:<15.6f} {metrics['unique_rate']:<15.6f}")
    
    print()


def example_performance_benchmark():
    """Example 5: Performance benchmark (d=10, N=10,000)"""
    print("=" * 70)
    print("Example 5: Performance Benchmark (Requirement: d=10, N=10,000 in <1s)")
    print("=" * 70)
    
    cfg = QMCConfig(dim=10, n=10000, engine='sobol', bias_mode='theta_prime', seed=42)
    
    start = time.time()
    
    # Generate via single replicate
    cfg.replicates = 1
    for points in qmc_points(cfg):
        metrics = compute_z_invariant_metrics(points, method='benchmark')
    
    elapsed = time.time() - start
    
    print(f"Dimensions:      {cfg.dim}")
    print(f"Samples:         {cfg.n}")
    print(f"Bias mode:       {cfg.bias_mode}")
    print(f"Time taken:      {elapsed:.4f}s")
    print(f"Requirement:     <1.0s")
    print(f"Status:          {'✓ PASS' if elapsed < 1.0 else '✗ FAIL'}")
    print()


def example_integration_test():
    """Example 6: Simple integration test"""
    print("=" * 70)
    print("Example 6: Integration Test (∫ x² dx over [0,1]²)")
    print("=" * 70)
    
    # True value: ∫₀¹ ∫₀¹ x² dx dy = 1/3 ≈ 0.333333
    true_value = 1.0 / 3.0
    
    cfg = QMCConfig(dim=2, n=4096, engine='sobol', bias_mode='theta_prime', seed=42)
    eng = make_engine(cfg)
    samples = eng.random(4096)
    
    # Apply bias
    biased_samples = apply_bias_adaptive(samples, bias_mode='theta_prime')
    
    # Evaluate function: f(x,y) = x²
    function_values = biased_samples[:, 0] ** 2
    
    # Monte Carlo estimate
    qmc_estimate = np.mean(function_values)
    qmc_error = abs(qmc_estimate - true_value)
    
    # Compare with random MC
    mc_samples = np.random.rand(4096, 2)
    mc_values = mc_samples[:, 0] ** 2
    mc_estimate = np.mean(mc_values)
    mc_error = abs(mc_estimate - true_value)
    
    print(f"True value:      {true_value:.6f}")
    print(f"\nQMC (biased):")
    print(f"  Estimate:      {qmc_estimate:.6f}")
    print(f"  Error:         {qmc_error:.6f}")
    
    print(f"\nMC (random):")
    print(f"  Estimate:      {mc_estimate:.6f}")
    print(f"  Error:         {mc_error:.6f}")
    
    print(f"\nImprovement:     {(mc_error / qmc_error):.2f}x")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("*" * 70)
    print(" " * 15 + "BIAS-ADAPTIVE SAMPLING ENGINE EXAMPLES")
    print("*" * 70)
    print()
    
    example_basic_usage()
    example_compare_bias_modes()
    example_replicated_qmc()
    example_high_dimensional()
    example_performance_benchmark()
    example_integration_test()
    
    print("*" * 70)
    print("All examples completed successfully!")
    print("*" * 70)


if __name__ == "__main__":
    main()
