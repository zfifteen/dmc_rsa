#!/usr/bin/env python3
"""
Benchmark script for elliptic cyclic lattice vs standard methods
Demonstrates performance characteristics of elliptic geometry embedding
"""

import sys
sys.path.append('scripts')

import numpy as np
from rank1_lattice import (
    Rank1LatticeConfig, generate_rank1_lattice,
    compute_lattice_quality_metrics
)
from qmc_engines import QMCConfig, make_engine
import time


def benchmark_method(name, config_or_engine, n):
    """Benchmark a single method"""
    start = time.time()
    
    if isinstance(config_or_engine, QMCConfig):
        engine = make_engine(config_or_engine)
        points = engine.random(n)
    else:
        points = generate_rank1_lattice(config_or_engine)
    
    gen_time = time.time() - start
    
    # Compute quality metrics
    start = time.time()
    metrics = compute_lattice_quality_metrics(points)
    metrics_time = time.time() - start
    
    return {
        'name': name,
        'n_points': len(points),
        'gen_time': gen_time,
        'metrics_time': metrics_time,
        'min_distance': metrics['min_distance'],
        'covering_radius': metrics['covering_radius']
    }


def run_benchmarks():
    """Run comprehensive benchmarks comparing methods"""
    print("="*80)
    print("Elliptic Cyclic Lattice Benchmark")
    print("="*80)
    
    n_values = [64, 128, 256]
    
    for n in n_values:
        print(f"\n{'='*80}")
        print(f"n = {n} points, d = 2 dimensions")
        print(f"{'='*80}\n")
        
        results = []
        
        # 1. Sobol baseline
        try:
            cfg = QMCConfig(dim=2, n=n, engine="sobol", scramble=True, seed=42)
            results.append(benchmark_method("Sobol-Owen", cfg, n))
        except Exception as e:
            print(f"Sobol benchmark failed: {e}")
        
        # 2. Halton baseline
        try:
            cfg = QMCConfig(dim=2, n=n, engine="halton", scramble=True, seed=42)
            results.append(benchmark_method("Halton-Scrambled", cfg, n))
        except Exception as e:
            print(f"Halton benchmark failed: {e}")
        
        # 3. Fibonacci lattice
        try:
            cfg = Rank1LatticeConfig(n=n, d=2, generator_type="fibonacci", scramble=False, seed=42)
            results.append(benchmark_method("Rank1-Fibonacci", cfg, n))
        except Exception as e:
            print(f"Fibonacci benchmark failed: {e}")
        
        # 4. Cyclic lattice
        try:
            subgroup_order = max(2, n // 4)
            cfg = Rank1LatticeConfig(
                n=n, d=2, generator_type="cyclic", 
                subgroup_order=subgroup_order, scramble=False, seed=42
            )
            results.append(benchmark_method(f"Rank1-Cyclic (m={subgroup_order})", cfg, n))
        except Exception as e:
            print(f"Cyclic benchmark failed: {e}")
        
        # 5. Elliptic cyclic with e=0.6 (b=0.8a)
        try:
            cfg = Rank1LatticeConfig(
                n=n, d=2, generator_type="elliptic_cyclic",
                subgroup_order=n, elliptic_b=0.8, scramble=False, seed=42
            )
            results.append(benchmark_method("Elliptic-Cyclic (e=0.6)", cfg, n))
        except Exception as e:
            print(f"Elliptic cyclic (e=0.6) benchmark failed: {e}")
        
        # 6. Elliptic cyclic with e=0.75 (b=0.66a)
        try:
            cfg = Rank1LatticeConfig(
                n=n, d=2, generator_type="elliptic_cyclic",
                subgroup_order=n, elliptic_b=0.66, scramble=False, seed=42
            )
            results.append(benchmark_method("Elliptic-Cyclic (e=0.75)", cfg, n))
        except Exception as e:
            print(f"Elliptic cyclic (e=0.75) benchmark failed: {e}")
        
        # Print results table
        print(f"{'Method':<30} {'Gen Time':<12} {'Min Dist':<12} {'Cover Rad':<12}")
        print("-"*80)
        
        for r in results:
            print(f"{r['name']:<30} {r['gen_time']:>10.4f}s {r['min_distance']:>11.6f} {r['covering_radius']:>11.6f}")
        
        # Calculate improvements relative to Fibonacci
        if len(results) >= 3:
            fib_idx = 2  # Fibonacci is typically 3rd
            fib_min_dist = results[fib_idx]['min_distance']
            fib_cover = results[fib_idx]['covering_radius']
            
            print(f"\n{'Improvements vs Fibonacci:':<30}")
            print("-"*80)
            
            for r in results[3:]:  # Start from cyclic onwards
                if r['min_distance'] > 0:
                    min_dist_improvement = (r['min_distance'] / fib_min_dist - 1) * 100
                else:
                    min_dist_improvement = -100
                
                if r['covering_radius'] > 0:
                    cover_improvement = (1 - r['covering_radius'] / fib_cover) * 100
                else:
                    cover_improvement = 0
                
                print(f"{r['name']:<30} Min Dist: {min_dist_improvement:>6.1f}%  Cover: {cover_improvement:>6.1f}%")
    
    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)
    print("\nKey Observations:")
    print("- Elliptic embedding provides geometric optimization for cyclic lattices")
    print("- Eccentricity parameter (e) controls point distribution shape")
    print("- Best results when n ≈ subgroup_order (avoids multi-cycle aliasing)")
    print("- Trade-offs between Euclidean metrics and elliptic arc uniformity")
    print("="*80)


if __name__ == "__main__":
    run_benchmarks()
