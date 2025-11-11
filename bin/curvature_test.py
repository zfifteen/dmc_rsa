#!/usr/bin/env python3
"""
curvature_test.py - Curvature Reduction Analysis with Bootstrap CI

Analyzes curvature reduction κ(n) across prime-mapped slots with bootstrap
confidence intervals. Validates empirical predictions from Z-framework.

Usage:
    python bin/curvature_test.py --slots 1000 --prime nearest --output curvature.csv
"""

import sys
import os
import argparse
import csv
import numpy as np
import time
from typing import List, Dict, Tuple

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_number_theory import kappa
import sympy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Curvature Reduction Analysis with Bootstrap CI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python bin/curvature_test.py --slots 1000 --prime nearest --output curvature.csv
  python bin/curvature_test.py --slots 100000 --prime nearest --bootstrap 1000 --output curvature_z5d.csv
        """
    )
    
    parser.add_argument('--slots', type=int, default=1000,
                       help='Number of slots to analyze (default: 1000)')
    
    parser.add_argument('--prime', type=str, default='nearest',
                       choices=['nearest', 'next', 'prev'],
                       help='Prime mapping strategy (default: nearest)')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file for curvature data')
    
    parser.add_argument('--bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def get_prime_mapped_slot(slot: int, strategy: str = 'nearest') -> int:
    """
    Map slot index to prime number using specified strategy.
    
    Args:
        slot: Slot index (positive integer)
        strategy: 'nearest', 'next', or 'prev'
        
    Returns:
        Prime-mapped slot index
    """
    if slot < 1:
        slot = 1
    
    if strategy == 'nearest':
        # Find nearest prime to slot
        if slot <= 2:
            return 2
        
        next_prime = sympy.nextprime(slot - 1)  # Next prime >= slot
        
        # Check if there's a previous prime
        if slot > 2:
            prev_prime = sympy.prevprime(slot)  # Previous prime < slot
            
            # Choose nearest
            if abs(next_prime - slot) <= abs(prev_prime - slot):
                return int(next_prime)
            else:
                return int(prev_prime)
        else:
            return int(next_prime)
    
    elif strategy == 'next':
        return int(sympy.nextprime(slot - 1))
    
    elif strategy == 'prev':
        if slot <= 2:
            return 2
        return int(sympy.prevprime(slot))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compute_curvature_reduction(baseline_kappa: float, 
                                biased_kappa: float) -> float:
    """
    Compute curvature reduction percentage.
    
    Reduction = (1 - biased_kappa / baseline_kappa) * 100%
    
    Args:
        baseline_kappa: Baseline curvature value
        biased_kappa: Biased/reduced curvature value
        
    Returns:
        Reduction percentage (positive means improvement)
    """
    if baseline_kappa == 0:
        return 0.0
    
    reduction = (1.0 - biased_kappa / baseline_kappa) * 100.0
    return reduction


def bootstrap_confidence_interval(data: np.ndarray,
                                  statistic_fn,
                                  n_iterations: int = 1000,
                                  confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Input data array
        statistic_fn: Function to compute statistic (e.g., np.mean)
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        stat = statistic_fn(resample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles for CI
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_stat = np.mean(bootstrap_stats)
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return mean_stat, lower_bound, upper_bound


def analyze_curvature_reduction(slots: List[int],
                                prime_strategy: str = 'nearest',
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95,
                                verbose: bool = False) -> Dict:
    """
    Analyze curvature reduction across slots with bootstrap CI.
    
    Args:
        slots: List of slot indices to analyze
        prime_strategy: Prime mapping strategy
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
        verbose: Print progress
        
    Returns:
        Dictionary with analysis results
    """
    if verbose:
        print(f"Analyzing curvature reduction for {len(slots)} slots...")
        print(f"Prime mapping strategy: {prime_strategy}")
        print(f"Bootstrap iterations: {n_bootstrap}")
    
    # Compute baseline and biased kappa values
    baseline_kappas = []
    biased_kappas = []
    prime_mapped_slots = []
    reductions = []
    
    start_time = time.time()
    
    for i, slot in enumerate(slots):
        # Map to prime
        prime_slot = get_prime_mapped_slot(slot, prime_strategy)
        prime_mapped_slots.append(prime_slot)
        
        # Compute kappa values
        baseline_k = float(kappa(slot))
        biased_k = float(kappa(prime_slot))
        
        baseline_kappas.append(baseline_k)
        biased_kappas.append(biased_k)
        
        # Compute reduction
        reduction = compute_curvature_reduction(baseline_k, biased_k)
        reductions.append(reduction)
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(slots)} slots...")
    
    elapsed_time = time.time() - start_time
    
    # Convert to numpy arrays
    reductions = np.array(reductions)
    baseline_kappas = np.array(baseline_kappas)
    biased_kappas = np.array(biased_kappas)
    
    # Compute statistics with bootstrap CI
    if verbose:
        print(f"Computing bootstrap confidence intervals...")
    
    reduction_mean, reduction_lower, reduction_upper = bootstrap_confidence_interval(
        reductions, np.mean, n_bootstrap, confidence
    )
    
    kappa_mean, kappa_lower, kappa_upper = bootstrap_confidence_interval(
        biased_kappas, np.mean, n_bootstrap, confidence
    )
    
    # Compute additional statistics
    results = {
        'n_slots': len(slots),
        'prime_strategy': prime_strategy,
        'elapsed_time': elapsed_time,
        
        # Curvature reduction statistics
        'reduction_mean': reduction_mean,
        'reduction_std': np.std(reductions),
        'reduction_ci_lower': reduction_lower,
        'reduction_ci_upper': reduction_upper,
        'reduction_min': np.min(reductions),
        'reduction_max': np.max(reductions),
        
        # Baseline kappa statistics
        'baseline_kappa_mean': np.mean(baseline_kappas),
        'baseline_kappa_std': np.std(baseline_kappas),
        
        # Biased kappa statistics
        'biased_kappa_mean': kappa_mean,
        'biased_kappa_ci_lower': kappa_lower,
        'biased_kappa_ci_upper': kappa_upper,
        
        # Raw data
        'slots': slots,
        'prime_mapped_slots': prime_mapped_slots,
        'baseline_kappas': baseline_kappas.tolist(),
        'biased_kappas': biased_kappas.tolist(),
        'reductions': reductions.tolist(),
    }
    
    return results


def write_results_csv(results: Dict, output_path: str, verbose: bool = False):
    """Write analysis results to CSV file"""
    if verbose:
        print(f"Writing results to {output_path}...")
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata as comments
        writer.writerow([f"# Curvature Reduction Analysis"])
        writer.writerow([f"# n_slots: {results['n_slots']}"])
        writer.writerow([f"# prime_strategy: {results['prime_strategy']}"])
        writer.writerow([f"# elapsed_time: {results['elapsed_time']:.4f}s"])
        writer.writerow([])
        
        # Write summary statistics
        writer.writerow([f"# Curvature Reduction Statistics"])
        writer.writerow([f"# mean: {results['reduction_mean']:.4f}%"])
        writer.writerow([f"# std: {results['reduction_std']:.4f}%"])
        writer.writerow([f"# 95% CI: [{results['reduction_ci_lower']:.4f}%, {results['reduction_ci_upper']:.4f}%]"])
        writer.writerow([f"# range: [{results['reduction_min']:.4f}%, {results['reduction_max']:.4f}%]"])
        writer.writerow([])
        
        # Write header
        writer.writerow(['slot', 'prime_mapped_slot', 'baseline_kappa', 'biased_kappa', 'reduction_percent'])
        
        # Write data rows
        for i in range(results['n_slots']):
            writer.writerow([
                results['slots'][i],
                results['prime_mapped_slots'][i],
                f"{results['baseline_kappas'][i]:.8f}",
                f"{results['biased_kappas'][i]:.8f}",
                f"{results['reductions'][i]:.4f}"
            ])
    
    if verbose:
        print(f"Results written successfully.")


def main():
    """Main entry point"""
    args = parse_args()
    
    if args.verbose:
        print("=" * 70)
        print("Curvature Reduction Analysis with Bootstrap CI")
        print("=" * 70)
        print()
    
    # Generate slot indices
    slots = list(range(1, args.slots + 1))
    
    # Analyze curvature reduction
    results = analyze_curvature_reduction(
        slots=slots,
        prime_strategy=args.prime,
        n_bootstrap=args.bootstrap,
        confidence=args.confidence,
        verbose=args.verbose
    )
    
    # Print summary
    if args.verbose:
        print()
        print("=" * 70)
        print("Summary Statistics")
        print("=" * 70)
        print(f"Slots analyzed:        {results['n_slots']}")
        print(f"Prime strategy:        {results['prime_strategy']}")
        print(f"Elapsed time:          {results['elapsed_time']:.4f}s")
        print()
        print(f"Curvature Reduction:")
        print(f"  Mean:                {results['reduction_mean']:.2f}%")
        print(f"  Std Dev:             {results['reduction_std']:.2f}%")
        print(f"  95% CI:              [{results['reduction_ci_lower']:.2f}%, {results['reduction_ci_upper']:.2f}%]")
        print(f"  Range:               [{results['reduction_min']:.2f}%, {results['reduction_max']:.2f}%]")
        print()
        print(f"Biased Kappa:")
        print(f"  Mean:                {results['biased_kappa_mean']:.6f}")
        print(f"  95% CI:              [{results['biased_kappa_ci_lower']:.6f}, {results['biased_kappa_ci_upper']:.6f}]")
        print("=" * 70)
        print()
    
    # Write results to CSV
    write_results_csv(results, args.output, verbose=args.verbose)
    
    if args.verbose:
        print(f"Analysis complete. Results saved to {args.output}")
        print()


if __name__ == '__main__':
    main()
