#!/usr/bin/env python3
"""
run_demo.py - Bias-Adaptive Sampling Engine Demo Script

Generates bias-adaptive QMC samples and outputs to CSV for analysis.

Usage:
    python scripts/run_demo.py --method sobol_owen --bias theta_prime --bits 32 --output results/samples.csv
"""

import sys
import os
import argparse
import numpy as np
import time
import csv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.qmc_engines import QMCConfig, make_engine, qmc_points, compute_z_invariant_metrics, apply_bias_adaptive
from scripts.qmc_factorization_analysis import QMCFactorization


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bias-Adaptive Sampling Engine Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Sobol-Owen samples with theta_prime bias for 32-bit factors
  python scripts/run_demo.py --method sobol_owen --bias theta_prime --bits 32 --output results/samples.csv
  
  # Generate Halton samples with prime_density bias
  python scripts/run_demo.py --method halton --bias prime_density --bits 40 --output results/samples.csv
  
  # No bias (baseline)
  python scripts/run_demo.py --method sobol_owen --bits 32 --output results/baseline.csv
        """
    )
    
    parser.add_argument('--method', type=str, default='sobol_owen',
                       choices=['sobol_owen', 'sobol', 'halton', 'rank1_lattice'],
                       help='QMC method (default: sobol_owen)')
    
    parser.add_argument('--bias', type=str, default=None,
                       choices=[None, 'theta_prime', 'prime_density', 'golden_spiral'],
                       help='Bias mode (default: None)')
    
    parser.add_argument('--bits', type=int, default=32,
                       help='Target bit length for RSA factors (default: 32)')
    
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    
    parser.add_argument('--dim', type=int, default=2,
                       help='Number of dimensions (default: 2)')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    
    parser.add_argument('--k', type=float, default=0.3,
                       help='Theta k parameter (default: 0.3)')
    
    parser.add_argument('--beta', type=float, default=2.0,
                       help='Bias exponent (default: 2.0)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def generate_samples(args):
    """Generate bias-adaptive QMC samples"""
    
    if args.verbose:
        print(f"Generating {args.samples} samples...")
        print(f"  Method: {args.method}")
        print(f"  Bias: {args.bias if args.bias else 'None'}")
        print(f"  Dimensions: {args.dim}")
        print(f"  Bits: {args.bits}")
    
    # Configure engine
    engine = args.method.replace('_owen', '')  # sobol_owen -> sobol
    scramble = '_owen' in args.method or args.method == 'sobol_owen'
    
    cfg = QMCConfig(
        dim=args.dim,
        n=args.samples,
        engine=engine,
        scramble=scramble,
        bias_mode=args.bias,
        z_k=args.k,
        beta=args.beta,
        seed=args.seed,
        replicates=1  # Single replicate for demo
    )
    
    # Generate samples
    start_time = time.time()
    
    eng = make_engine(cfg)
    samples = eng.random(args.samples)
    
    # Apply bias if specified
    if args.bias:
        samples = apply_bias_adaptive(samples, bias_mode=args.bias, k=args.k, beta=args.beta)
    
    generation_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_z_invariant_metrics(samples, method=f"{args.method}_{args.bias or 'baseline'}")
    
    if args.verbose:
        print(f"\nGeneration completed in {generation_time:.3f}s")
        print(f"  Discrepancy: {metrics['discrepancy']:.6f}")
        print(f"  Unique rate: {metrics['unique_rate']:.6f}")
        if metrics['mean_kappa'] is not None:
            print(f"  Mean kappa: {metrics['mean_kappa']:.4f}")
        print(f"  Savings estimate: {metrics['savings_estimate']:.2f}x")
    
    return samples, metrics


def save_samples(samples, output_path, args, metrics):
    """Save samples to CSV file"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata as comments
        writer.writerow(['# Bias-Adaptive Sampling Engine Demo Output'])
        writer.writerow([f'# Method: {args.method}'])
        writer.writerow([f'# Bias: {args.bias if args.bias else "None"}'])
        writer.writerow([f'# Samples: {args.samples}'])
        writer.writerow([f'# Dimensions: {args.dim}'])
        writer.writerow([f'# Bits: {args.bits}'])
        writer.writerow([f'# Seed: {args.seed}'])
        writer.writerow([f'# Discrepancy: {metrics["discrepancy"]:.6f}'])
        writer.writerow([f'# Unique_rate: {metrics["unique_rate"]:.6f}'])
        if metrics['mean_kappa'] is not None:
            writer.writerow([f'# Mean_kappa: {metrics["mean_kappa"]:.4f}'])
        writer.writerow([f'# Savings_estimate: {metrics["savings_estimate"]:.2f}'])
        writer.writerow([])  # Blank line
        
        # Write header
        header = [f'dim_{i}' for i in range(args.dim)]
        writer.writerow(header)
        
        # Write samples
        for sample in samples:
            writer.writerow(sample)
    
    if args.verbose:
        print(f"\nSamples saved to: {output_path}")


def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Generate samples
        samples, metrics = generate_samples(args)
        
        # Save to CSV
        save_samples(samples, args.output, args, metrics)
        
        print(f"✓ Successfully generated {len(samples)} samples")
        print(f"  Output: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
