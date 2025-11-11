#!/usr/bin/env python3
"""
discrepancy_test.py - Bootstrap Confidence Interval Analysis for QMC Samples

Reads QMC samples from CSV, computes bootstrap confidence intervals for 
discrepancy and other metrics, and outputs results.

Usage:
    python bin/discrepancy_test.py --input results/samples.csv --n_boot 1000 --output results/metrics.csv
"""

import sys
import os
import argparse
import csv
import numpy as np
from scipy.stats import qmc

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.qmc_engines import compute_z_invariant_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bootstrap Confidence Interval Analysis for QMC Samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python bin/discrepancy_test.py --input results/samples.csv --n_boot 1000 --output results/metrics.csv
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with samples')
    
    parser.add_argument('--n_boot', type=int, default=1000,
                       help='Number of bootstrap iterations (default: 1000)')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file for metrics')
    
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Confidence level (default: 0.95)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def read_samples_csv(filepath):
    """Read samples from CSV file"""
    samples = []
    metadata = {}
    
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        
        # Read metadata from comments
        for row in reader:
            if len(row) == 0:
                continue
            
            if row[0].startswith('#'):
                # Parse metadata
                line = row[0]
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip('# ')
                    value = value.strip()
                    metadata[key] = value
            elif row[0].startswith('dim_'):
                # Header row, skip
                continue
            else:
                # Data row
                try:
                    sample = [float(x) for x in row]
                    samples.append(sample)
                except ValueError:
                    continue
    
    return np.array(samples), metadata


def bootstrap_discrepancy(samples, n_boot, confidence=0.95, verbose=False):
    """
    Compute bootstrap confidence intervals for discrepancy.
    
    Args:
        samples: Array of QMC samples (n, d)
        n_boot: Number of bootstrap iterations
        confidence: Confidence level (default: 0.95)
        verbose: Print progress
        
    Returns:
        Dictionary with mean, std, CI bounds, and all bootstrap values
    """
    n, d = samples.shape
    bootstrap_discs = []
    
    if verbose:
        print(f"Running {n_boot} bootstrap iterations...")
    
    for i in range(n_boot):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resampled = samples[indices]
        
        # Compute discrepancy
        try:
            disc = qmc.discrepancy(resampled, method='L2')
        except:
            # Fallback to simple estimate
            disc = np.mean(np.std(resampled, axis=0))
        
        bootstrap_discs.append(disc)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_boot} iterations")
    
    bootstrap_discs = np.array(bootstrap_discs)
    
    # Compute statistics
    mean = np.mean(bootstrap_discs)
    std = np.std(bootstrap_discs)
    
    # Compute confidence interval using percentile method
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_discs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_discs, 100 * (1 - alpha / 2))
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_values': bootstrap_discs
    }


def bootstrap_unique_rate(samples, n_boot, confidence=0.95, precision=12):
    """
    Compute bootstrap confidence intervals for unique sample rate.
    
    Args:
        samples: Array of QMC samples (n, d)
        n_boot: Number of bootstrap iterations
        confidence: Confidence level
        precision: Decimal places for uniqueness check
        
    Returns:
        Dictionary with mean, std, CI bounds
    """
    n, d = samples.shape
    bootstrap_rates = []
    
    for i in range(n_boot):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resampled = samples[indices]
        
        # Compute unique rate
        unique_count = len(np.unique(np.round(resampled, precision), axis=0))
        rate = unique_count / n
        bootstrap_rates.append(rate)
    
    bootstrap_rates = np.array(bootstrap_rates)
    
    # Compute statistics
    mean = np.mean(bootstrap_rates)
    std = np.std(bootstrap_rates)
    
    # Compute confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_rates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_rates, 100 * (1 - alpha / 2))
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def save_metrics(output_path, metadata, disc_results, unique_results, base_metrics):
    """Save metrics to CSV file"""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata
        writer.writerow(['# Bootstrap Confidence Interval Analysis Results'])
        for key, value in metadata.items():
            writer.writerow([f'# {key}: {value}'])
        writer.writerow([])
        
        # Write results
        writer.writerow(['Metric', 'Mean', 'Std', 'CI_Lower', 'CI_Upper'])
        
        writer.writerow([
            'Discrepancy',
            f"{disc_results['mean']:.6f}",
            f"{disc_results['std']:.6f}",
            f"{disc_results['ci_lower']:.6f}",
            f"{disc_results['ci_upper']:.6f}"
        ])
        
        writer.writerow([
            'Unique_Rate',
            f"{unique_results['mean']:.6f}",
            f"{unique_results['std']:.6f}",
            f"{unique_results['ci_lower']:.6f}",
            f"{unique_results['ci_upper']:.6f}"
        ])
        
        if base_metrics.get('mean_kappa') is not None:
            writer.writerow([
                'Mean_Kappa',
                f"{base_metrics['mean_kappa']:.6f}",
                'N/A',
                'N/A',
                'N/A'
            ])
        
        writer.writerow([
            'Savings_Estimate',
            f"{base_metrics['savings_estimate']:.4f}",
            'N/A',
            'N/A',
            'N/A'
        ])


def main():
    """Main function"""
    args = parse_args()
    
    try:
        # Read samples
        if args.verbose:
            print(f"Reading samples from: {args.input}")
        
        samples, metadata = read_samples_csv(args.input)
        
        if args.verbose:
            print(f"  Loaded {len(samples)} samples with {samples.shape[1]} dimensions")
            print(f"  Metadata: {metadata}")
        
        # Compute base metrics
        base_metrics = compute_z_invariant_metrics(samples, method=metadata.get('Method', 'unknown'))
        
        # Bootstrap discrepancy
        if args.verbose:
            print(f"\nBootstrapping discrepancy ({args.n_boot} iterations)...")
        
        disc_results = bootstrap_discrepancy(
            samples, 
            args.n_boot, 
            confidence=args.confidence,
            verbose=args.verbose
        )
        
        # Bootstrap unique rate
        if args.verbose:
            print(f"\nBootstrapping unique rate ({args.n_boot} iterations)...")
        
        unique_results = bootstrap_unique_rate(
            samples,
            args.n_boot,
            confidence=args.confidence
        )
        
        # Save results
        save_metrics(args.output, metadata, disc_results, unique_results, base_metrics)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Bootstrap Analysis Results")
        print(f"{'='*60}")
        print(f"Discrepancy:")
        print(f"  Mean:   {disc_results['mean']:.6f}")
        print(f"  Std:    {disc_results['std']:.6f}")
        print(f"  95% CI: [{disc_results['ci_lower']:.6f}, {disc_results['ci_upper']:.6f}]")
        print(f"\nUnique Rate:")
        print(f"  Mean:   {unique_results['mean']:.6f}")
        print(f"  Std:    {unique_results['std']:.6f}")
        print(f"  95% CI: [{unique_results['ci_lower']:.6f}, {unique_results['ci_upper']:.6f}]")
        print(f"\n✓ Results saved to: {args.output}")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
