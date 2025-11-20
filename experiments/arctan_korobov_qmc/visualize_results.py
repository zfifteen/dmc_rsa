#!/usr/bin/env python3
"""
Visualization Script for Arctan-Refined Korobov Lattice Experiment
===================================================================

Generate plots and visualizations of experimental results.

Author: Z-Mode experiment framework
Date: November 2025
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_latest_results() -> dict:
    """Load the most recent results JSON file"""
    data_dir = Path(__file__).parent / 'data'
    json_files = sorted(data_dir.glob('results_*.json'), reverse=True)
    
    if not json_files:
        raise FileNotFoundError("No results files found in data/")
    
    latest = json_files[0]
    print(f"Loading results from: {latest.name}")
    
    with open(latest, 'r') as f:
        return json.load(f)


def plot_variance_reduction_by_test(results: dict, output_path: str):
    """
    Plot variance reduction for each test function at α=1.0
    """
    experiments = results['experiments']
    
    test_names = []
    var_reductions = []
    
    for exp in experiments:
        test_names.append(exp['description'].replace('-', '\n'))
        var_reductions.append(exp['variance_reductions'].get(1.0, 0.0))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color bars based on positive/negative
    colors = ['green' if v > 0 else 'red' for v in var_reductions]
    
    bars = ax.bar(range(len(test_names)), var_reductions, color=colors, alpha=0.7, edgecolor='black')
    
    # Add horizontal lines for claimed range
    ax.axhline(y=10, color='blue', linestyle='--', linewidth=2, label='Claimed Min (10%)')
    ax.axhline(y=30, color='blue', linestyle='--', linewidth=2, label='Claimed Max (30%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Test Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Reduction by Test Function (α=1.0)\nHypothesis: 10-30% reduction', 
                fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(len(test_names)))
    ax.set_xticklabels(test_names, fontsize=9, rotation=0)
    
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, var_reductions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_variance_reduction_by_alpha(results: dict, output_path: str):
    """
    Plot variance reduction across different α values
    """
    experiments = results['experiments']
    
    # Aggregate by alpha
    alphas = [0.5, 1.0, 1.5, 2.0]
    var_red_by_alpha = {alpha: [] for alpha in alphas}
    
    for exp in experiments:
        for alpha in alphas:
            if alpha in exp['variance_reductions']:
                var_red_by_alpha[alpha].append(exp['variance_reductions'][alpha])
    
    # Compute statistics
    means = []
    stds = []
    for alpha in alphas:
        values = var_red_by_alpha[alpha]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(alphas, means, yerr=stds, marker='o', markersize=10, 
               linewidth=2, capsize=5, capthick=2, label='Mean ± Std')
    
    # Add claimed range
    ax.axhline(y=10, color='blue', linestyle='--', linewidth=2, label='Claimed Min (10%)')
    ax.axhline(y=30, color='blue', linestyle='--', linewidth=2, label='Claimed Max (30%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and title
    ax.set_xlabel('α Parameter (Arctan Scaling)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Reduction vs α Parameter\nAggregated Across All Tests', 
                fontsize=14, fontweight='bold')
    
    ax.set_xticks(alphas)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_confidence_interval(results: dict, output_path: str):
    """
    Plot confidence interval for α=1.0
    """
    if 'summary' not in results or 'alpha_1.0' not in results['summary']:
        print("Warning: Summary data not available")
        return
    
    summary = results['summary']['alpha_1.0']
    
    mean = summary['mean_variance_reduction']
    ci_lower = summary['ci_lower']
    ci_upper = summary['ci_upper']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot confidence interval
    ax.plot([1, 1], [ci_lower, ci_upper], 'o-', linewidth=3, markersize=10, 
           color='darkgreen', label='95% CI')
    ax.plot(1, mean, 'o', markersize=15, color='darkred', label=f'Mean: {mean:.2f}%')
    
    # Add claimed range
    ax.axhspan(10, 30, alpha=0.2, color='blue', label='Claimed Range (10-30%)')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Zero Effect')
    
    # Labels
    ax.set_ylabel('Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('95% Bootstrap Confidence Interval (α=1.0)\nAcross All Experiments', 
                fontsize=14, fontweight='bold')
    
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels(['α=1.0'], fontsize=12)
    
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add text annotation
    annotation = f"Mean: {mean:.2f}%\nCI: [{ci_lower:.2f}%, {ci_upper:.2f}%]\n"
    if summary['ci_overlaps_zero']:
        annotation += "⚠ CI includes zero\n(not significant)"
    if not summary['in_claimed_range']:
        annotation += "\n❌ Outside claimed range"
    
    ax.text(1.3, mean, annotation, fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_histogram_variance_reductions(results: dict, output_path: str):
    """
    Plot histogram of variance reductions at α=1.0
    """
    experiments = results['experiments']
    
    var_reductions = []
    for exp in experiments:
        if 1.0 in exp['variance_reductions']:
            var_reductions.append(exp['variance_reductions'][1.0])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins, patches = ax.hist(var_reductions, bins=10, edgecolor='black', 
                               alpha=0.7, color='steelblue')
    
    # Color bars based on claimed range
    for i, patch in enumerate(patches):
        if bins[i] >= 10 and bins[i+1] <= 30:
            patch.set_facecolor('green')
        elif bins[i] < 0:
            patch.set_facecolor('red')
    
    # Add vertical lines
    mean_val = np.mean(var_reductions)
    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.2f}%')
    ax.axvline(10, color='blue', linestyle='--', linewidth=2, label='Claimed Min (10%)')
    ax.axvline(30, color='blue', linestyle='--', linewidth=2, label='Claimed Max (30%)')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    # Labels
    ax.set_xlabel('Variance Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Variance Reductions (α=1.0)\nAcross All Experiments', 
                fontsize=14, fontweight='bold')
    
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    """Generate all plots"""
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    # Load results
    results = load_latest_results()
    
    # Create plots directory
    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_variance_reduction_by_test(
        results, 
        str(plots_dir / 'variance_reduction_by_test.png')
    )
    
    plot_variance_reduction_by_alpha(
        results, 
        str(plots_dir / 'variance_reduction_by_alpha.png')
    )
    
    plot_confidence_interval(
        results, 
        str(plots_dir / 'confidence_interval.png')
    )
    
    plot_histogram_variance_reductions(
        results, 
        str(plots_dir / 'histogram_variance_reductions.png')
    )
    
    print("\n" + "=" * 60)
    print("Visualization Complete")
    print(f"Plots saved to: {plots_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
