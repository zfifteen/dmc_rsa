#!/usr/bin/env python3
"""
Generate visualization plots for θ′-biased QMC falsification experiment
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

def load_results():
    """Load all experimental results"""
    deltas_dir = Path("deltas")
    
    results = []
    for json_file in sorted(deltas_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    
    return results


def plot_unique_candidates_comparison(results, output_path):
    """Plot unique candidates: baseline vs policy"""
    if not HAS_MATPLOTLIB:
        return
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by dataset
    rsa129 = [r for r in results if r['dataset'] == 'rsa-129']
    rsa155 = [r for r in results if r['dataset'] == 'rsa-155']
    
    def plot_dataset(data, ax, title):
        configs = []
        baseline_vals = []
        policy_vals = []
        colors = []
        
        for r in data:
            label = f"α={r['alpha']}, σ={r['sigma_ms']}"
            configs.append(label)
            baseline_vals.append(r['baseline_metrics']['mean_unique'])
            policy_vals.append(r['policy_metrics']['mean_unique'])
            
            # Color based on alpha
            if r['alpha'] == 0.1:
                colors.append('#3498db')  # Blue for α=0.1
            else:
                colors.append('#e74c3c')  # Red for α=0.2
        
        x = np.arange(len(configs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (MC)', 
                      color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, policy_vals, width, label='Policy (QMC+θ′)',
                      color=colors, alpha=0.8)
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unique Candidates', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    plot_dataset(rsa129, ax1, 'RSA-129 (N=899)')
    plot_dataset(rsa155, ax2, 'RSA-155 (N=10403)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_delta_with_ci(results, output_path):
    """Plot delta percentage with 95% CI error bars"""
    if not HAS_MATPLOTLIB:
        return
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by dataset
    rsa129 = [r for r in results if r['dataset'] == 'rsa-129']
    rsa155 = [r for r in results if r['dataset'] == 'rsa-155']
    
    def plot_dataset(data, ax, title):
        configs = []
        delta_pcts = []
        ci_lows = []
        ci_highs = []
        colors = []
        
        for r in data:
            label = f"α={r['alpha']}\nσ={r['sigma_ms']}"
            configs.append(label)
            
            # Get delta percentage
            delta_pct = r['delta_unique']['pct']
            delta_pcts.append(delta_pct)
            
            # Compute CI width
            baseline_mean = r['baseline_metrics']['mean_unique']
            ci_low_pct = (r['delta_unique']['ci_low'] / baseline_mean) * 100
            ci_high_pct = (r['delta_unique']['ci_high'] / baseline_mean) * 100
            
            ci_lows.append(delta_pct - ci_low_pct)
            ci_highs.append(ci_high_pct - delta_pct)
            
            # Color based on result
            if delta_pct >= 5:
                colors.append('#2ecc71')  # Green for confirmed
            elif delta_pct > 0:
                colors.append('#f39c12')  # Orange for positive but small
            else:
                colors.append('#e74c3c')  # Red for negative
        
        x = np.arange(len(configs))
        
        bars = ax.bar(x, delta_pcts, color=colors, alpha=0.8, 
                     yerr=[ci_lows, ci_highs], capsize=5, error_kw={'linewidth': 2})
        
        # Add hypothesis region
        ax.axhspan(5, 15, alpha=0.2, color='green', label='Hypothesis Range (5-15%)')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='No Effect')
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Δ Unique % (Policy - Baseline)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, delta_pcts)):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.2f}%',
                   ha='center', va='bottom' if val >= 0 else 'top', 
                   fontsize=10, fontweight='bold')
    
    plot_dataset(rsa129, ax1, 'RSA-129: Delta with 95% CI')
    plot_dataset(rsa155, ax2, 'RSA-155: Delta with 95% CI')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_discrepancy_comparison(results, output_path):
    """Plot L2 discrepancy: baseline vs policy"""
    if not HAS_MATPLOTLIB:
        return
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Group by dataset
    rsa129 = [r for r in results if r['dataset'] == 'rsa-129']
    rsa155 = [r for r in results if r['dataset'] == 'rsa-155']
    
    def plot_dataset(data, ax, title):
        configs = []
        baseline_disc = []
        policy_disc = []
        
        for r in data:
            label = f"α={r['alpha']}, σ={r['sigma_ms']}"
            configs.append(label)
            baseline_disc.append(r['z_metrics_baseline']['discrepancy'])
            policy_disc.append(r['z_metrics_policy']['discrepancy'])
        
        x = np.arange(len(configs))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_disc, width, label='Baseline (MC)',
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, policy_disc, width, label='Policy (QMC+θ′)',
                      color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('L2 Discrepancy (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plot_dataset(rsa129, ax1, 'RSA-129: Discrepancy Comparison')
    plot_dataset(rsa155, ax2, 'RSA-155: Discrepancy Comparison')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_summary_heatmap(results, output_path):
    """Create summary heatmap of all configurations"""
    if not HAS_MATPLOTLIB:
        return
    
    # Organize data as matrix
    alphas = sorted(list(set(r['alpha'] for r in results)))
    sigmas = sorted(list(set(r['sigma_ms'] for r in results)))
    datasets = sorted(list(set(r['dataset'] for r in results)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        
        # Create matrix
        matrix = np.zeros((len(alphas), len(sigmas)))
        for i, alpha in enumerate(alphas):
            for j, sigma in enumerate(sigmas):
                # Find matching result
                match = [r for r in results if r['dataset'] == dataset and 
                        r['alpha'] == alpha and r['sigma_ms'] == sigma]
                if match:
                    matrix[i, j] = match[0]['delta_unique']['pct']
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-6, vmax=6)
        
        # Set ticks
        ax.set_xticks(np.arange(len(sigmas)))
        ax.set_yticks(np.arange(len(alphas)))
        ax.set_xticklabels([f'{s}ms' for s in sigmas])
        ax.set_yticklabels([f'α={a}' for a in alphas])
        
        ax.set_xlabel('Drift σ', fontsize=12, fontweight='bold')
        ax.set_ylabel('Bias α', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset.upper()}: Δ Unique %', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(alphas)):
            for j in range(len(sigmas)):
                ax.text(j, i, f'{matrix[i, j]:.2f}%',
                       ha="center", va="center", color="black", 
                       fontsize=11, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Δ Unique % (Policy - Baseline)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    """Main visualization generation"""
    print("\nGenerating Visualizations...")
    print("=" * 70)
    
    # Load results
    results = load_results()
    print(f"Loaded {len(results)} experimental results")
    
    if not HAS_MATPLOTLIB:
        print("\nSkipping visualizations (matplotlib not installed)")
        print("Install with: pip install matplotlib")
        return 0
    
    # Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_unique_candidates_comparison(results, "plots/unique_candidates_comparison.png")
    plot_delta_with_ci(results, "plots/delta_with_confidence_intervals.png")
    plot_discrepancy_comparison(results, "plots/discrepancy_comparison.png")
    plot_summary_heatmap(results, "plots/summary_heatmap.png")
    
    print("\n" + "=" * 70)
    print("Visualization generation complete!")
    print("\nGenerated plots:")
    print("  - plots/unique_candidates_comparison.png")
    print("  - plots/delta_with_confidence_intervals.png")
    print("  - plots/discrepancy_comparison.png")
    print("  - plots/summary_heatmap.png")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
