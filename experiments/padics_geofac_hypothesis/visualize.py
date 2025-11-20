"""
Visualization Script for P-adic Hypothesis Experiment

Generates plots and visualizations to illustrate:
1. P-adic distance vs traditional distance
2. Ultrametric clustering
3. Descent chain convergence
4. Geofac spine structure
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.padics_geofac_hypothesis.padic import (
    p_adic_distance, p_adic_expansion,
    analyze_geofac_spine, demonstrate_descent_chain,
    compute_cauchy_sequence_convergence
)


def plot_distance_comparison(n_range, p, output_file):
    """
    Compare p-adic distance vs Euclidean distance from reference point.
    """
    reference = 1000
    numbers = list(range(reference - n_range, reference + n_range + 1))
    
    euclidean_dists = [abs(n - reference) for n in numbers]
    padic_dists = [p_adic_distance(n, reference, p) if n != reference else 0 
                   for n in numbers]
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Euclidean distance
    ax1.plot(numbers, euclidean_dists, 'b-', linewidth=2, label='Euclidean distance')
    ax1.set_xlabel('n')
    ax1.set_ylabel('Distance')
    ax1.set_title(f'Euclidean Distance from {reference}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # P-adic distance
    ax2.scatter(numbers, padic_dists, c='r', s=20, alpha=0.6, label=f'{p}-adic distance')
    ax2.set_xlabel('n')
    ax2.set_ylabel(f'{p}-adic Distance')
    ax2.set_title(f'{p}-adic Distance from {reference} (smaller = closer)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved distance comparison to {output_file}")
    plt.close()


def plot_ultrametric_clustering(reference_points, cluster_size, p, output_file):
    """
    Visualize ultrametric clustering around reference points.
    """
    fig, axes = plt.subplots(len(reference_points), 1, 
                             figsize=(12, 4*len(reference_points)))
    
    if len(reference_points) == 1:
        axes = [axes]
    
    for idx, ref in enumerate(reference_points):
        ax = axes[idx]
        
        # Generate cluster around reference
        cluster = list(range(ref - cluster_size, ref + cluster_size + 1))
        distances = [p_adic_distance(n, ref, p) if n != ref else 0 
                    for n in cluster]
        
        # Color by distance (log scale)
        colors = plt.cm.RdYlGn_r(np.log10(np.array(distances) + 1e-10))
        
        ax.scatter(cluster, distances, c=colors, s=50, alpha=0.7, edgecolors='black')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=ref, color='k', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('n')
        ax.set_ylabel(f'{p}-adic Distance from {ref}')
        ax.set_title(f'Ultrametric Clustering around {ref} (p={p})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved ultrametric clustering to {output_file}")
    plt.close()


def plot_descent_convergence(start, p, steps, output_file):
    """
    Plot descent chain convergence in p-adic metric.
    """
    # Generate descent chain
    chain = demonstrate_descent_chain(start, p, steps)
    distances = compute_cauchy_sequence_convergence(chain, p)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sequence values
    ax1.plot(range(len(chain)), chain, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Sequence Value')
    ax1.set_title(f'Descent Chain in {p}-adic Topology\n(starting from {start})')
    ax1.grid(True, alpha=0.3)
    
    # Consecutive distances
    ax2.plot(range(len(distances)), distances, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Step')
    ax2.set_ylabel(f'{p}-adic Distance')
    ax2.set_title(f'Consecutive {p}-adic Distances\n(Cauchy property: distances → 0)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved descent convergence to {output_file}")
    plt.close()


def plot_geofac_spine(n, p, max_level, output_file):
    """
    Visualize the geofac spine structure with p-adic interpretation.
    """
    spine = analyze_geofac_spine(n, p, max_level)
    expansion = p_adic_expansion(n, p, max_level)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Spine residues
    levels = [s[0] for s in spine]
    residues = [s[1] for s in spine]
    
    ax1.bar(levels, residues, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel(f'Level k (mod {p}^k)')
    ax1.set_ylabel(f'Residue of {n}')
    ax1.set_title(f'Geofac Spine for n={n}, p={p}')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # P-adic expansion
    nonzero_indices = [i for i, d in enumerate(expansion) if d != 0]
    nonzero_digits = [expansion[i] for i in nonzero_indices]
    
    ax2.stem(nonzero_indices, nonzero_digits, basefmt=' ', 
             linefmt='red', markerfmt='ro', label='Non-zero digits')
    ax2.set_xlabel('Position i (coefficient of p^i)')
    ax2.set_ylabel('Digit value')
    ax2.set_title(f'{p}-adic Expansion of {n}\nn = Σ aᵢ·{p}^i')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved geofac spine visualization to {output_file}")
    plt.close()


def plot_ultrametric_triangle(a, b, c, p, output_file):
    """
    Visualize the ultrametric triangle inequality.
    """
    dab = p_adic_distance(a, b, p)
    dbc = p_adic_distance(b, c, p)
    dac = p_adic_distance(a, c, p)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw triangle
    points = np.array([[0, 0], [1, 0], [0.5, 0.8]])
    triangle = plt.Polygon(points, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Label vertices
    labels = [f'{a}', f'{b}', f'{c}']
    for i, (x, y) in enumerate(points):
        ax.text(x, y, labels[i], fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black'))
    
    # Label edges with distances
    edge_labels = [
        (0.5, -0.1, f'd({a},{b}) = {dab:.4f}'),
        (0.75, 0.4, f'd({b},{c}) = {dbc:.4f}'),
        (0.25, 0.4, f'd({a},{c}) = {dac:.4f}')
    ]
    
    for x, y, label in edge_labels:
        ax.text(x, y, label, fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Add ultrametric verification
    max_dist = max(dab, dbc)
    is_valid = dac <= max_dist + 1e-10
    
    verification_text = (
        f'\nUltrametric Property:\n'
        f'd({a},{c}) ≤ max(d({a},{b}), d({b},{c}))\n'
        f'{dac:.4f} ≤ max({dab:.4f}, {dbc:.4f})\n'
        f'{dac:.4f} ≤ {max_dist:.4f}\n'
        f'Valid: {"✓ YES" if is_valid else "✗ NO"}'
    )
    
    ax.text(0.5, -0.4, verification_text, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if is_valid else 'lightcoral',
                     edgecolor='black', linewidth=2))
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.6, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'{p}-adic Ultrametric Triangle\n(Strong Triangle Inequality)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved ultrametric triangle to {output_file}")
    plt.close()


def main():
    """Generate all visualizations."""
    output_dir = os.path.dirname(__file__)
    
    print("\n" + "="*70)
    print("Generating P-adic Hypothesis Visualizations")
    print("="*70 + "\n")
    
    # 1. Distance comparison
    plot_distance_comparison(
        n_range=50,
        p=2,
        output_file=os.path.join(output_dir, 'viz_distance_comparison.png')
    )
    
    # 2. Ultrametric clustering
    plot_ultrametric_clustering(
        reference_points=[1000, 2000],
        cluster_size=20,
        p=2,
        output_file=os.path.join(output_dir, 'viz_clustering.png')
    )
    
    # 3. Descent convergence (2-adic)
    plot_descent_convergence(
        start=1000,
        p=2,
        steps=20,
        output_file=os.path.join(output_dir, 'viz_descent_2adic.png')
    )
    
    # 4. Descent convergence (5-adic)
    plot_descent_convergence(
        start=1000,
        p=5,
        steps=15,
        output_file=os.path.join(output_dir, 'viz_descent_5adic.png')
    )
    
    # 5. Geofac spine for 2024
    plot_geofac_spine(
        n=2024,
        p=2,
        max_level=15,
        output_file=os.path.join(output_dir, 'viz_spine_2024.png')
    )
    
    # 6. Geofac spine for 899
    plot_geofac_spine(
        n=899,
        p=29,
        max_level=5,
        output_file=os.path.join(output_dir, 'viz_spine_899.png')
    )
    
    # 7. Ultrametric triangle
    plot_ultrametric_triangle(
        a=1000,
        b=1024,
        c=1040,
        p=2,
        output_file=os.path.join(output_dir, 'viz_ultrametric_triangle.png')
    )
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
