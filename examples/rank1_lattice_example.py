#!/usr/bin/env python3
"""
Example: Using Rank-1 Lattices for RSA Factorization
Demonstrates the new subgroup-based rank-1 lattice constructions
"""

import sys
sys.path.append('scripts')

import numpy as np
from qmc_engines import QMCConfig, make_engine, map_points_to_candidates
from rank1_lattice import compute_lattice_quality_metrics

def example_basic_usage():
    """Basic example: Generate rank-1 lattice points"""
    print("="*70)
    print("Example 1: Basic Rank-1 Lattice Generation")
    print("="*70)
    
    # Create configuration for cyclic subgroup construction
    cfg = QMCConfig(
        dim=2,
        n=128,
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=16,
        scramble=True,
        seed=42
    )
    
    # Create engine and generate points
    engine = make_engine(cfg)
    points = engine.random(128)
    
    print(f"\nGenerated {len(points)} lattice points in {points.shape[1]}D space")
    print(f"Point range: [{points.min():.4f}, {points.max():.4f}]")
    
    # Compute quality metrics
    metrics = compute_lattice_quality_metrics(points)
    print(f"\nQuality Metrics:")
    print(f"  Minimum distance: {metrics['min_distance']:.4f}")
    print(f"  Covering radius:  {metrics['covering_radius']:.4f}")
    
    return points


def example_rsa_factorization():
    """Example: Apply to RSA factorization"""
    print("\n\n" + "="*70)
    print("Example 2: RSA Factorization with Rank-1 Lattices")
    print("="*70)
    
    # RSA semiprime
    N = 899  # 29 × 31
    p, q = 29, 31
    phi_N = (p-1) * (q-1)  # 840
    
    print(f"\nTarget: N = {N} = {p} × {q}")
    print(f"φ(N) = {phi_N}")
    
    # Use subgroup order that divides φ(N)
    subgroup_order = 20  # Divides 840
    
    cfg = QMCConfig(
        dim=2,
        n=256,  # Power of 2
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=subgroup_order,
        scramble=True,
        seed=42
    )
    
    print(f"\nConfiguration:")
    print(f"  Lattice size: {cfg.n}")
    print(f"  Generator: {cfg.lattice_generator}")
    print(f"  Subgroup order: {subgroup_order}")
    
    # Generate lattice points
    engine = make_engine(cfg)
    points = engine.random(256)
    
    # Map to RSA candidate factors
    window_radius = 10
    candidates = map_points_to_candidates(points, N, window_radius)
    unique_candidates = np.unique(candidates)
    
    print(f"\nGenerated {len(unique_candidates)} unique candidates")
    
    # Find factors
    hits = [c for c in unique_candidates if N % c == 0 and c > 1 and c < N]
    
    print(f"\n✓ Found {len(hits)} factors: {set(hits)}")
    
    if hits:
        for factor in sorted(set(hits)):
            other = N // factor
            print(f"  {N} = {factor} × {other}")
    
    return unique_candidates, hits


def example_compare_generators():
    """Example: Compare different lattice generators"""
    print("\n\n" + "="*70)
    print("Example 3: Comparing Lattice Generator Types")
    print("="*70)
    
    N = 899
    window_radius = 10
    
    generator_types = [
        ("fibonacci", None, "Golden ratio-based"),
        ("korobov", None, "Primitive root-based"),
        ("cyclic", 20, "Group-theoretic (subgroup order 20)")
    ]
    
    results = []
    
    for gen_type, subgroup_order, description in generator_types:
        print(f"\n{gen_type.upper()} - {description}")
        print("-" * 50)
        
        cfg = QMCConfig(
            dim=2,
            n=128,
            engine="rank1_lattice",
            lattice_generator=gen_type,
            subgroup_order=subgroup_order,
            scramble=True,
            seed=42
        )
        
        engine = make_engine(cfg)
        points = engine.random(128)
        
        # Compute metrics
        metrics = compute_lattice_quality_metrics(points)
        
        # Map to candidates
        candidates = map_points_to_candidates(points, N, window_radius)
        unique_candidates = np.unique(candidates)
        hits = [c for c in unique_candidates if N % c == 0 and c > 1 and c < N]
        
        print(f"  Unique candidates: {len(unique_candidates)}")
        print(f"  Factors found:     {len(hits)}")
        print(f"  Min distance:      {metrics['min_distance']:.4f}")
        print(f"  Covering radius:   {metrics['covering_radius']:.4f}")
        
        results.append({
            'type': gen_type,
            'unique': len(unique_candidates),
            'hits': len(hits),
            'min_dist': metrics['min_distance'],
            'cov_rad': metrics['covering_radius']
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Generator':<15} {'Candidates':<12} {'Hits':<8} {'Min Dist':<10} {'Cov Rad':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['type']:<15} {r['unique']:<12} {r['hits']:<8} {r['min_dist']:<10.4f} {r['cov_rad']:<10.4f}")
    
    return results


def example_replicated_analysis():
    """Example: Replicated analysis for confidence intervals"""
    print("\n\n" + "="*70)
    print("Example 4: Replicated Analysis with Confidence Intervals")
    print("="*70)
    
    from qmc_engines import qmc_points
    
    N = 899
    window_radius = 10
    num_replicates = 8
    
    cfg = QMCConfig(
        dim=2,
        n=128,
        engine="rank1_lattice",
        lattice_generator="cyclic",
        subgroup_order=20,
        scramble=True,
        seed=42,
        replicates=num_replicates
    )
    
    print(f"\nRunning {num_replicates} replicates...")
    
    unique_counts = []
    hit_counts = []
    
    for i, points in enumerate(qmc_points(cfg)):
        candidates = map_points_to_candidates(points, N, window_radius)
        unique_candidates = np.unique(candidates)
        hits = [c for c in unique_candidates if N % c == 0 and c > 1 and c < N]
        
        unique_counts.append(len(unique_candidates))
        hit_counts.append(len(hits))
        
        print(f"  Replicate {i+1}: {len(unique_candidates)} candidates, {len(hits)} factors")
    
    # Calculate statistics
    mean_unique = np.mean(unique_counts)
    std_unique = np.std(unique_counts, ddof=1)
    ci_lower = mean_unique - 1.96 * std_unique / np.sqrt(num_replicates)
    ci_upper = mean_unique + 1.96 * std_unique / np.sqrt(num_replicates)
    
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"\nUnique Candidates:")
    print(f"  Mean: {mean_unique:.2f}")
    print(f"  Std:  {std_unique:.2f}")
    print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"\nFactors Found:")
    print(f"  Mean: {np.mean(hit_counts):.2f}")
    print(f"  Consistency: {len([h for h in hit_counts if h > 0])}/{num_replicates} replicates found factors")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("RANK-1 LATTICE EXAMPLES FOR RSA FACTORIZATION")
    print("="*70)
    
    # Run examples
    example_basic_usage()
    example_rsa_factorization()
    example_compare_generators()
    example_replicated_analysis()
    
    print("\n\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Rank-1 lattices provide theoretically-motivated point distributions")
    print("2. Cyclic subgroup construction leverages RSA algebraic structure")
    print("3. Quality metrics validate enhanced regularity properties")
    print("4. Replicated randomization enables statistical inference")
    print("5. Multiple generator types available for different applications")
    print("\nFor more details, see docs/RANK1_LATTICE_INTEGRATION.md")
    print("="*70)


if __name__ == "__main__":
    main()
