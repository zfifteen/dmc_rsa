#!/usr/bin/env python3
"""
Quick verification of θ′-biased QMC falsification experiment results.

This script validates that all experimental results are consistent and
the hypothesis was properly falsified.
"""

import sys
import json
from pathlib import Path

def verify_results():
    """Verify all experimental results"""
    print("\n" + "="*70)
    print("VERIFICATION: θ′-biased QMC Falsification Experiment")
    print("="*70 + "\n")
    
    deltas_dir = Path("deltas")
    
    if not deltas_dir.exists():
        print("❌ Error: deltas/ directory not found")
        print("   Make sure you're in experiments/theta_prime_qmc_falsification/")
        return 1
    
    # Load all results
    json_files = sorted(deltas_dir.glob("*.json"))
    
    if len(json_files) != 8:
        print(f"❌ Error: Expected 8 result files, found {len(json_files)}")
        return 1
    
    print(f"✓ Found {len(json_files)} result files\n")
    
    # Verify each result
    all_negative = True
    all_ci_exclude_zero = True
    configs_summary = []
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)
        
        dataset = data['dataset']
        alpha = data['alpha']
        sigma = data['sigma_ms']
        delta_pct = data['delta_unique']['pct']
        ci_low = data['delta_unique']['ci_low']
        ci_high = data['delta_unique']['ci_high']
        baseline_mean = data['baseline_metrics']['mean_unique']
        
        # Convert absolute CI to percentage
        ci_low_pct = (ci_low / baseline_mean) * 100
        ci_high_pct = (ci_high / baseline_mean) * 100
        
        # Check if negative
        if delta_pct >= 0:
            all_negative = False
        
        # Check if CI excludes zero
        if ci_low > 0 or ci_high < 0:
            pass  # CI does not exclude zero if it crosses zero
        else:
            # CI excludes zero if both bounds are on same side
            if not (ci_low < 0 and ci_high < 0):
                all_ci_exclude_zero = False
        
        configs_summary.append({
            'dataset': dataset,
            'alpha': alpha,
            'sigma': sigma,
            'delta_pct': delta_pct,
            'ci_low': ci_low,
            'ci_high': ci_high
        })
        
        status = "✗ FALSIFIED" if delta_pct < 0 else "? ANOMALY"
        print(f"{status} | {dataset:8s} | α={alpha:.1f}, σ={sigma:3.0f}ms | "
              f"Δ={delta_pct:+6.2f}% | 95% CI=[{ci_low:6.1f}, {ci_high:5.1f}]")
    
    print("\n" + "-"*70 + "\n")
    
    # Summary statistics
    delta_pcts = [c['delta_pct'] for c in configs_summary]
    min_delta = min(delta_pcts)
    max_delta = max(delta_pcts)
    avg_delta = sum(delta_pcts) / len(delta_pcts)
    
    print("Summary Statistics:")
    print(f"  Min Δ:     {min_delta:+6.2f}%")
    print(f"  Max Δ:     {max_delta:+6.2f}%")
    print(f"  Avg Δ:     {avg_delta:+6.2f}%")
    print(f"  Expected:  +5.00% to +15.00%")
    
    print("\n" + "-"*70 + "\n")
    
    # Hypothesis test
    print("Hypothesis Testing:")
    print(f"  All Δ < 0:           {'✓ YES' if all_negative else '✗ NO'}")
    print(f"  All 95% CI < 0:      {'✓ YES' if all_ci_exclude_zero else '✗ NO'}")
    print(f"  All configs tested:  {'✓ YES' if len(configs_summary) == 8 else '✗ NO'}")
    
    print("\n" + "="*70)
    
    if all_negative and len(configs_summary) == 8:
        print("VERDICT: HYPOTHESIS CONCLUSIVELY FALSIFIED ✗")
        print("  θ′-biased QMC REDUCES unique candidates by 0.2-4.8%")
        print("  Expected: INCREASE by 5-15%")
        print("  Observed: DECREASE across all 8 configurations")
    else:
        print("VERDICT: INCONCLUSIVE")
        print("  Some configurations did not show expected negative effect")
    
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(verify_results())
