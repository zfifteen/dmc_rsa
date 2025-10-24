#!/usr/bin/env python3

import sys
sys.path.append('scripts')

from qmc_factorization_analysis import QMCFactorization
import pandas as pd

n = 1000000
print(f"Testing N={n}")

df_large = QMCFactorization.run_statistical_analysis(n, num_samples=200, num_trials=10)  # Very few trials for test

# Calculate improvements
mc_base = df_large[df_large['method'] == 'MC']['unique_count_mean'].values[0]
for _, row in df_large.iterrows():
    improvement = row['unique_count_mean'] / mc_base
    print(f"  {row['method']}: {row['unique_count_mean']:.1f} unique "
          f"[{row['unique_count_ci_lower']:.1f}, {row['unique_count_ci_upper']:.1f}], "
          f"{improvement:.2f}× vs MC")

print("Done")