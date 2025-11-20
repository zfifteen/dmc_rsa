# Quick Start Guide: Arctan-Refined Korobov Lattice Experiment

## TL;DR

This experiment **falsified** the hypothesis that arctan-refined curvature improves Korobov lattice QMC performance by 10-30%.

**Result:** Mean improvement was only **0.61%** (95% CI: [-0.92%, 2.90%]), far below claimed 10-30%.

---

## Run the Experiment

### Prerequisites

```bash
pip install numpy scipy sympy matplotlib
```

### Quick Test (2 minutes)

```bash
cd experiments/arctan_korobov_qmc
python run_experiment.py --quick
```

This runs:
- 20 trials per test (vs 100 in full version)
- 100 bootstrap resamples (vs 1000 in full version)
- 8 test configurations

### Full Experiment (10-15 minutes)

```bash
python run_experiment.py --trials 100 --bootstrap 1000
```

### Generate Plots

```bash
python visualize_results.py
```

Output in `plots/`:
- `variance_reduction_by_test.png` - Results by test function
- `variance_reduction_by_alpha.png` - Performance vs α parameter
- `confidence_interval.png` - Bootstrap CI
- `histogram_variance_reductions.png` - Distribution

---

## Understanding the Code

### Core Modules

1. **arctan_curvature.py**
   - `arctan_refinement(n)` - Computes arctan(φ · frac(n/φ))
   - `kappa_arctan(n, α)` - Refined curvature κ(n) + α·arctan(...)
   - `generate_korobov_lattice(n, d, use_arctan)` - Lattice generator

2. **qmc_integration_tests.py**
   - Test functions: ProductCosine, SmoothPeriodic, MultiFrequency, Genz
   - `run_integration_comparison()` - Compare baseline vs arctan-refined
   - `compute_variance_reduction()` - Calculate improvement metrics

3. **run_experiment.py**
   - Main experiment runner
   - Bootstrap confidence intervals
   - Statistical significance testing
   - JSON output with all results

### Quick Code Example

```python
from arctan_curvature import generate_korobov_lattice, measure_lattice_quality

# Generate baseline Korobov lattice
points_baseline = generate_korobov_lattice(n=127, d=2, use_arctan=False)
metrics_baseline = measure_lattice_quality(points_baseline)
print(f"Baseline L2 discrepancy: {metrics_baseline['l2_discrepancy']:.6f}")

# Generate arctan-refined lattice
points_arctan = generate_korobov_lattice(n=127, d=2, use_arctan=True, alpha=1.0)
metrics_arctan = measure_lattice_quality(points_arctan)
print(f"Arctan L2 discrepancy: {metrics_arctan['l2_discrepancy']:.6f}")

# Compare
improvement = (metrics_baseline['l2_discrepancy'] - metrics_arctan['l2_discrepancy']) / metrics_baseline['l2_discrepancy'] * 100
print(f"Improvement: {improvement:+.2f}%")
```

---

## Key Results at a Glance

### By Test Function (α=1.0)

| Test | Var. Reduction | Status |
|------|----------------|--------|
| ProductCosine-2D-127 | 0.00% | No effect |
| ProductCosine-2D-251 | 0.00% | No effect |
| SmoothPeriodic-2D-127 | -2.07% | Degraded |
| SmoothPeriodic-3D-127 | -1.07% | Degraded |
| GenzContinuous-2D-127 | 8.15% | Best case |

### Summary Statistics

- **Mean:** 0.61% (claimed: 10-30%)
- **95% CI:** [-0.92%, 2.90%] (includes zero)
- **Verdict:** **Hypothesis falsified**

---

## Extending the Experiment

### Test New α Values

Edit `run_experiment.py`, line ~115:

```python
alpha_values = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]  # Add your values
```

### Add New Test Functions

In `qmc_integration_tests.py`, create a new class:

```python
class MyTestFunction(PeriodicTestFunction):
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.name = "MyFunction"
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # Your function here
        return np.sum(x, axis=1)
    
    def analytical_integral(self) -> float:
        return 0.5  # If known
```

Then add to `run_experiment.py`, line ~105:

```python
test_configs = [
    # ... existing configs ...
    (MyTestFunction, 2, 127, "MyFunction-2D-127pts"),
]
```

### Test Larger Lattices

```python
# Edit test_configs in run_experiment.py
test_configs = [
    (ProductCosine, 2, 509, "ProductCosine-2D-509pts"),  # Larger prime
    (ProductCosine, 5, 127, "ProductCosine-5D-127pts"),  # Higher dimension
]
```

### Different Generator Selection

In `arctan_curvature.py`, line ~124, modify:

```python
# Current: select minimum curvature
optimal_idx = np.argmin(curvatures)

# Try: select maximum curvature
optimal_idx = np.argmax(curvatures)

# Try: select median
optimal_idx = len(curvatures) // 2
```

---

## Understanding the Math

### Baseline Curvature

```
κ(n) = d(n) · ln(n+1) / e²
```

where:
- d(n) = number of divisors of n
- e² ≈ 7.389

### Arctan Refinement

```
κ_arctan(n) = κ(n) + α · arctan(φ · frac(n/φ))
```

where:
- φ = (1 + √5) / 2 ≈ 1.618 (golden ratio)
- frac(x) = x - floor(x) (fractional part)
- α = scaling parameter (tested: 0.5, 1.0, 1.5, 2.0)

### Korobov Lattice

Rank-1 lattice with generating vector:

```
z = (1, a, a², ..., a^(d-1)) mod n
```

Points:

```
x_i = (i · z / n) mod 1,  i = 0, ..., n-1
```

The hypothesis was: use κ_arctan to select optimal 'a'.

---

## Troubleshooting

### "No module named numpy"

```bash
pip install numpy scipy sympy
```

### "ModuleNotFoundError: cognitive_number_theory"

Run from repository root:

```bash
cd /path/to/dmc_rsa
python experiments/arctan_korobov_qmc/run_experiment.py
```

### Experiment too slow

Use quick mode:

```bash
python run_experiment.py --quick
```

Or reduce trials:

```bash
python run_experiment.py --trials 20 --bootstrap 100
```

### Want raw data

Check `data/results_*.json` for complete results in JSON format.

---

## Files Overview

```
experiments/arctan_korobov_qmc/
├── README.md                   # Comprehensive documentation
├── FINAL_REPORT.md            # Scientific report
├── EXPERIMENT_SUMMARY.txt     # Executive summary
├── QUICKSTART.md              # This file
├── arctan_curvature.py        # Core implementation
├── qmc_integration_tests.py   # Test suite
├── run_experiment.py          # Main experiment
├── visualize_results.py       # Plotting
├── data/
│   └── results_*.json         # Experiment results
├── plots/
│   ├── variance_reduction_by_test.png
│   ├── variance_reduction_by_alpha.png
│   ├── confidence_interval.png
│   └── histogram_variance_reductions.png
└── logs/
    └── experiment_run.log     # Full execution log
```

---

## Citation

If using this experiment in your research:

```
Arctan-Refined Curvature in Korobov Lattices for QMC: 
Hypothesis Falsification Study
Z-Framework Validation Experiment
github.com/zfifteen/dmc_rsa/experiments/arctan_korobov_qmc
November 2025
```

---

## Questions?

- 📖 See `README.md` for detailed methodology
- 📊 See `FINAL_REPORT.md` for scientific analysis
- 📂 Check `data/` for raw results
- 🖼️ View `plots/` for visualizations

**Key Takeaway:** Arctan refinement does **not** improve Korobov lattice QMC performance as claimed. Use baseline κ(n) = d(n)·ln(n+1)/e² instead.
