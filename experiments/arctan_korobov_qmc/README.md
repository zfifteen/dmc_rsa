# Arctan-Refined Curvature in Korobov Lattices for QMC

## Hypothesis Falsification Experiment

**Experiment Date:** November 20, 2025  
**Framework:** Z-Mode experiment framework  
**Repository:** zfifteen/dmc_rsa

---

## Executive Summary

### Hypothesis Statement

The hypothesis claimed that:

> "Augmenting κ(n) with arctan(φ · frac(n/φ)) terms enhances Korobov lattice parameter tuning, achieving **10-30% variance cuts** in QMC for periodic integrands via golden-ratio equidistribution."

Where:
- κ(n) = d(n) · ln(n+1) / e² (baseline divisor-weighted curvature)
- φ = (1 + √5) / 2 (golden ratio)
- frac(x) = x - floor(x) (fractional part)

### Verdict: **HYPOTHESIS FALSIFIED** ❌

**Key Findings:**
- **Mean variance reduction (α=1.0): 0.61%** (far below claimed 10-30%)
- **95% Confidence Interval: [-0.92%, 2.90%]** (overlaps zero, not statistically significant)
- **Range across experiments: [-2.07%, 8.15%]**

**Reasons for Falsification:**
1. Mean reduction 0.61% is **outside** claimed [10%, 30%] range
2. Confidence interval **includes zero** (not statistically significant)
3. In 4 out of 8 experiments, arctan refinement **degraded** performance (negative variance reduction)
4. Best case improvement (8.15% in Genz function) still **below** minimum claimed 10%

---

## Experimental Design

### Methodology

1. **Test Functions:** Standard QMC benchmark integrands
   - ProductCosine: ∏ᵢ cos(2πxᵢ) - highly periodic
   - SmoothPeriodic: ∏ᵢ (1 + 0.5 sin(2πxᵢ)) - smooth periodic
   - MultiFrequencyCosine: cos(2π Σᵢ wᵢxᵢ) - mixed frequencies
   - GenzContinuous: exp(-Σᵢ cᵢ|xᵢ - wᵢ|) - standard benchmark

2. **Dimensions:** 2D, 3D

3. **Lattice Sizes:** 127, 251 points (prime numbers for optimal Korobov properties)

4. **Statistical Rigor:**
   - 100 independent trials per test
   - 1000 bootstrap resamples for confidence intervals
   - Cranley-Patterson randomization for independent trials
   - 95% confidence intervals

5. **Comparison Points:**
   - Baseline Korobov: α = 0.0 (no arctan refinement)
   - Arctan-refined: α = 0.5, 1.0, 1.5, 2.0

### Implementation

The arctan-refined curvature is defined as:

```python
κ_arctan(n) = κ(n) + α · arctan(φ · frac(n/φ))
```

This is used to select the Korobov generator 'a' by:
1. Computing κ_arctan(a) for all valid generators a coprime to n
2. Selecting generator with minimum curvature
3. Generating lattice vector z = (1, a, a², ..., a^(d-1)) mod n

---

## Detailed Results

### Summary Statistics (α=1.0)

| Metric | Value |
|--------|-------|
| Mean Variance Reduction | 0.61% |
| 95% CI Lower Bound | -0.92% |
| 95% CI Upper Bound | 2.90% |
| Minimum (across tests) | -2.07% |
| Maximum (across tests) | 8.15% |

### Results by Test Function

| Test | Dimension | Points | Variance Reduction (α=1.0) |
|------|-----------|--------|----------------------------|
| ProductCosine | 2D | 127 | 0.00% |
| ProductCosine | 2D | 251 | 0.00% |
| ProductCosine | 3D | 127 | 0.00% |
| SmoothPeriodic | 2D | 127 | **-2.07%** ⚠️ |
| SmoothPeriodic | 2D | 251 | -0.08% |
| SmoothPeriodic | 3D | 127 | **-1.07%** ⚠️ |
| MultiFrequency | 2D | 127 | -0.08% |
| GenzContinuous | 2D | 127 | **8.15%** ✓ |

⚠️ = Performance degradation  
✓ = Best case improvement (still below claimed 10% minimum)

### Analysis by α Parameter

Testing different arctan scaling factors:

| α | Mean Variance Reduction | CI | In Claimed Range? |
|---|------------------------|----|-------------------|
| 0.0 (baseline) | 0.00% | - | N/A |
| 0.5 | -0.10% | [-1.00%, 1.45%] | ❌ |
| 1.0 | 0.61% | [-0.92%, 2.90%] | ❌ |
| 1.5 | 0.86% | [-0.71%, 3.21%] | ❌ |
| 2.0 | -0.01% | [-1.15%, 1.74%] | ❌ |

**None of the α values achieve the claimed 10-30% variance reduction.**

---

## Interpretation

### Why the Hypothesis Failed

1. **Curvature Selection Strategy:** Using κ_arctan to select Korobov generators does not improve lattice quality for periodic functions. The arctan term adds high-frequency oscillations that disrupt the smooth divisor-based curvature.

2. **Golden Ratio Equidistribution:** The claimed enhancement via "golden-ratio equidistribution" is not realized in practice. The arctan(φ · frac(n/φ)) term does not provide meaningful geometric insight for lattice construction.

3. **Periodic Integrand Specificity:** For truly periodic functions (ProductCosine, SmoothPeriodic), the arctan refinement provides **zero or negative** benefit. The hypothesis specifically claimed improvements for periodic integrands.

4. **Non-Periodic Functions:** The only positive result (8.15% for GenzContinuous) was for a **non-periodic** function, contradicting the hypothesis's focus on periodic integrands.

### Statistical Significance

The confidence interval **[-0.92%, 2.90%]** includes zero, indicating that the observed 0.61% mean improvement is **not statistically significant**. This means we cannot reject the null hypothesis that arctan refinement has no effect.

### Reproducibility

Results are reproducible with seed=42. Multiple runs with different seeds show similar patterns: arctan refinement either has no effect or degrades performance.

---

## Files and Artifacts

### Code Files

- `arctan_curvature.py` - Core implementation of arctan-refined curvature
- `qmc_integration_tests.py` - QMC integration test suite with standard test functions
- `run_experiment.py` - Main experiment runner with bootstrap confidence intervals

### Data Files

- `data/results_20251120_174206.json` - Complete experimental results (JSON)
- `EXPERIMENT_SUMMARY.txt` - Executive summary text file
- `logs/experiment_run.log` - Full experiment execution log

### Key Metrics

All experiments include:
- Integration error vs analytical solution
- Variance estimates
- L2 star discrepancy
- Bootstrap confidence intervals (1000 resamples)
- Statistical significance tests

---

## Conclusion

The hypothesis that arctan-refined curvature provides 10-30% variance reduction in Korobov lattice QMC integration for periodic integrands is **conclusively falsified** by this rigorous experimental study.

**Evidence:**
1. Mean improvement of 0.61% is **far below** the claimed 10% minimum
2. 95% CI includes zero (not statistically significant)
3. Performance degradation in 50% of test cases
4. No test achieved even the minimum claimed 10% reduction
5. Best case (8.15%) was for a non-periodic function, contradicting the hypothesis

**Recommendation:** The arctan refinement approach should **not** be used for Korobov lattice construction in QMC applications. The baseline divisor-weighted curvature κ(n) = d(n) · ln(n+1) / e² performs as well or better without the added complexity.

---

## Reproducibility

To reproduce this experiment:

```bash
cd experiments/arctan_korobov_qmc

# Full experiment (100 trials, 1000 bootstrap)
python run_experiment.py --trials 100 --bootstrap 1000

# Quick validation (20 trials, 100 bootstrap)
python run_experiment.py --quick

# Custom configuration
python run_experiment.py --trials 50 --bootstrap 500 --seed 123
```

**Requirements:**
- Python 3.7+
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SymPy >= 1.9
- See `../../requirements.txt`

---

## References

### Z-Framework
- Cognitive Number Theory: κ(n) curvature (cognitive_number_theory/divisor_density.py)
- Wave CRISPR Signal: θ'(n,k) bias resolution (wave_crispr_signal/z_framework.py)

### QMC Literature
- Korobov lattice rules for QMC integration
- Cranley-Patterson randomization for QMC variance estimation
- Genz test functions for QMC benchmarking

### Statistical Methods
- Bootstrap confidence intervals (Efron & Tibshirani)
- Hypothesis testing with multiple comparisons

---

## Contact

This experiment was conducted as part of the Z-Framework validation effort. For questions or to contribute additional tests, please open an issue in the zfifteen/dmc_rsa repository.

**Experiment Framework:** Z-Mode  
**Scientific Rigor:** Peer-reviewable methodology with full reproducibility  
**Data Availability:** All data, code, and logs included in this directory
