# Final Experiment Report: Arctan-Refined Curvature Hypothesis Falsification

## Z-Framework Validation Study
**Date:** November 20, 2025  
**Repository:** zfifteen/dmc_rsa  
**Experiment Location:** `experiments/arctan_korobov_qmc/`

---

## 🎯 Objective

To rigorously test the hypothesis that arctan-refined curvature enhances Korobov lattice QMC performance, specifically:

> "Augmenting κ(n) with arctan(φ · frac(n/φ)) terms achieves **10-30% variance cuts** in QMC for periodic integrands via golden-ratio equidistribution."

---

## 📊 Executive Summary

### **HYPOTHESIS FALSIFIED** ❌

The claimed 10-30% variance reduction was **not observed** in rigorous experimental testing.

**Key Results:**
- **Mean variance reduction: 0.61%** (far below claimed 10-30%)
- **95% Confidence Interval: [-0.92%, 2.90%]**
- **CI overlaps zero:** Not statistically significant
- **50% of tests:** Performance degraded (negative variance reduction)
- **Best case: 8.15%** (still below claimed minimum of 10%)

---

## 🔬 Experimental Methodology

### Test Design

1. **Test Functions (Standard QMC Benchmarks):**
   - ProductCosine: ∏ᵢ cos(2πxᵢ) - highly periodic
   - SmoothPeriodic: ∏ᵢ (1 + 0.5 sin(2πxᵢ)) - smooth periodic  
   - MultiFrequencyCosine: cos(2π Σᵢ wᵢxᵢ) - mixed frequencies
   - GenzContinuous: exp(-Σᵢ cᵢ|xᵢ - wᵢ|) - standard benchmark

2. **Dimensions:** 2D, 3D

3. **Lattice Sizes:** 127, 251 points (prime numbers for optimal Korobov properties)

4. **Statistical Rigor:**
   - 100 independent trials per test
   - 1000 bootstrap resamples for confidence intervals
   - Cranley-Patterson randomization for variance estimation
   - 95% confidence intervals throughout

5. **Implementation:**
   ```python
   κ_arctan(n) = κ(n) + α · arctan(φ · frac(n/φ))
   ```
   where:
   - κ(n) = d(n) · ln(n+1) / e² (baseline curvature)
   - φ = (1 + √5) / 2 (golden ratio)
   - α = scaling factor (tested: 0.0, 0.5, 1.0, 1.5, 2.0)

### Falsification Criteria

The hypothesis is falsified if:
1. ✅ Mean variance reduction < 10% or > 30%
2. ✅ 95% CI includes zero (not statistically significant)
3. ✅ Majority of tests show no benefit or degradation
4. ✅ Results not reproducible across different test functions

**All four criteria were met.**

---

## 📈 Detailed Results

### Variance Reduction by Test Function (α=1.0)

| Test | Dimension | Points | Variance Reduction | Status |
|------|-----------|--------|--------------------|--------|
| ProductCosine | 2D | 127 | 0.00% | No effect |
| ProductCosine | 2D | 251 | 0.00% | No effect |
| ProductCosine | 3D | 127 | 0.00% | No effect |
| SmoothPeriodic | 2D | 127 | **-2.07%** | ❌ Degraded |
| SmoothPeriodic | 2D | 251 | -0.08% | Negligible |
| SmoothPeriodic | 3D | 127 | **-1.07%** | ❌ Degraded |
| MultiFrequency | 2D | 127 | -0.08% | Negligible |
| GenzContinuous | 2D | 127 | **8.15%** | Best case* |

*Best case still below claimed 10% minimum

### Analysis Across α Values

| α | Mean Variance Reduction | 95% CI | Verdict |
|---|------------------------|--------|---------|
| 0.0 (baseline) | 0.00% | - | N/A |
| 0.5 | -0.10% | [-1.00%, 1.45%] | ❌ Not significant |
| 1.0 | 0.61% | [-0.92%, 2.90%] | ❌ Not significant |
| 1.5 | 0.86% | [-0.71%, 3.21%] | ❌ Not significant |
| 2.0 | -0.01% | [-1.15%, 1.74%] | ❌ Not significant |

**None of the α values achieve the claimed 10-30% variance reduction.**

### Statistical Significance

The 95% confidence interval **[-0.92%, 2.90%]** includes zero, indicating:
- The observed 0.61% mean improvement is **not statistically significant**
- We cannot reject the null hypothesis (arctan refinement has no effect)
- If there is any effect, it is too small to detect reliably

---

## 🔍 Why the Hypothesis Failed

### 1. Curvature Selection Strategy Flawed

Using κ_arctan to select Korobov generators does **not** improve lattice quality. The arctan term adds high-frequency oscillations that disrupt the smooth divisor-based curvature without providing geometric benefit.

### 2. Golden Ratio Equidistribution Not Realized

The claimed enhancement via "golden-ratio equidistribution" is **not realized in practice**. The term arctan(φ · frac(n/φ)) does not meaningfully interact with the Korobov lattice structure.

### 3. Periodic Function Specificity Failed

For truly periodic functions (ProductCosine, SmoothPeriodic), the arctan refinement provides:
- **Zero benefit** (ProductCosine: 0.00% across all tests)
- **Negative benefit** (SmoothPeriodic: -2.07% degradation)

The hypothesis **specifically claimed** improvements for periodic integrands, which was not observed.

### 4. Non-Periodic Exception

The only positive result (8.15% for GenzContinuous) was for a **non-periodic** function, which:
- Still falls below the claimed 10% minimum
- Contradicts the hypothesis's focus on periodic integrands
- Is not statistically robust (CI includes lower values)

---

## 📁 Experiment Artifacts

All code, data, and visualizations are available in `experiments/arctan_korobov_qmc/`:

### Core Implementation
- `arctan_curvature.py` - Arctan refinement and Korobov generator
- `qmc_integration_tests.py` - Test suite with standard test functions
- `run_experiment.py` - Main experiment with bootstrap analysis

### Results
- `data/results_20251120_174206.json` - Complete results (100 trials, 1000 bootstrap)
- `EXPERIMENT_SUMMARY.txt` - Executive summary
- `logs/experiment_run.log` - Full execution log

### Documentation
- `README.md` - Comprehensive experiment documentation
- `visualize_results.py` - Plotting script

### Visualizations
- `plots/variance_reduction_by_test.png` - Bar chart by test function
- `plots/variance_reduction_by_alpha.png` - Performance vs α parameter
- `plots/confidence_interval.png` - Bootstrap CI for α=1.0
- `plots/histogram_variance_reductions.png` - Distribution of results

---

## 🎓 Scientific Rigor

### Strengths of This Study

1. ✅ **Standard benchmarks:** Used well-established QMC test functions
2. ✅ **Adequate sample size:** 100 trials per test, 8 test configurations
3. ✅ **Bootstrap CIs:** 1000 resamples for robust uncertainty quantification
4. ✅ **Multiple α values:** Tested range of scaling factors
5. ✅ **Reproducible:** All code and data provided, fixed random seeds
6. ✅ **Clear criteria:** Pre-specified falsification criteria
7. ✅ **Negative results reported:** No cherry-picking, all results shown

### Limitations

1. **Limited to small dimensions:** Only tested 2D and 3D (computational constraints)
2. **Prime lattice sizes:** Only tested n=127, 251 (could extend to larger primes)
3. **Single generator selection strategy:** Used minimum curvature; alternatives exist

However, these limitations do **not** explain the complete failure to observe the claimed 10-30% reduction.

---

## 💡 Recommendations

### For Practitioners

1. **Do NOT use** arctan refinement for Korobov lattice construction
2. **Use baseline curvature** κ(n) = d(n) · ln(n+1) / e² without arctan term
3. **Standard Korobov rules** (e.g., a ≈ n/3 coprime to n) perform as well or better

### For Future Research

1. **Investigate alternative refinements:** Other golden-ratio embeddings may work
2. **Test on cryptographic applications:** RSA factorization context from hypothesis
3. **Explore higher dimensions:** Perhaps benefit appears in d > 3
4. **Different generator selection:** Maximum curvature or other criteria

### For the Z-Framework

This falsification demonstrates the **importance of empirical validation** for Z-Framework hypotheses. Proposed extensions should be rigorously tested before claiming specific performance improvements.

---

## 📝 Conclusion

The hypothesis that arctan-refined curvature provides 10-30% variance reduction in Korobov lattice QMC for periodic integrands is **conclusively falsified** by this rigorous experimental study.

**Evidence Summary:**
1. ❌ Mean improvement (0.61%) far below claimed minimum (10%)
2. ❌ 95% CI includes zero (not statistically significant)
3. ❌ 50% of tests showed performance degradation
4. ❌ Best case (8.15%) still below minimum claim
5. ❌ Zero benefit for periodic functions (ProductCosine: 0.00%)
6. ❌ Degradation for smooth periodic functions (up to -2.07%)

**Scientific Verdict:**

The arctan refinement approach should **not** be adopted for practical QMC applications. The baseline divisor-weighted curvature κ(n) = d(n) · ln(n+1) / e² is simpler and performs equally well or better.

This study exemplifies **rigorous hypothesis testing** in the Z-Framework context, demonstrating that not all proposed mathematical refinements translate to empirical improvements.

---

## 🔄 Reproducibility

To reproduce this experiment:

```bash
cd experiments/arctan_korobov_qmc

# Full experiment (100 trials, 1000 bootstrap)
python run_experiment.py --trials 100 --bootstrap 1000

# Quick validation (20 trials, 100 bootstrap)
python run_experiment.py --quick

# Generate visualizations
python visualize_results.py
```

**Requirements:** Python 3.7+, NumPy, SciPy, SymPy, Matplotlib (see `requirements.txt`)

---

**Experiment Conducted By:** Z-Mode Framework  
**Scientific Standards:** Peer-reviewable methodology with full data transparency  
**Repository:** github.com/zfifteen/dmc_rsa/experiments/arctan_korobov_qmc
