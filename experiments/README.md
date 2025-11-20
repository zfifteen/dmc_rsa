# Z-Framework Experiments

This directory contains rigorous experimental validations and falsification studies for Z-Framework hypotheses.

## Current Experiments

### 1. Arctan-Refined Curvature in Korobov Lattices for QMC

**Status:** ✅ Complete - **HYPOTHESIS FALSIFIED**  
**Directory:** `arctan_korobov_qmc/`  
**Date:** November 20, 2025

**Hypothesis Tested:**
> Augmenting κ(n) with arctan(φ · frac(n/φ)) terms achieves 10-30% variance cuts in QMC for periodic integrands.

**Result:**
- Mean variance reduction: **0.61%** (claimed: 10-30%)
- 95% CI: **[-0.92%, 2.90%]** (not statistically significant)
- **Verdict: Hypothesis conclusively falsified**

**Key Files:**
- `FINAL_REPORT.md` - Complete scientific report
- `README.md` - Methodology and detailed results
- `QUICKSTART.md` - Quick start guide
- `run_experiment.py` - Main experiment (100 trials, 1000 bootstrap)
- `data/results_*.json` - Complete experimental data
- `plots/` - 4 publication-quality visualizations

**Quick Start:**
```bash
cd arctan_korobov_qmc
python run_experiment.py --quick
python visualize_results.py
```

---

## Experiment Guidelines

All experiments in this directory follow rigorous scientific methodology:

### Required Elements

1. ✅ **Clear hypothesis statement** with specific, testable claims
2. ✅ **Pre-specified falsification criteria** before running experiments
3. ✅ **Standard benchmarks** or well-justified custom tests
4. ✅ **Statistical rigor** with confidence intervals and significance tests
5. ✅ **Complete code** with all dependencies specified
6. ✅ **Raw data** in machine-readable format (JSON, CSV)
7. ✅ **Visualizations** for key findings
8. ✅ **Reproducibility** with fixed random seeds and clear instructions
9. ✅ **Documentation** including executive summary and full report

### Directory Structure Template

```
experiment_name/
├── README.md                   # Comprehensive documentation
├── FINAL_REPORT.md            # Scientific report with interpretation
├── EXPERIMENT_SUMMARY.txt     # Executive summary (plain text)
├── QUICKSTART.md              # Quick start guide
├── .gitignore                 # Exclude __pycache__, etc.
├── core_implementation.py     # Main experimental code
├── test_suite.py              # Test functions/benchmarks
├── run_experiment.py          # Experiment runner with statistics
├── visualize_results.py       # Plotting script
├── data/
│   └── results_*.json         # Experimental results
├── plots/
│   └── *.png                  # Visualizations
└── logs/
    └── *.log                  # Execution logs
```

### Reporting Standards

#### Executive Summary Format

```
HYPOTHESIS: [Clear statement]

EXPERIMENTAL DESIGN:
  - Test functions: [List]
  - Sample sizes: [Values]
  - Statistical methods: [Bootstrap, etc.]
  - Trials: [Number]

KEY FINDINGS:
  - Metric: [Value] (95% CI: [lower, upper])
  - [Additional metrics]

VERDICT:
  [✓ Hypothesis supported / ❌ Hypothesis falsified]
  
  Reasons:
    - [Key reason 1]
    - [Key reason 2]

DETAILED RESULTS:
  [Table or list of results by configuration]
```

#### Code Quality

- ✅ Type hints for function signatures
- ✅ Docstrings for all public functions
- ✅ Clear variable names (no single-letter variables except i, j, k in loops)
- ✅ Modular design with reusable components
- ✅ Error handling with informative messages
- ✅ Command-line interface with `--help`

#### Statistical Standards

- ✅ Minimum 30 trials (preferably 100+)
- ✅ Bootstrap CIs with ≥1000 resamples
- ✅ Report both point estimates and uncertainty
- ✅ Test for statistical significance (p-values or CI overlap with null)
- ✅ Multiple comparison corrections if applicable

---

## Contributing New Experiments

### Before Starting

1. **Search existing literature** for similar studies
2. **Consult Z-Framework documentation** for validated methods
3. **Define clear hypothesis** with specific numerical claims
4. **Identify appropriate benchmarks** or justify custom tests
5. **Estimate computational requirements** (time, memory)

### During Execution

1. **Implement baseline first** before testing new methods
2. **Validate baseline** against known results if possible
3. **Run pilot studies** with small sample sizes to debug
4. **Document unexpected findings** even if negative
5. **Track all hyperparameter choices** and their justification

### After Completion

1. **Generate all required artifacts** (report, plots, data)
2. **Write executive summary** with clear verdict
3. **Test reproducibility** on a fresh environment
4. **Commit all files** following the directory structure template
5. **Update this index** with your experiment

---

## Experiment History

| Date | Experiment | Hypothesis | Verdict | Key Finding |
|------|-----------|------------|---------|-------------|
| 2025-11-20 | Arctan-Refined Korobov | 10-30% variance reduction | ❌ Falsified | Only 0.61% reduction (CI overlaps zero) |

---

## Future Experiment Ideas

### High Priority

1. **Z5D Prime Density at Ultra Scales**
   - Test k*≈0.04449 at N > 10^18
   - Validate claimed 210% density boost
   - Compare against Stadlmann's θ ≈ 0.525

2. **RQMC for RSA Factorization**
   - Test O(N^{-3/2+ε}) convergence claim
   - Compare 256-bit RSA factorization performance
   - Benchmark against standard QMC methods

3. **Golden-Ratio Prime Clustering**
   - Validate 15-20% density boost claim
   - Test GeodesicMapper across scales
   - Compare with sieve methods

### Medium Priority

4. **Spectral Analysis for Prime Geodesics**
   - Cross-domain integration test
   - Validate <0.01% error claim at k ≥ 10^5

5. **Biological Sequence Analysis**
   - Test geometric mappings on real genomic data
   - Compare with BioPython standard methods

6. **Quantum Alignment Hypothesis**
   - Design quantum-inspired validation
   - Benchmark against classical methods

### Research Questions

- Can curvature-based methods improve other lattice types (Fibonacci, cyclic)?
- Do Z-Framework methods scale to dimensions d > 10?
- Are there specific problem domains where arctan refinement helps?

---

## Resources

### Z-Framework Core

- `cognitive_number_theory/` - κ(n) curvature functions
- `wave_crispr_signal/` - θ'(n,k) bias resolution
- `scripts/rank1_lattice.py` - Korobov lattice implementations
- `scripts/qmc_engines.py` - QMC sampling engines

### External References

- **QMC Literature:** See `docs/QMC_RSA_SUMMARY.md`
- **Statistical Methods:** Bootstrap (Efron), Genz test functions
- **Z-Framework:** GitHub repos (unified-framework, z-sandbox, etc.)

---

## Contact

For questions about experiments or to propose new studies:
- Open an issue in the zfifteen/dmc_rsa repository
- Follow the experiment guidelines above
- Include clear hypothesis and proposed methodology

**Remember:** Negative results (falsifications) are valuable scientific contributions!
