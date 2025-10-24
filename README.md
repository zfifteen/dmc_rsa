# QMC RSA Factorization

This project implements and demonstrates the first documented application of Quasi-Monte Carlo (QMC) variance reduction techniques to RSA factorization candidate sampling. It compares QMC methods against Monte Carlo (MC) baselines and explores φ-biased transformations for improved candidate generation.

## Overview

RSA factorization involves searching for prime factors of large semiprimes (products of two primes). Traditional methods use random candidate sampling, but QMC can provide deterministic, low-discrepancy sequences that reduce variance and improve hit rates.

Key features:
- Rigorous statistical comparison with bootstrap confidence intervals
- Interactive web demos for visualization
- Python analysis scripts for benchmarking
- Support for Cranley-Patterson shifts for QMC variance estimation

## Key Findings

- QMC provides measurable improvements over MC: 1.03× to 1.34× better unique candidates
- Improvements scale with semiprime size
- φ-bias currently reduces performance (needs refinement)
- Statistical significance confirmed with 1000 trials and 95% CIs

**New: Enhanced QMC Capabilities (October 2025)**
- Sobol with Owen scrambling as recommended default
- Replicated QMC with Cranley-Patterson randomization
- Confidence intervals from independent replicates
- L2 discrepancy and stratification balance metrics
- Smooth candidate mapping to preserve low-discrepancy properties

For detailed results, see [docs/QMC_RSA_SUMMARY.md](docs/QMC_RSA_SUMMARY.md).

## Project Structure

```
.
├── docs/
│   └── QMC_RSA_SUMMARY.md          # Detailed implementation summary and findings
├── demos/
│   ├── qmc_rsa_demo_v2.html        # Interactive HTML demo (main)
│   ├── grok.html                   # Alternative demo
│   └── qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx  # React component demo
├── scripts/
│   └── qmc_factorization_analysis.py  # Python analysis script
├── reports/
│   └── qmc_statistical_results_899.csv  # Benchmark results for N=899
└── README.md
```

## Setup and Requirements

### Python Analysis
- Python 3.7+
- NumPy >= 1.20.0
- Pandas >= 1.3.0
- SciPy >= 1.7.0 (for QMC support with Owen scrambling)

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scipy
```

**Note:** The enhanced QMC features require scipy.stats.qmc with Sobol/Owen scrambling, available in SciPy 1.7+.

### Web Demos
- Modern web browser with JavaScript enabled
- No additional setup required for HTML demos
- React demo requires a React environment (not standalone)

## Usage

### Running the Interactive Demo
1. Open `demos/qmc_rsa_demo_v2.html` in your web browser
2. Choose sampling method (QMC, MC, or φ-biased variants)
3. Adjust parameters (sample size, semiprime N)
4. Run trials and view real-time statistics and visualizations

### Running the Analysis Script
```bash
python scripts/qmc_factorization_analysis.py
```

This generates comprehensive benchmarks and statistical analysis for various semiprime sizes.

### Running the Enhanced QMC Demo (New)
```bash
python examples/qmc_directions_demo.py
```

This demonstrates:
- Replicated QMC with confidence intervals (Cranley-Patterson randomization)
- Sobol vs Halton engine comparison
- Effect of Owen scrambling
- Statistical significance testing
- Usage recommendations

### Quick Example: Replicated QMC Analysis
```python
from qmc_factorization_analysis import QMCFactorization

results = QMCFactorization.run_replicated_qmc_analysis(
    n=899,                # Semiprime to factor
    num_samples=256,      # Power of 2 for Sobol
    num_replicates=16,    # For confidence intervals
    engine_type="sobol",  # Recommended default
    scramble=True,        # Owen scrambling
    seed=42               # Reproducibility
)

print(f"Mean unique candidates: {results['unique_count']['mean']:.2f}")
print(f"95% CI: [{results['unique_count']['ci_lower']:.2f}, "
      f"{results['unique_count']['ci_upper']:.2f}]")
print(f"L2 discrepancy: {results['l2_discrepancy']['mean']:.4f}")
```

### React Demo
The React component in `demos/qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx` can be integrated into a React application. It provides an interactive interface with charts and controls.

## Files Description

### Core Files
- **qmc_rsa_demo_v2.html**: Standalone interactive web demo with fair comparisons
- **qmc_factorization_analysis.py**: Python script for rigorous statistical analysis
- **qmc_engines.py**: Enhanced QMC engine module with Sobol/Halton support
- **qmc_statistical_results_899.csv**: Raw data from 1000 trials on N=899
- **QMC_RSA_SUMMARY.md**: Comprehensive summary of implementation, fixes, and findings

### Examples
- **qmc_directions_demo.py**: Comprehensive demonstration of enhanced QMC capabilities

### Tests
- **test_large.py**: Original test for baseline methods
- **test_qmc_engines.py**: Tests for enhanced QMC engine module
- **test_replicated_qmc.py**: Tests for replicated QMC analysis with confidence intervals

## Results

For N=899 (p=29, q=31):
- QMC unique candidates: 1.03× improvement over MC
- Hit probability improvements
- Star discrepancy metrics

See the summary document for full results across multiple semiprime sizes.

## Contributing

This is a research implementation. For extensions:
- ✅ **DONE:** Sobol sequences with Owen scrambling (now default)
- Refine φ-bias parameters for balanced semiprimes
- Test on larger cryptographic-scale numbers
- **Suggested next steps:**
  - ECM σ parameter sampling via QMC
  - GNFS polynomial selection sweeps
  - Multi-armed bandits for method selection
  - Extend to higher dimensions (residue classes, window widths, etc.)

## License

Research code - see individual files for licensing.

## Citation

If using this work, please cite as the first documented QMC application to RSA factorization candidate sampling (October 2025).