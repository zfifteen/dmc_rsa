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
- NumPy
- Pandas
- SciPy

Install dependencies:
```bash
pip install numpy pandas scipy
```

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

### React Demo
The React component in `demos/qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx` can be integrated into a React application. It provides an interactive interface with charts and controls.

## Files Description

- **qmc_rsa_demo_v2.html**: Standalone interactive web demo with fair comparisons
- **qmc_factorization_analysis.py**: Python script for rigorous statistical analysis
- **qmc_statistical_results_899.csv**: Raw data from 1000 trials on N=899
- **QMC_RSA_SUMMARY.md**: Comprehensive summary of implementation, fixes, and findings

## Results

For N=899 (p=29, q=31):
- QMC unique candidates: 1.03× improvement over MC
- Hit probability improvements
- Star discrepancy metrics

See the summary document for full results across multiple semiprime sizes.

## Contributing

This is a research implementation. For extensions:
- Add Sobol sequences for better high-dimensional performance
- Refine φ-bias parameters for balanced semiprimes
- Test on larger cryptographic-scale numbers

## License

Research code - see individual files for licensing.

## Citation

If using this work, please cite as the first documented QMC application to RSA factorization candidate sampling (October 2025).