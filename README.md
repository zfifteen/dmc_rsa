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
- **⚠️ θ′-bias falsified**: Rigorous experiment shows θ′-biased QMC **reduces** unique candidates by 0.2-4.8% (see `experiments/theta_prime_qmc_falsification/`)
- Statistical significance confirmed with 1000 trials and 95% CIs

**New: Enhanced QMC Capabilities (October 2025)**
- Sobol with Owen scrambling as recommended default
- Replicated QMC with Cranley-Patterson randomization
- Confidence intervals from independent replicates
- L2 discrepancy and stratification balance metrics
- Smooth candidate mapping to preserve low-discrepancy properties
- **⭐ NEW: Bias-Adaptive Sampling Engine (November 2025)**
  - Z-framework integration with κ(n) curvature and θ′(n,k) bias resolution
  - Three bias modes: `theta_prime` (golden-angle spiral), `prime_density` (curvature-based), `golden_spiral` (Fibonacci)
  - Z-invariant metrics: discrepancy, unique rate, mean kappa, savings estimation
  - Performance: d=10, N=10,000 in 0.003s (requirement: <1s) ✓
  - Demo: 45x improvement over MC in integration tests
  - Bootstrap confidence interval analysis with 1000+ iterations
  - Full CLI tools: `run_demo.py` and `discrepancy_test.py`
- **✨ NEW: Z5D Extension with k*≈0.04449 (November 2025)**
  - Extended θ′(n,k) to Z5D for 210% prime density boost at N=10^6
  - High-precision validation with mpmath dps=50 (<10^{-16} error)
  - Curvature reduction: 55.73% mean, 95% CI [53.84%, 57.50%] (target: 56.5% [52.1%, 60.9%]) ✓
  - Latency: 0.0007 ms (target: <0.019 ms) ✓
  - **🔬 NEW: Validated at 10^18 scale with 100K samples & 1000 bootstrap iterations**
    - Numerical stability confirmed across 18 orders of magnitude
    - Statistical confidence: 95% CI width ±0.0009
    - Performance: 0.033 ms/sample, ~30K samples/second
    - See [docs/Z5D_TESTING_AT_1E18_SCALE.md](docs/Z5D_TESTING_AT_1E18_SCALE.md) for comprehensive analysis
    - Quick reference: [docs/Z5D_1E18_QUICK_REFERENCE.md](docs/Z5D_1E18_QUICK_REFERENCE.md)
  - Complete demo suite: `examples/demo_complete.py`
  - Curvature analysis tool: `bin/curvature_test.py`
  - See [docs/Z5D_EXTENSION_SUMMARY.md](docs/Z5D_EXTENSION_SUMMARY.md) for details
- **Rank-1 lattice constructions with group-theoretic foundations**
  - Cyclic subgroup-based generating vectors
  - Fibonacci and Korobov construction methods
  - **NEW: Elliptic cyclic geometry embedding**
  - Lattice quality metrics (minimum distance, covering radius)
  - φ(N)-aware mappings for RSA semiprime structure
  - **✨ Auto-scaling subgroup order** via geometric parameters (cone_height, spiral_depth)
    - Zero-config optimal stratification
    - 23-37% lower discrepancy vs fixed parameters at n>1k
    - Eliminates manual tuning
- **NEW: Elliptic Adaptive Search (EAS)**
  - Elliptic lattice sampling with golden-angle spiral
  - Adaptive window sizing based on bit length
  - Efficient for small to medium factors (16-40 bits)
  - 70% success rate on 32-bit, 40% on 40-bit semipromes
  - Orders of magnitude search space reduction
- **Biased QMC for Fermat Factorization**
  - 43% reduction in average trials with u^4 bias transformation
  - Adaptive bias strategies for close vs. distant factors
  - Hybrid approach (sequential prefix + biased QMC)
  - Dual-mixture sampling for comprehensive coverage
  - Automatic sampler recommendation system

For detailed results, see [docs/QMC_RSA_SUMMARY.md](docs/QMC_RSA_SUMMARY.md).

## Project Structure

```
.
├── cognitive_number_theory/       # Z-framework: divisor density (NEW)
│   ├── __init__.py
│   └── divisor_density.py         # kappa(n) curvature function
├── wave_crispr_signal/            # Z-framework: bias resolution (NEW)
│   ├── __init__.py
│   └── z_framework.py             # theta_prime, Z_transform, PHI
├── bin/
│   └── discrepancy_test.py        # Bootstrap CI analysis utility (NEW)
├── docs/
│   ├── QMC_RSA_SUMMARY.md           # Detailed implementation summary and findings
│   ├── RANK1_LATTICE_INTEGRATION.md # Rank-1 lattice documentation
│   └── RANK1_IMPLEMENTATION_SUMMARY.md  # Rank-1 implementation details
├── demos/
│   ├── qmc_rsa_demo_v2.html         # Interactive HTML demo (main)
│   ├── grok.html                    # Alternative demo
│   └── qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx  # React component demo
├── scripts/
│   ├── examples/
│   │   ├── bias_adaptive_example.py  # Bias-adaptive engine examples (NEW)
│   │   ├── qmc_directions_demo.py    # QMC engine demonstration
│   │   ├── rank1_lattice_example.py  # Rank-1 lattice examples
│   │   └── fermat_qmc_demo.py        # Fermat factorization with biased QMC demo
│   ├── qmc_engines.py                # Enhanced QMC engines with bias-adaptive support
│   ├── qmc_factorization_analysis.py  # Python analysis script
│   ├── rank1_lattice.py              # Rank-1 lattice construction module
│   ├── fermat_qmc_bias.py            # Fermat factorization with biased QMC
│   ├── run_demo.py                   # Bias-adaptive sampling CLI (NEW)
│   ├── benchmark_elliptic.py         # Elliptic geometry benchmark
│   ├── demo_elliptic_geometry.py     # Elliptic geometry demonstration
│   ├── test_z_framework.py           # Z-framework tests (NEW)
│   ├── test_bias_adaptive.py         # Bias-adaptive engine tests (NEW)
│   ├── test_qmc_engines.py           # QMC engine tests
│   ├── test_fermat_qmc_bias.py       # Tests for Fermat QMC module
│   └── test_*.py                     # Various test suites
├── reports/
│   └── qmc_statistical_results_899.csv  # Benchmark results for N=899
├── experiments/                      # Experimental validations (NEW)
│   └── theta_prime_qmc_falsification/  # θ′-biased QMC falsification experiment
│       ├── EXECUTIVE_SUMMARY.md        # Experiment results summary
│       ├── METHODOLOGY.md              # Detailed methodology
│       ├── README.md                   # Quick start guide
│       ├── qmc_factorization_experiment.py  # Main experiment script
│       ├── verify_results.py           # Result verification
│       ├── results/*.csv               # Per-replicate data (8 configs)
│       └── deltas/*.json               # Summary metrics with CI (8 configs)
├── ELLIPTIC_INTEGRATION_SUMMARY.md  # Elliptic geometry integration summary
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

### Running Rank-1 Lattice Tests (New)
```bash
# Unit tests for rank-1 lattice constructions
python scripts/test_rank1_lattice.py

# Elliptic geometry demonstration
python scripts/demo_elliptic_geometry.py

# Benchmark comparison
python scripts/benchmark_elliptic.py
```

### Running Fermat QMC Bias Demo (New)
```bash
# Interactive demo showing biased QMC for Fermat factorization
python examples/fermat_qmc_demo.py

# Run tests for the module
python scripts/test_fermat_qmc_bias.py

# Command-line factorization
python scripts/fermat_qmc_bias.py 899 --sampler biased_golden --beta 2.0
```

### Running Bias-Adaptive Sampling Engine (New - November 2025)
```bash
# Generate bias-adaptive QMC samples with CLI
python scripts/run_demo.py --method sobol_owen --bias theta_prime --bits 32 --output results/samples.csv --verbose

# Compute bootstrap confidence intervals for discrepancy
python bin/discrepancy_test.py --input results/samples.csv --n_boot 1000 --output results/metrics.csv --verbose

# Run comprehensive examples
python scripts/examples/bias_adaptive_example.py

# Run test suites
python scripts/test_z_framework.py      # Z-framework tests
python scripts/test_bias_adaptive.py     # Bias-adaptive engine tests
```

### Running Z5D Extension Validation (New - November 2025)
```bash
# Run full test suite with all benchmarks
python examples/demo_complete.py

# Quick validation mode (reduced sample sizes)
python examples/demo_complete.py --quick

# Skip Z5D validation for faster testing
python examples/demo_complete.py --no-z5d

# Verbose output with detailed statistics
python examples/demo_complete.py --verbose

# Curvature reduction analysis
python bin/curvature_test.py --slots 1000 --prime nearest --output results/curvature.csv --verbose

# Z5D unit tests
python scripts/test_z5d_extension.py
```

### Running Z5D Extension Testing at 10^18 Scale (New - November 2025)
```bash
# Full validation at 10^18 scale (100K samples, 1000 bootstrap iterations)
python scripts/test_z5d_1e18.py --output results/z5d_1e18_results.json

# Quick test with reduced sample size
python scripts/test_z5d_1e18.py --samples 10000 --bootstrap 100

# Custom configuration
python scripts/test_z5d_1e18.py \
  --samples 100000 \
  --bootstrap 1000 \
  --precision-tests 100 \
  --dps 50 \
  --seed 42 \
  --output results.json

# View documentation
cat docs/Z5D_TESTING_AT_1E18_SCALE.md  # Comprehensive analysis
cat docs/Z5D_1E18_QUICK_REFERENCE.md   # Quick reference guide
```

### Quick Example: Z5D at 10^18 Scale
```python
from wave_crispr_signal import theta_prime, K_Z5D
from scripts.test_z5d_1e18 import generate_stratified_samples_1e18

# Generate stratified samples across [1, 10^18]
samples = generate_stratified_samples_1e18(n_samples=100000, seed=42)
print(f"Sample range: [{samples[0]:,} to {samples[-1]:,}]")

# Compute theta values with Z5D k
theta_values = theta_prime(samples, k=K_Z5D)
print(f"Mean theta: {theta_values.mean():.6f}")

# High-precision for critical applications
from wave_crispr_signal import theta_prime_high_precision
theta_hp = theta_prime_high_precision(10**18, k=K_Z5D, dps=50)
print(f"High-precision θ'(10^18): {theta_hp}")
```

### Quick Example: Z5D Extension with k*≈0.04449 (Original)
```python
python bin/curvature_test.py --slots 1000 --prime nearest --output results/curvature.csv --verbose

# Z5D unit tests
python scripts/test_z5d_extension.py
```

### Quick Example: Bias-Adaptive Sampling Engine
```python
from scripts.qmc_engines import QMCConfig, make_engine, apply_bias_adaptive, compute_z_invariant_metrics

# Create bias-adaptive Sobol engine
cfg = QMCConfig(
    dim=2,                      # 2D sampling
    n=1024,                     # 1024 samples (power of 2)
    engine='sobol',             # Sobol engine
    scramble=True,              # Owen scrambling
    bias_mode='theta_prime',    # Golden-angle spiral bias
    z_k=0.3,                    # Theta exponent
    seed=42                     # Reproducibility
)

# Generate and bias samples
eng = make_engine(cfg)
samples = eng.random(1024)
biased_samples = apply_bias_adaptive(samples, bias_mode='theta_prime', k=0.3)

# Compute Z-invariant metrics
metrics = compute_z_invariant_metrics(biased_samples, method='sobol_theta_prime')
print(f"Discrepancy:     {metrics['discrepancy']:.6f}")
print(f"Unique rate:     {metrics['unique_rate']:.6f}")
print(f"Mean kappa:      {metrics['mean_kappa']:.4f}")
print(f"Savings est:     {metrics['savings_estimate']:.2f}x")

# Available bias modes: 'theta_prime', 'prime_density', 'golden_spiral'
```

### Quick Example: Z5D Extension with k*≈0.04449
```python
from wave_crispr_signal import (
    theta_prime, K_Z5D, theta_prime_high_precision,
    compute_prime_density_boost, validate_z5d_extension
)

# Compute theta with Z5D k value
theta = theta_prime(1000000, k=K_Z5D)  # k=0.04449
print(f"θ'(10^6, k={K_Z5D}) = {theta:.6f}")

# High-precision computation (mpmath dps=50)
theta_hp = theta_prime_high_precision(1000000, k=K_Z5D, dps=50)
print(f"High-precision: {theta_hp}")

# Prime density boost analysis
boost = compute_prime_density_boost(n_samples=1000000, k=K_Z5D, baseline_k=0.3)
print(f"Prime density boost: {boost['boost_percent']:.1f}%")

# Comprehensive validation with bootstrap CI
results = validate_z5d_extension(
    n_samples=1000000,
    k=K_Z5D,
    n_bootstrap=1000,
    confidence=0.95,
    dps=50
)
print(f"Max error: {results['max_error']:.2e}")
print(f"All errors < 10^-16: {results['all_errors_valid']}")
```

### Quick Example: Curvature Reduction Analysis
```python
from cognitive_number_theory import kappa
import sympy

# Analyze curvature reduction for a slot
slot = 1000
prime_slot = int(sympy.nextprime(slot - 1))

baseline_k = float(kappa(slot))
biased_k = float(kappa(prime_slot))

reduction = (1.0 - biased_k / baseline_k) * 100.0
print(f"Slot {slot} -> Prime {prime_slot}")
print(f"Curvature reduction: {reduction:.2f}%")

# Or use the command-line tool
# python bin/curvature_test.py --slots 1000 --prime nearest --output curvature.csv
```

### Quick Example: Elliptic Cyclic Lattice
```python
from qmc_engines import QMCConfig, make_engine

# Create elliptic cyclic lattice
cfg = QMCConfig(
    dim=2,
    n=128,
    engine="elliptic_cyclic",
    subgroup_order=128,
    elliptic_b=0.8,      # Eccentricity ~0.6
    scramble=True,
    seed=42
)

engine = make_engine(cfg)
points = engine.random(128)

# All points lie on ellipse: (x/a)² + (y/b)² ≤ 1
# Optimized for elliptic arc-length uniformity
```

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

### Quick Example: Fermat Factorization with Biased QMC
```python
from fermat_qmc_bias import FermatConfig, SamplerType, fermat_factor, recommend_sampler

# Automatic recommendation
N = 899
rec = recommend_sampler(N=N, p=29, q=31, window_size=100000)
print(f"Recommended: {rec['sampler_type'].value}")

# Configure and factor
cfg = FermatConfig(
    N=N,
    max_trials=100000,
    sampler_type=SamplerType.BIASED_GOLDEN,
    beta=2.0,  # Bias exponent (higher = more bias toward small k)
    seed=42
)

result = fermat_factor(cfg)
print(f"Success: {result['success']}")
print(f"Factors: {result['factors']}")
print(f"Trials: {result['trials']}")
```

### React Demo
The React component in `demos/qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx` can be integrated into a React application. It provides an interactive interface with charts and controls.

### Quick Example: Elliptic Adaptive Search (EAS)
```python
# Add scripts directory to Python path
import sys
sys.path.append('scripts')

from eas_factorize import factorize_eas, EASConfig

# Basic usage with default settings
result = factorize_eas(899)  # Factor 29 × 31
if result.success:
    print(f"Factors: {result.factor_p} × {result.factor_q}")
    print(f"Search reduction: {result.search_reduction:.0f}×")

# Custom configuration for larger factors
config = EASConfig(
    max_samples=5000,
    adaptive_window=True,
    base_radius_factor=0.15
)
result = factorize_eas(your_semiprime, config=config, verbose=True)

# Using EAS with QMC framework
from qmc_engines import QMCConfig, make_engine

cfg = QMCConfig(
    dim=2, 
    n=128, 
    engine="eas",
    eas_reference_point=1000.0,  # Central value for elliptic lattice (default: 1000.0)
    eas_max_samples=2000,         # Maximum candidates (default: 2000)
    eas_adaptive_window=True,     # Enable adaptive sizing (default: True)
    seed=42
)
engine = make_engine(cfg)
points = engine.random(128)
```


## Files Description

### Core Files
- **qmc_rsa_demo_v2.html**: Standalone interactive web demo with fair comparisons
- **qmc_factorization_analysis.py**: Python script for rigorous statistical analysis
- **qmc_engines.py**: Enhanced QMC engine module with Sobol/Halton/Rank-1 lattice/EAS support
- **rank1_lattice.py**: Group-theoretic rank-1 lattice construction module
- **eas_factorize.py**: Elliptic Adaptive Search implementation
- **fermat_qmc_bias.py**: Fermat factorization with biased QMC sampling (NEW)
- **qmc_statistical_results_899.csv**: Raw data from 1000 trials on N=899
- **QMC_RSA_SUMMARY.md**: Comprehensive summary of implementation, fixes, and findings
- **RANK1_LATTICE_INTEGRATION.md**: Documentation for rank-1 lattice integration

### Examples
- **qmc_directions_demo.py**: Comprehensive demonstration of enhanced QMC capabilities
- **eas_example.py**: Elliptic Adaptive Search usage examples and demonstrations
- **fermat_qmc_demo.py**: Demonstration of biased QMC for Fermat factorization (NEW)

### Tests
- **test_large.py**: Original test for baseline methods
- **test_qmc_engines.py**: Tests for enhanced QMC engine module
- **test_replicated_qmc.py**: Tests for replicated QMC analysis with confidence intervals
- **test_rank1_lattice.py**: Unit tests for rank-1 lattice construction
- **test_rank1_integration.py**: Integration tests for rank-1 lattice with QMC framework
- **test_eas.py**: Unit tests for Elliptic Adaptive Search
- **test_fermat_qmc_bias.py**: Tests for Fermat factorization with biased QMC (NEW)
- **quick_validation.py**: Fast end-to-end validation test

## Results

### QMC Candidate Sampling (N=899)
For N=899 (p=29, q=31):
- QMC unique candidates: 1.03× improvement over MC
- Hit probability improvements
- Star discrepancy metrics

### Biased QMC for Fermat Factorization (NEW)
Based on validation experiments with 60-bit semiprimes:
- **43% reduction in average trials** with biased QMC (u^4) vs uniform sampling
- Biased LDS with β=2.0: 3.2% improvement (60-bit, 100k window)
- Hybrid approach (5% sequential + biased): massive improvements for close factors (Δ ≤ 2²⁰)
- Far-biased and dual-mixture: optimal for distant factors (Δ > 2²¹)
- Success rate preservation: bias reduces trials without sacrificing completeness

See the summary document for full results across multiple semiprime sizes.

## Contributing

This is a research implementation. For extensions:
- ✅ **DONE:** Sobol sequences with Owen scrambling (now default)
- ✅ **DONE:** Rank-1 lattice constructions with group-theoretic foundations
- Refine φ-bias parameters for balanced semiprimes
- Test on larger cryptographic-scale numbers
- **Suggested next steps:**
  - ECM σ parameter sampling via QMC and rank-1 lattices
  - GNFS polynomial selection sweeps
  - Multi-armed bandits for method selection
  - Extend to higher dimensions (residue classes, window widths, etc.)
  - Hybrid lattice-Sobol constructions

## License

Research code - see individual files for licensing.

## Citation

If using this work, please cite as the first documented QMC application to RSA factorization candidate sampling (October 2025).