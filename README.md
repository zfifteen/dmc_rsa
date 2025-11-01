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
  - 70% success rate on 32-bit, 40% on 40-bit semiprimes
  - Orders of magnitude search space reduction
- **Biased QMC for Fermat Factorization**
  - 43% reduction in average trials with u^4 bias transformation
  - Adaptive bias strategies for close vs. distant factors
  - Hybrid approach (sequential prefix + biased QMC)
  - Dual-mixture sampling for comprehensive coverage
  - Automatic sampler recommendation system
- **✨ NEW: κ-Weighted Rank-1 Lattices (November 2025)**
  - Weight lattice points by divisor density curvature κ(n) = d(n)·ln(n)/e²
  - Bias toward low-curvature candidates for improved factorization
  - Expected 5-12% lift in hit rate on distant-factor RSA (p/q > 1.2)
  - Integrated with all lattice types (Fibonacci, Korobov, Cyclic)
  - Vectorized κ computation for efficiency
  - CLI flag `--with-kappa-weight` for analysis scripts

For detailed results, see [docs/QMC_RSA_SUMMARY.md](docs/QMC_RSA_SUMMARY.md).

## Project Structure

```
.
├── docs/
│   ├── QMC_RSA_SUMMARY.md           # Detailed implementation summary and findings
│   ├── RANK1_LATTICE_INTEGRATION.md # Rank-1 lattice documentation
│   └── RANK1_IMPLEMENTATION_SUMMARY.md  # Rank-1 implementation details
├── demos/
│   ├── qmc_rsa_demo_v2.html         # Interactive HTML demo (main)
│   ├── grok.html                    # Alternative demo
│   └── qmc_φ_biased_rsa_candidate_sampler_web_demo_react.jsx  # React component demo
├── examples/
│   ├── qmc_directions_demo.py       # QMC engine demonstration
│   ├── rank1_lattice_example.py     # Rank-1 lattice examples
│   └── fermat_qmc_demo.py           # Fermat factorization with biased QMC demo
├── scripts/
│   ├── qmc_factorization_analysis.py  # Python analysis script
│   ├── rank1_lattice.py              # Rank-1 lattice construction module
│   ├── qmc_engines.py                # Enhanced QMC engines
│   ├── fermat_qmc_bias.py            # Fermat factorization with biased QMC
│   ├── benchmark_elliptic.py         # Elliptic geometry benchmark
│   ├── demo_elliptic_geometry.py     # Elliptic geometry demonstration
│   ├── test_fermat_qmc_bias.py       # Tests for Fermat QMC module
│   └── test_*.py                     # Various test suites
├── reports/
│   └── qmc_statistical_results_899.csv  # Benchmark results for N=899
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

### Quick Example: κ-Weighted Rank-1 Lattices (NEW)
```python
import sys
sys.path.append('scripts')

from cognitive_number_theory.divisor_density import kappa
from scripts.qmc_engines import QMCConfig, make_engine

# Compute κ (kappa) curvature for a semiprime
N = 899  # 29 × 31
k = kappa(N)
print(f"κ({N}) = {k:.3f}")  # Lower κ → better factorization candidates

# Create κ-weighted rank-1 lattice
cfg = QMCConfig(
    dim=2,
    n=256,
    engine="rank1_lattice",
    lattice_generator="fibonacci",  # or "korobov", "cyclic"
    with_kappa_weight=True,         # Enable κ-weighting
    kappa_n=N,                      # Semiprime for κ computation
    seed=42
)

engine = make_engine(cfg)
points = engine.random(256)

# Points are now weighted by 1/κ(N) to bias toward low-curvature regions
print(f"Generated {len(points)} κ-weighted lattice points")

# Use with analysis script
# python scripts/qmc_factorization_analysis.py \
#     --semiprimes rsa100.txt rsa129.txt \
#     --engines rank1_korobov \
#     --with-kappa-weight \
#     --samples 5000 \
#     --out results.csv

# Run demo
# python kappa_lattice_demo.py
```


## Files Description

### Core Files
- **qmc_rsa_demo_v2.html**: Standalone interactive web demo with fair comparisons
- **qmc_factorization_analysis.py**: Python script for rigorous statistical analysis (now with `--with-kappa-weight` flag)
- **qmc_engines.py**: Enhanced QMC engine module with Sobol/Halton/Rank-1 lattice/EAS support (includes κ-weighting)
- **rank1_lattice.py**: Group-theoretic rank-1 lattice construction module
- **eas_factorize.py**: Elliptic Adaptive Search implementation
- **fermat_qmc_bias.py**: Fermat factorization with biased QMC sampling
- **cognitive_number_theory/divisor_density.py**: κ (kappa) curvature computation module (NEW)
- **kappa_lattice_demo.py**: Demonstration of κ-weighted rank-1 lattices (NEW)
- **qmc_statistical_results_899.csv**: Raw data from 1000 trials on N=899
- **QMC_RSA_SUMMARY.md**: Comprehensive summary of implementation, fixes, and findings
- **RANK1_LATTICE_INTEGRATION.md**: Documentation for rank-1 lattice integration

### Examples
- **qmc_directions_demo.py**: Comprehensive demonstration of enhanced QMC capabilities
- **eas_example.py**: Elliptic Adaptive Search usage examples and demonstrations
- **fermat_qmc_demo.py**: Demonstration of biased QMC for Fermat factorization
- **kappa_lattice_demo.py**: κ-weighted rank-1 lattice demonstration (NEW)

### Tests
- **test_large.py**: Original test for baseline methods
- **test_qmc_engines.py**: Tests for enhanced QMC engine module
- **test_replicated_qmc.py**: Tests for replicated QMC analysis with confidence intervals
- **test_rank1_lattice.py**: Unit tests for rank-1 lattice construction (includes κ-weighting tests)
- **test_rank1_integration.py**: Integration tests for rank-1 lattice with QMC framework
- **test_eas.py**: Unit tests for Elliptic Adaptive Search
- **test_fermat_qmc_bias.py**: Tests for Fermat factorization with biased QMC
- **test_kappa_integration.py**: Integration tests for κ-weighted lattices (NEW)
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