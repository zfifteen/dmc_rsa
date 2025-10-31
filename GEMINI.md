# Gemini Code Assistant Context

This document provides a comprehensive overview of the `qmc_rsa` repository to guide the Gemini code assistant. The repository contains a primary research project focused on RSA factorization using Quasi-Monte Carlo (QMC) methods, and it integrates with other related research projects, notably `wave-crispr-signal`.

## Repository Overview

This repository is a multi-faceted research environment centered around number theory, cryptography, and signal processing. It consists of two main projects:

1.  **`qmc_rsa`**: The core project focused on applying advanced mathematical techniques to RSA factorization.
2.  **`wave-crispr-signal`**: A sub-project exploring a "Z-Framework" in the context of biological signals, which also provides functionalities for the `qmc_rsa` project.

The projects are primarily written in Python and rely on a rich ecosystem of scientific computing libraries.

---

## 1. QMC RSA Factorization (`qmc_rsa`)

This is the main project in the repository. It implements and benchmarks various methods for RSA factorization, with a special focus on Quasi-Monte Carlo (QMC) techniques.

### Project Purpose

*   To explore and document the application of QMC variance reduction to RSA factorization candidate sampling.
*   To compare the performance of QMC methods against traditional Monte Carlo (MC) approaches.
*   To develop and test advanced factorization techniques, including Rank-1 lattices, Elliptic Adaptive Search (EAS), and biased QMC for Fermat factorization.

### Technologies

*   **Language:** Python 3.7+
*   **Core Libraries:**
    *   `numpy>=1.20.0`
    *   `pandas>=1.3.0`
    *   `scipy>=1.7.0`
    *   `sympy>=1.9`

### How to Build and Run

**Setup:**

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

**Running Analysis and Demos:**

The project includes a variety of scripts for running analysis, benchmarks, and demonstrations.

*   **Main analysis script:**
    ```bash
    python scripts/qmc_factorization_analysis.py
    ```

*   **Demonstration scripts:**
    ```bash
    # Enhanced QMC capabilities
    python examples/qmc_directions_demo.py

    # Rank-1 lattice examples
    python examples/rank1_lattice_example.py

    # Fermat factorization with biased QMC
    python examples/fermat_qmc_demo.py

    # Elliptic geometry demonstration
    python scripts/demo_elliptic_geometry.py
    ```

*   **Interactive Web Demos:**
    Open `demos/qmc_rsa_demo_v2.html` in a web browser to interact with the factorization methods visually.

### Testing

The project has a suite of tests to ensure the correctness of the implemented algorithms.

*   **Run specific test suites:**
    ```bash
    # Test Rank-1 lattice constructions
    python scripts/test_rank1_lattice.py

    # Test Fermat QMC module
    python scripts/test_fermat_qmc_bias.py
    ```

---

## 2. Wave CRISPR Signal (`wave-crispr-signal`)

This is a sub-project that appears to be a research framework for analyzing biological signals, with a component called the "Z-Framework". It is used as a dependency by the `qmc_rsa` project.

### Project Purpose

*   To analyze biological signals using advanced mathematical and signal processing techniques.
*   The "Z-Framework" seems to be a core component, providing functions like `theta_prime` that are used to bias the QMC sampling in the `qmc_rsa` project.

### Technologies

*   **Language:** Python
*   **Core Libraries:** A comprehensive stack for scientific computing, including `numpy`, `scipy`, `pandas`, `biopython`, `scikit-learn`, `matplotlib`, `torch`, and `pydicom`.

### How to Build and Run

**Setup:**

The project uses a `Makefile` to streamline common tasks. To install dependencies:

```bash
make install
```

**Running Tasks:**

The `Makefile` provides several targets for running tests and experiments.

*   **Run the full test suite:**
    ```bash
    make test
    ```

*   **Run smoke tests for continuous integration:**
    ```bash
    make smoke
    ```

*   **Run specific experiments:**
    The `Makefile` contains targets like `run-mve`, `run-mri-z5d`, and `run-fus-enhancer` for running various experiments.

### Integration with `qmc_rsa`

The `TASK.md` file indicates that the `qmc_rsa` project uses functions from `wave-crispr-signal` (e.g., `theta_prime`) and `cognitive-number-theory` (e.g., `kappa`) to create a "Z-bias" for its QMC sampler. This suggests a cross-disciplinary research effort to improve cryptographic algorithms with concepts from other scientific fields.
