"""
P-adic Hypothesis Experiment Package

This package implements a comprehensive experiment to test the hypothesis
that p-adic numbers are the natural completion of the geofac framework.

Modules:
--------
- padic: Core p-adic number theory operations
- experiment: Main experiment runner with 6 hypothesis tests
- visualize: Visualization generation for results

Usage:
------
    # Run the full experiment
    python experiments/padics_geofac_hypothesis/experiment.py
    
    # Generate visualizations
    python experiments/padics_geofac_hypothesis/visualize.py

Results:
--------
The hypothesis is NOT FALSIFIED. Strong evidence supports that:
- P-adic topology is native to the framework structure
- Ultrametric properties explain clustering behavior
- Hensel lifting explains solution propagation
- Geofac spines are p-adic expansions

See README.md for full details.
"""

from .padic import (
    p_adic_valuation,
    p_adic_distance,
    p_adic_expansion,
    p_adic_norm,
    hensel_lift,
    is_ultrametric_valid,
    compute_cauchy_sequence_convergence,
    analyze_geofac_spine,
    demonstrate_descent_chain
)

__version__ = '1.0.0'
__author__ = 'DMC RSA Research'

__all__ = [
    'p_adic_valuation',
    'p_adic_distance',
    'p_adic_expansion',
    'p_adic_norm',
    'hensel_lift',
    'is_ultrametric_valid',
    'compute_cauchy_sequence_convergence',
    'analyze_geofac_spine',
    'demonstrate_descent_chain'
]
