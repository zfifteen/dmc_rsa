#!/usr/bin/env python3
"""
QMC Integration Test Suite for Arctan-Refined Korobov Lattices
================================================================

This module implements rigorous integration tests to falsify the hypothesis
that arctan-refined curvature improves Korobov lattice performance for QMC
on periodic integrands.

Test functions include:
1. Genz test functions (standard QMC benchmarks)
2. Periodic trigonometric integrands
3. Product of cosines (highly periodic)
4. Custom periodic functions with varying smoothness

Metrics:
- Integration error (vs analytical solution)
- Variance across independent trials
- Convergence rate analysis

Author: Z-Mode experiment framework
Date: November 2025
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from arctan_curvature import generate_korobov_lattice, measure_lattice_quality


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests"""
    n_points: int = 256          # Number of QMC points
    dimension: int = 2           # Integration dimension
    n_trials: int = 100          # Number of independent trials
    use_arctan: bool = True      # Use arctan-refined curvature
    alpha: float = 1.0           # Arctan scaling factor
    seed: int = 42               # Random seed


class PeriodicTestFunction:
    """Base class for periodic test functions"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.name = "BaseFunction"
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate function at points x of shape (n, d)"""
        raise NotImplementedError
    
    def analytical_integral(self) -> float:
        """Return analytical integral over [0,1]^d if known"""
        raise NotImplementedError


class ProductCosine(PeriodicTestFunction):
    """
    f(x) = ∏_{i=1}^d cos(2π x_i)
    
    Highly periodic function. Analytical integral over [0,1]^d = 0.
    This is a classic test for QMC on periodic functions.
    """
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.name = "ProductCosine"
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate product of cosines"""
        return np.prod(np.cos(2 * np.pi * x), axis=1)
    
    def analytical_integral(self) -> float:
        """Analytical integral = 0"""
        return 0.0


class SmoothPeriodic(PeriodicTestFunction):
    """
    f(x) = ∏_{i=1}^d (1 + 0.5 * sin(2π x_i))
    
    Smooth periodic function with positive values.
    Analytical integral over [0,1]^d = 1.
    """
    
    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.name = "SmoothPeriodic"
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate smooth periodic function"""
        return np.prod(1.0 + 0.5 * np.sin(2 * np.pi * x), axis=1)
    
    def analytical_integral(self) -> float:
        """Analytical integral = 1"""
        return 1.0


class MultiFrequencyCosine(PeriodicTestFunction):
    """
    f(x) = cos(2π Σ_{i=1}^d w_i x_i)
    
    Multi-frequency periodic function with weights w_i.
    Tests lattice performance on mixed frequencies.
    """
    
    def __init__(self, dimension: int, weights: Optional[np.ndarray] = None):
        super().__init__(dimension)
        self.name = "MultiFrequencyCosine"
        
        if weights is None:
            # Use Fibonacci-like weights for varying frequencies
            self.weights = np.array([1.0 * (1.618 ** i) for i in range(dimension)])
        else:
            self.weights = weights
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate multi-frequency cosine"""
        weighted_sum = np.sum(x * self.weights, axis=1)
        return np.cos(2 * np.pi * weighted_sum)
    
    def analytical_integral(self) -> float:
        """Analytical integral ≈ 0 (oscillates)"""
        return 0.0


class GenzContinuous(PeriodicTestFunction):
    """
    Genz Continuous test function:
    f(x) = exp(-Σ_{i=1}^d c_i |x_i - w_i|)
    
    Standard QMC benchmark with varying difficulty based on c_i parameters.
    """
    
    def __init__(self, dimension: int, difficulty: float = 5.0):
        super().__init__(dimension)
        self.name = "GenzContinuous"
        self.c = np.ones(dimension) * difficulty
        self.w = np.ones(dimension) * 0.5
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Genz continuous function"""
        diff = np.abs(x - self.w)
        exponent = -np.sum(self.c * diff, axis=1)
        return np.exp(exponent)
    
    def analytical_integral(self) -> float:
        """Analytical integral (product form)"""
        integral = 1.0
        for c_i in self.c:
            integral *= (2.0 - 2.0 * np.exp(-c_i)) / c_i
        return integral


def qmc_integrate(points: np.ndarray, 
                 func: PeriodicTestFunction) -> Dict[str, float]:
    """
    Perform QMC integration using given point set.
    
    Args:
        points: QMC points of shape (n, d) in [0,1)^d
        func: Test function to integrate
        
    Returns:
        Dictionary with integration results
    """
    n_points = points.shape[0]
    
    # Evaluate function at QMC points
    f_values = func.evaluate(points)
    
    # QMC estimate: mean of function values
    integral_estimate = np.mean(f_values)
    
    # Estimate variance (not true variance, but useful metric)
    # For replicated QMC, we'd use multiple randomizations
    estimate_variance = np.var(f_values) / n_points
    
    # Analytical integral (if known)
    analytical = func.analytical_integral()
    error = abs(integral_estimate - analytical)
    
    return {
        'estimate': integral_estimate,
        'analytical': analytical,
        'error': error,
        'variance': estimate_variance,
        'n_points': n_points
    }


def run_integration_comparison(
    test_func: PeriodicTestFunction,
    n_points: int = 256,
    n_trials: int = 100,
    alpha_values: list = [0.0, 0.5, 1.0, 2.0],
    seed: int = 42
) -> Dict[str, any]:
    """
    Compare integration performance of baseline vs arctan-refined Korobov lattices.
    
    Args:
        test_func: Test function to integrate
        n_points: Number of QMC points (should be prime for Korobov)
        n_trials: Number of independent trials
        alpha_values: List of alpha values to test (0.0 = baseline)
        seed: Random seed
        
    Returns:
        Dictionary with comparison results
    """
    rng = np.random.default_rng(seed)
    d = test_func.dimension
    
    # Ensure n_points is prime for good Korobov properties
    from sympy import nextprime
    if n_points < 2:
        n_points = 2
    n_points = int(nextprime(n_points - 1))
    
    results = {
        'function': test_func.name,
        'dimension': d,
        'n_points': n_points,
        'n_trials': n_trials,
        'alphas': {}
    }
    
    for alpha in alpha_values:
        use_arctan = (alpha > 0)
        
        errors = []
        variances = []
        estimates = []
        
        for trial in range(n_trials):
            # Generate lattice with different random shifts
            # (This provides independent trials for variance estimation)
            points = generate_korobov_lattice(n_points, d, 
                                             use_arctan=use_arctan, 
                                             alpha=alpha)
            
            # Apply random shift (Cranley-Patterson rotation)
            shift = rng.uniform(0, 1, d)
            points = (points + shift) % 1.0
            
            # Integrate
            result = qmc_integrate(points, test_func)
            
            errors.append(result['error'])
            variances.append(result['variance'])
            estimates.append(result['estimate'])
        
        # Compute statistics
        errors_arr = np.array(errors)
        variances_arr = np.array(variances)
        estimates_arr = np.array(estimates)
        
        results['alphas'][alpha] = {
            'mean_error': np.mean(errors_arr),
            'std_error': np.std(errors_arr),
            'median_error': np.median(errors_arr),
            'mean_variance': np.mean(variances_arr),
            'std_variance': np.std(variances_arr),
            'rmse': np.sqrt(np.mean(errors_arr ** 2)),
            'estimates_mean': np.mean(estimates_arr),
            'estimates_std': np.std(estimates_arr)
        }
    
    return results


def compute_variance_reduction(results: Dict[str, any]) -> Dict[str, float]:
    """
    Compute variance reduction factors compared to baseline (alpha=0).
    
    Args:
        results: Results from run_integration_comparison
        
    Returns:
        Dictionary with variance reduction metrics
    """
    baseline_variance = results['alphas'][0.0]['mean_variance']
    baseline_error = results['alphas'][0.0]['mean_error']
    
    reductions = {}
    for alpha, stats in results['alphas'].items():
        if alpha == 0.0:
            reductions[alpha] = {
                'variance_reduction_pct': 0.0,
                'error_reduction_pct': 0.0
            }
        else:
            var_reduction = (baseline_variance - stats['mean_variance']) / baseline_variance * 100
            err_reduction = (baseline_error - stats['mean_error']) / baseline_error * 100
            
            reductions[alpha] = {
                'variance_reduction_pct': var_reduction,
                'error_reduction_pct': err_reduction
            }
    
    return reductions


if __name__ == "__main__":
    print("QMC Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Product Cosine (highly periodic)
    print("\n1. Product Cosine Function (d=2)")
    test_func = ProductCosine(dimension=2)
    results = run_integration_comparison(
        test_func,
        n_points=127,
        n_trials=50,
        alpha_values=[0.0, 0.5, 1.0],
        seed=42
    )
    
    reductions = compute_variance_reduction(results)
    
    print(f"  Function: {results['function']}")
    print(f"  Points: {results['n_points']}, Trials: {results['n_trials']}")
    print(f"  Analytical integral: {test_func.analytical_integral()}")
    
    for alpha in [0.0, 0.5, 1.0]:
        stats = results['alphas'][alpha]
        red = reductions[alpha]
        label = "Baseline" if alpha == 0.0 else f"Arctan (α={alpha})"
        print(f"\n  {label}:")
        print(f"    Mean error:     {stats['mean_error']:.6e}")
        print(f"    Mean variance:  {stats['mean_variance']:.6e}")
        print(f"    RMSE:           {stats['rmse']:.6e}")
        if alpha > 0:
            print(f"    Variance reduction: {red['variance_reduction_pct']:+.2f}%")
            print(f"    Error reduction:    {red['error_reduction_pct']:+.2f}%")
    
    # Test 2: Smooth Periodic
    print("\n2. Smooth Periodic Function (d=2)")
    test_func2 = SmoothPeriodic(dimension=2)
    results2 = run_integration_comparison(
        test_func2,
        n_points=127,
        n_trials=50,
        alpha_values=[0.0, 1.0],
        seed=43
    )
    
    reductions2 = compute_variance_reduction(results2)
    
    print(f"  Function: {results2['function']}")
    print(f"  Analytical integral: {test_func2.analytical_integral()}")
    
    for alpha in [0.0, 1.0]:
        stats = results2['alphas'][alpha]
        red = reductions2[alpha]
        label = "Baseline" if alpha == 0.0 else f"Arctan (α={alpha})"
        print(f"\n  {label}:")
        print(f"    Mean error:     {stats['mean_error']:.6e}")
        print(f"    Variance reduction: {red['variance_reduction_pct']:+.2f}%")
