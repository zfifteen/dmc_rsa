"""
P-adic Number Theory Module

Implements p-adic valuations, distances, expansions, and Hensel lifting
to test the hypothesis that p-adics are the natural completion of the 
geofac framework.

Key concepts:
- p-adic valuation: vₚ(n) = max{k : p^k | n}
- p-adic distance: d(a,b) = p^(-vₚ(a-b))
- p-adic expansion: representing n in base-p (infinite to the left)
- Hensel lifting: lifting solutions from ℤ/p^k to ℤ/p^(k+1)
"""

from typing import List, Tuple, Optional
import sympy


def p_adic_valuation(n: int, p: int) -> int:
    """
    Compute p-adic valuation vₚ(n) = max{k : p^k | n}.
    
    The p-adic valuation measures "how divisible" n is by p.
    For n = 0, we define vₚ(0) = ∞ (represented as -1 for practical purposes).
    
    Args:
        n: Integer to evaluate
        p: Prime base
        
    Returns:
        p-adic valuation (or -1 for n=0 representing ∞)
        
    Examples:
        >>> p_adic_valuation(8, 2)  # 8 = 2^3
        3
        >>> p_adic_valuation(12, 2)  # 12 = 2^2 * 3
        2
        >>> p_adic_valuation(15, 5)  # 15 = 3 * 5
        1
    """
    if n == 0:
        return -1  # Represents infinity
    
    if not sympy.isprime(p):
        raise ValueError(f"p must be prime, got {p}")
    
    n = abs(n)
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def p_adic_distance(a: int, b: int, p: int) -> float:
    """
    Compute p-adic distance d(a,b) = p^(-vₚ(a-b)).
    
    This is an ultrametric: d(a,c) ≤ max(d(a,b), d(b,c)).
    Two numbers are "close" in the p-adic metric if their difference
    is highly divisible by p.
    
    Args:
        a, b: Integers
        p: Prime base
        
    Returns:
        p-adic distance
        
    Examples:
        >>> p_adic_distance(8, 0, 2)  # vₚ(8) = 3, so d = 2^-3 = 0.125
        0.125
        >>> p_adic_distance(10, 2, 2)  # vₚ(8) = 3, so d = 2^-3 = 0.125
        0.125
    """
    if a == b:
        return 0.0
    
    val = p_adic_valuation(a - b, p)
    if val == -1:  # a - b = 0, already handled above
        return 0.0
    
    return p ** (-val)


def p_adic_expansion(n: int, p: int, num_digits: int = 20) -> List[int]:
    """
    Compute p-adic expansion of n to num_digits.
    
    For positive integers, this gives the standard base-p representation,
    but conceptually it extends infinitely to the left (toward higher powers).
    
    The expansion is: n = a₀ + a₁·p + a₂·p² + ... where 0 ≤ aᵢ < p
    
    Args:
        n: Integer to expand (must be >= 0)
        p: Prime base
        num_digits: Number of digits to compute
        
    Returns:
        List of digits [a₀, a₁, a₂, ...] in base p
        
    Examples:
        >>> p_adic_expansion(13, 2, 5)  # 13 = 1 + 0·2 + 1·2² + 1·2³ = 1101₂
        [1, 0, 1, 1, 0]
        >>> p_adic_expansion(10, 5, 3)  # 10 = 0 + 2·5 = 20₅
        [0, 2, 0]
    """
    if n < 0:
        raise ValueError("p-adic expansion only implemented for n >= 0")
    
    if not sympy.isprime(p):
        raise ValueError(f"p must be prime, got {p}")
    
    digits = []
    remainder = n
    for _ in range(num_digits):
        digit = remainder % p
        digits.append(digit)
        remainder //= p
        if remainder == 0:
            break
    
    # Pad with zeros if needed
    while len(digits) < num_digits:
        digits.append(0)
    
    return digits


def p_adic_from_expansion(digits: List[int], p: int) -> int:
    """
    Convert p-adic expansion back to integer.
    
    Args:
        digits: List of base-p digits [a₀, a₁, a₂, ...]
        p: Prime base
        
    Returns:
        Integer value n = Σ aᵢ·p^i
    """
    result = 0
    for i, digit in enumerate(digits):
        result += digit * (p ** i)
    return result


def p_adic_norm(n: int, p: int) -> float:
    """
    Compute p-adic absolute value (norm) |n|ₚ = p^(-vₚ(n)).
    
    This satisfies |ab|ₚ = |a|ₚ · |b|ₚ (multiplicative).
    
    Args:
        n: Integer
        p: Prime base
        
    Returns:
        p-adic norm
        
    Examples:
        >>> p_adic_norm(8, 2)  # |8|₂ = 2^-3 = 0.125
        0.125
        >>> p_adic_norm(5, 5)  # |5|₅ = 5^-1 = 0.2
        0.2
    """
    if n == 0:
        return 0.0
    
    val = p_adic_valuation(n, p)
    if val == -1:
        return 0.0
    return p ** (-val)


def hensel_lift(f, df, a: int, p: int, k: int) -> Optional[int]:
    """
    Hensel lifting: lift a solution modulo p^k to modulo p^(k+1).
    
    Given f(a) ≡ 0 (mod p^k) and df(a) ≢ 0 (mod p), 
    find b such that f(b) ≡ 0 (mod p^(k+1)) and b ≡ a (mod p^k).
    
    The lift is: b = a - f(a)/f'(a) (mod p^(k+1))
    
    Args:
        f: Function (should take int and return int)
        df: Derivative of f
        a: Current solution modulo p^k
        p: Prime base
        k: Current exponent
        
    Returns:
        Lifted solution modulo p^(k+1), or None if lifting fails
        
    Examples:
        >>> # Lift sqrt(1) mod 4 to mod 8
        >>> f = lambda x: x**2 - 1
        >>> df = lambda x: 2*x
        >>> hensel_lift(f, df, 1, 2, 2)  # 1^2 ≡ 1 (mod 4), lift to mod 8
        1
    """
    pk = p ** k
    pk1 = p ** (k + 1)
    
    fa = f(a)
    dfa = df(a)
    
    # Check that we have a valid solution mod p^k
    if fa % pk != 0:
        return None
    
    # Check that derivative is non-zero mod p (required for lifting)
    if dfa % p == 0:
        return None
    
    # Compute the lift
    # b = a - f(a) * (df(a))^(-1) mod p^(k+1)
    try:
        # Find modular inverse of df(a) mod p
        dfa_inv = pow(dfa, -1, p)
        t = (fa // pk) * dfa_inv % p
        b = (a - t * pk) % pk1
        return b
    except ValueError:
        return None


def is_ultrametric_valid(a: int, b: int, c: int, p: int, epsilon: float = 1e-10) -> bool:
    """
    Verify the strong triangle inequality (ultrametric property):
    d(a,c) ≤ max(d(a,b), d(b,c))
    
    Args:
        a, b, c: Three integers
        p: Prime base
        epsilon: Tolerance for floating point comparison
        
    Returns:
        True if ultrametric inequality holds
    """
    dac = p_adic_distance(a, c, p)
    dab = p_adic_distance(a, b, p)
    dbc = p_adic_distance(b, c, p)
    
    return dac <= max(dab, dbc) + epsilon


def compute_cauchy_sequence_convergence(sequence: List[int], p: int) -> List[float]:
    """
    Check if a sequence is Cauchy in the p-adic metric.
    
    A sequence is Cauchy if d(xₙ, xₘ) → 0 as n,m → ∞.
    Returns the distances d(xₙ, x_{n+1}) which should decrease.
    
    Args:
        sequence: List of integers forming the sequence
        p: Prime base
        
    Returns:
        List of consecutive distances [d(x₀,x₁), d(x₁,x₂), ...]
    """
    distances = []
    for i in range(len(sequence) - 1):
        dist = p_adic_distance(sequence[i], sequence[i+1], p)
        distances.append(dist)
    return distances


def analyze_geofac_spine(n: int, p: int, max_level: int = 10) -> List[Tuple[int, int, float]]:
    """
    Analyze the "geofac spine" - the tower of prime powers dividing n.
    
    For a number n with p-adic valuation v, this creates the sequence
    n, n, n, ... which in the p-adic completion corresponds to the spine
    going upward in prime powers.
    
    Returns information about the p-adic structure at each level.
    
    Args:
        n: Integer to analyze
        p: Prime base
        max_level: Maximum power of p to consider
        
    Returns:
        List of tuples (k, n mod p^k, |n|ₚ at level k)
    """
    val = p_adic_valuation(n, p)
    spine = []
    
    for k in range(1, min(max_level, val + 2) if val >= 0 else max_level):
        pk = p ** k
        residue = n % pk
        norm = p_adic_norm(n, p) if k == 1 else p ** (-(val if val >= 0 else 0))
        spine.append((k, residue, norm))
    
    return spine


def p_adic_series_sum(coeffs: List[int], p: int) -> Tuple[int, int]:
    """
    Compute the finite sum of a p-adic series Σ aᵢ·p^i.
    
    Returns the sum and the p-adic valuation of the sum.
    
    Args:
        coeffs: Coefficients [a₀, a₁, a₂, ...]
        p: Prime base
        
    Returns:
        Tuple of (sum, valuation)
    """
    total = p_adic_from_expansion(coeffs, p)
    val = p_adic_valuation(total, p)
    return total, val


def demonstrate_descent_chain(start: int, p: int, steps: int = 10) -> List[int]:
    """
    Create a descent chain in the p-adic topology.
    
    This simulates the "infinite descent" pattern by repeatedly
    adding higher powers of p to show convergence.
    
    Args:
        start: Starting integer
        p: Prime base
        steps: Number of steps in the descent
        
    Returns:
        List of integers forming the descent chain
    """
    chain = [start]
    current = start
    
    for k in range(1, steps):
        # Add a multiple of p^k to create descent
        # This makes consecutive terms closer in p-adic distance
        current = current + (k % p) * (p ** k)
        chain.append(current)
    
    return chain
