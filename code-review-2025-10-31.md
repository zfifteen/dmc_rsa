# QMC RSA – Comprehensive Code Review
Date: 2025-10-31
Reviewer: GitHub Copilot CLI
Status: **Blocker-level issues resolved as of 2025-10-31.**

## Executive summary
**Update: All blocker-level issues listed below have been fixed.** The repository presents ambitious research features (QMC engines, rank-1 lattices, EAS, biased Fermat sampling). The codebase is now in a runnable state after resolving multiple syntax/merge errors, import path mismatches, and API inconsistencies. The next priorities are addressing high-severity issues like reproducibility and API consistency.

## Blockers (FIXED)
- **scripts/qmc_engines.py**
  - **FIXED:** Corrupted/merged code sections and duplicate imports:
    - Z-framework imports were deduplicated.
  - **FIXED:** Broken function definitions and interleaved text:
    - `validate_sobol_sample_size(...)` and `z_bias(...)` were separated and corrected.
  - **FIXED:** Dataclass duplication and conflicting fields:
    - `QMCConfig` fields were deduplicated and `int | None` was replaced with `Optional[int]` for Python 3.7+ compatibility.
  - **FIXED:** Rank1LatticeEngine constructor argument list has duplicated keys and missing commas.
  - **FIXED:** Mixed/invalid engine names: Routing for `rank1_lattice` and `elliptic_cyclic` was clarified.
- **scripts/rank1_lattice.py**
  - **FIXED:** Dataclass duplicates and conflicting options:
    - `Rank1LatticeConfig` was consolidated.
  - **FIXED:** Control-flow/duplication bugs in `generate_rank1_lattice`:
    - Logic was refactored to remove duplicate branches.
  - **FIXED:** Non-ASCII identifier in code (`θ = ...`) was replaced with `theta`.
- **scripts/qmc_factorization_analysis.py**
  - **FIXED:** Typo merge: `import argparseimport argparse` was corrected to `import argparse`, and `import warnings` was added.
  - **FIXED:** CLI/main duplication: Duplicate `main()` call was removed.
  - **FIXED:** Writes to `outputs/` without ensuring directory exists. Added `os.makedirs` to create the directory.
- **Import path mismatches (modules can’t import):**
  - **FIXED:** `cognitive-number-theory/` and `wave-crispr-signal/` directories were renamed to `cognitive_number_theory` and `wave_crispr_signal` to match import statements.
- **Requirements vs code**
  - **FIXED:** Code was updated to use `Optional[int]` to maintain Python 3.7+ compatibility.

## High severity
- Reproducibility/PRNG: Several places use global `np.random.seed(...)` and `np.random.random(...)`; prefer per-instance `Generator` objects (PCG64) to avoid cross-test bleed and to support parallelism.
- Discrepancy/star discrepancy estimators are simplistic and O(n^2) with fixed 100×100 sampling; acceptable for demos but should be clearly labeled approximate and bounded by `min(n, 100)` samples (already partly done) and vectorized for speed.
- API inconsistencies and naming:
  - Engine strings: README/docs list `elliptic_cyclic`, `elliptic`, `rank1_lattice`, `spiral_conical` inconsistently across files; normalize single canonical names and deprecate aliases.
  - QMCConfig options include EAS- and lattice-specific knobs in one struct; consider separating per-engine configs or namespacing to reduce invalid combinations.
- Tests rely on printing rather than assertions for behavior in many places; some tests import missing modules and would immediately fail given current parser errors.

## Medium severity
- Performance
  - `estimate_l2_discrepancy` double loop can be vectorized; `estimate_covering_radius` loops over up to 200 points × 1000 tests; consider KD-tree or batch vectorization.
  - EAS uses trial division primality; fine for small numbers but document limits; consider Miller–Rabin for broader ranges.
- Robustness
  - `map_points_to_candidates`: forward adjustment loop (max 4 steps) to match residue classes could still miss intended residue for certain inputs; document guarantee bounds or adjust logic.
  - Several scripts assume presence of output dirs and will crash without `os.makedirs(..., exist_ok=True)`.
- Documentation drift
  - README markets “October 2025” features as “NEW” but code does not compile; align claims with actual functionality behind feature flags or branch.

## Low severity
- Mixed British/American spelling and inconsistent terminology (`lattice_generator` vs `generator_type`).
- Overuse of inline Unicode/math symbols in code and comments may hurt portability.
- Tests require `pytest` but it’s not in requirements.txt; also heavy use of prints makes them noisy in CI.

## Security/compliance
- No secrets found. Factorization demos are research-level; ensure README clarifies no claims about breaking cryptographic-strength RSA and that experiments are on small semiprimes.
- Licensing: README says “see individual files,” but there is no root LICENSE at repo top-level; consider adding one.

## Recommendations (prioritized)
**1) Unblock imports/build (COMPLETED)**
- **COMPLETED:** Fix scripts/qmc_engines.py
- **COMPLETED:** Fix scripts/rank1_lattice.py
- **COMPLETED:** Fix scripts/qmc_factorization_analysis.py
- **COMPLETED:** Import paths
- **COMPLETED:** Requirements/README

2) Testing/CI hygiene
- Make tests assert-driven; reduce prints, or gate verbosity behind `--verbose`.
- Add a quick syntax gate (`python -m compileall scripts`) in CI. Ensure outputs/ dir exists during tests.

3) API cleanup
- Split engine-specific config into separate dataclasses or validate QMCConfig per-engine to prevent invalid combos.
- Normalize engine string constants and document them in one place.

4) Performance/robustness
- Vectorize discrepancy/covering computations; consider sampling fewer points for large n.
- Use `numpy.random.Generator(PCG64)` instances throughout; avoid global state.

## Suggested sanity checks after fixes
- Run: `python -m compileall scripts` (should succeed)
- Run a minimal analysis: `python scripts/qmc_factorization_analysis.py --semiprimes rsa100.txt --num-samples 128 --replicates 8 --output results.csv` (should run or gracefully error if optional modules missing)
- Run tests that don’t require Z-framework: `python scripts/test_qmc_engines.py` and `python scripts/test_rank1_lattice.py`

## Documentation alignment
- Update README “Python Analysis” to require Python 3.10+ (if keeping union types) and add pytest as dev dependency.
- Temper “NEW” feature claims until code compiles; optionally move bleeding-edge work to a feature branch and mark as experimental.

## Notable positives
- Ambitious scope with clear research framing and many well-commented doc files.
- Good intentions around randomized QMC replicates, Owen scrambling, CI-ready statistical methodology.
- Mapping from [0,1]^2 to candidate integers with smooth-ish edges is a sensible approach.

## File-by-file highlights (non-exhaustive)
- README.md: Clear overview; environment claims (3.7+) conflict with code (3.10+).
- scripts/qmc_engines.py: Blocker—parse errors, duplicates, and API drift; primary repair target.
- scripts/rank1_lattice.py: Duplicate config fields, mixed generator names, and unicode identifiers.
- scripts/qmc_factorization_analysis.py: Import typos, missing `warnings` import, double `main()`, assumes outputs/.
- scripts/eas_factorize.py: Generally coherent; consider per-instance RNG and Miller–Rabin if scaling up.
- scripts/fermat_qmc_bias.py: Reasonable API; ensure `sympy` is in requirements (it is). Consider seeding via `Generator`.
- scripts/test_*: Many rely on prints; `test_z_bias.py` uses pytest but repo doesn’t list pytest in requirements.

## Risk assessment
- Current mainline is not runnable; changes are mostly mechanical cleanup but touch core modules—perform in small PRs with targeted tests.
- Optional modules (Z-framework, cognitive-number-theory) are not present as importable Python packages; keep them optional and guarded.

---
End of review.
---