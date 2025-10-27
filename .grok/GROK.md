# Instructions for Grok CLI in QMC RSA Project

## Project Overview

This project implements and demonstrates the application of Quasi-Monte Carlo (QMC) variance reduction techniques to RSA factorization candidate sampling. It compares QMC methods against Monte Carlo (MC) baselines and explores φ-biased transformations for improved candidate generation. The project includes rigorous statistical comparisons, interactive web demos, and Python analysis scripts.

## Key Findings

- QMC provides measurable improvements over MC: 1.03× to 1.34× better unique candidates
- Improvements scale with semiprime size
- φ-bias currently reduces performance (needs refinement)
- Statistical significance confirmed with 1000 trials and 95% confidence intervals

## Project Structure

- `demos/`: Interactive HTML and React demos for visualization
  - `qmc_rsa_demo_v2.html`: Main standalone demo
  - Other variants for different approaches
- `docs/`: Documentation
  - `QMC_RSA_SUMMARY.md`: Detailed implementation and findings
- `scripts/`: Python analysis scripts
  - `qmc_factorization_analysis.py`: Benchmarking and statistical analysis
- `reports/`: Raw data from benchmarks
  - `qmc_statistical_results_899.csv`: Results for N=899
- `.grok/`: Configuration directory
  - `settings.json`: Model configuration
  - `GROK.md`: These instructions

## Guidelines for Code and File Operations

- Always review README.md and docs/QMC_RSA_SUMMARY.md before making changes
- When editing demos, test in a modern web browser to ensure functionality
- For script modifications, run the analysis script to verify statistical integrity
- Maintain the rigorous statistical methodology (bootstrap confidence intervals, etc.)
- Preserve the project's focus on research and benchmarking

## Tool Usage Recommendations

- Use `search` to find specific code patterns or functions across files
- Use `bash` for running Python scripts, checking file structures, or executing commands
- Use `view_file` to read files before editing
- Use `str_replace_editor` for precise edits to existing files
- Create new files with `create_file` only for genuinely new additions
- Use `create_todo_list` for complex tasks involving multiple steps

## Research Notes

- This is the first documented application of QMC to RSA factorization
- Supports Cranley-Patterson shifts for QMC variance estimation
- Future extensions: Sobol sequences, refined φ-bias, larger cryptographic numbers

## Citation

If referencing this work, cite as the first documented QMC application to RSA factorization candidate sampling.
