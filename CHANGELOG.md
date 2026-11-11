# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-11

### Added
- **Apple Metal GPU Acceleration**: High-performance parallel sorting and CDF computation on Apple Silicon unified memory using Apple's native MLX framework (`mlx`).
- **PyTorch MPS Fallback**: Support for Metal Performance Shaders (MPS) via PyTorch as a secondary Metal backend.
- **Compute Device Selection**: Added `--device` (`-d`) flag accepting `auto` (default), `metal`, and `cpu`.
- **Negative Column Indexing**: Support negative column indexes in `--column` (e.g. `-1` for the last column).
- **Modern Python Tooling**:
  - Adopted PEP 735 `[dependency-groups]` for development dependencies (`pytest`, `pytest-cov`, `ruff`).
  - Added platform-specific optional dependency `metal` for `mlx>=0.20.0` on macOS Apple Silicon.
- **Test Coverage**: Expanded test suite to 47 unit tests achieving 100% statement coverage.
- **Performance Benchmarks**: Documented 3.02x speedup on Apple M2 GPU vs NumPy CPU for large datasets in README.

### Changed
- **Stream Purity**: Redirected all warning, fallback, and diagnostic messages to `sys.stderr` so that `stdout` remains pure data when piped to files or downstream tools.
- **Memory Optimization**: Eliminated redundant string allocation in `write_output` when writing directly to disk via `np.savetxt`.
- **CI Modernization**: Updated GitHub Actions CI workflow to sync dependencies with `uv sync --all-groups --all-extras`.

### Fixed
- **Whitespace Multi-Column Parsing**: Fixed bug in `read_numeric_data` where omitting `--delimiter` caused all columns of whitespace-separated files to be flattened together, ignoring the `--column` parameter.
- **Whitespace Fallback Recovery**: Fixed fatal `ValueError` in `_manual_parse` when encountering whitespace-separated multi-column rows during fallback parsing.
- **Non-Destructive Exception Handling**: Replaced process termination via `sys.exit(1)` in `read_numeric_data` with standard `ValueError` exception raising.
- **Input Dimension Handling**: Enforced array flattening in `generate_cdf` to safely handle multidimensional or nested sequence inputs.

## [0.1.0] - 2026-11-11

### Added
- Initial release of CDF_Gen with NumPy empirical cumulative distribution function (CDF) calculation.
- Command-line interface with custom delimiters and column extraction.
- Unit test suite with pytest and code formatting with Ruff.
