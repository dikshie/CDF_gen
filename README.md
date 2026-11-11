# CDF_Gen

A lightweight, high-performance command-line utility and Python module to compute the empirical **Cumulative Distribution Function (CDF)** from numeric datasets.

Built with [Apple Metal GPU acceleration](https://developer.apple.com/metal/) via [Apple MLX](https://github.com/ml-explore/mlx) / [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html) for lightning-fast hardware execution on macOS Apple Silicon, and [NumPy](https://numpy.org/) for CPU vectorized calculations and optimized file I/O.

---

## Features

- **Apple Metal GPU Acceleration**: Automatically leverages Apple Silicon Metal GPUs (via Apple's native MLX framework or PyTorch MPS) for high-throughput parallel sorting and CDF computation on unified memory.
- **High Performance**: 3x+ speedup on large datasets with zero-copy unified memory GPU acceleration, effortlessly scaling to tens of millions of records.
- **Flexible Input Formats**: Supports single-column files, whitespace-delimited files, and multi-column formats (e.g., CSV, TSV) with selectable column indices (including negative indices like `-1`).
- **Robust Data Parsing**:
  - Automatically ignores comment lines starting with `#` and blank lines.
  - Filters out `NaN` and infinite (`Inf`, `-Inf`) values.
  - Automatically falls back to manual line-by-line parsing if `numpy.loadtxt` encounters malformed or ragged rows.
  - Non-destructive error handling raising standard Python exceptions without abruptly terminating the process.
- **Clean Output Streams**: Diagnostic messages and parsing warnings are written to `stderr`, keeping `stdout` completely clean for UNIX piping (`python cdf.py data.txt > cdf.tsv`).
- **Tab-Separated Output**: Outputs formatted `probability\tvalue` pairs to `stdout` or directly to an output file.
- **Modern Python Tooling**: Managed with [`uv`](https://github.com/astral-sh/uv), linted and formatted with [`ruff`](https://astral.sh/ruff), and thoroughly tested with 100% coverage via [`pytest`](https://docs.pytest.org/).

---

## Mathematical Background

The empirical cumulative distribution function (eCDF) $F_n(x)$ for a sample of $n$ observations $X_1, X_2, \dots, X_n$ is defined as:

$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{X_i \le x}$$

Given sorted observations $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$, the probability associated with each observation $x_{(i)}$ is:

$$P(X \le x_{(i)}) = \frac{i}{n}, \quad \text{for } i = 1, 2, \dots, n$$

The tool outputs these $(P(X \le x_{(i)}), x_{(i)})$ pairs formatted to 4 decimal places.

---

## Performance Benchmarks

Tested on Apple M2 (20,000,000 random floating-point values):

| Backend | Hardware | Execution Time | Speedup |
|---|---|---|---|
| **Apple MLX** | **Apple Silicon GPU (Metal)** | **0.152s** | **3.02x** |
| NumPy | Apple M2 CPU | 0.459s | 1.00x |

---

## Requirements & Installation

### Requirements

- Python >= 3.9
- [NumPy](https://numpy.org/)
- *(Optional, for Apple Metal GPU acceleration)*: [mlx](https://github.com/ml-explore/mlx) on Apple Silicon macOS (`arm64`) or [torch](https://pytorch.org/) with MPS.

### Setup

Install with [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
# Sync dependencies and development tools
uv sync --all-groups

# Enable Apple Metal GPU support on Apple Silicon macOS
uv sync --all-groups --extra metal
```

Or run directly without installation:

```bash
uv run python cdf.py <filename>
```

---

## Usage

### Command Line Interface

```bash
./cdf.py <filename> [options]
```

#### Options

| Option | Flag | Description | Default |
|---|---|---|---|
| `filename` | (positional) | Path to input numeric data file | *Required* |
| `--delimiter` | | Column delimiter string (e.g., `,`, `\t`) | `None` (whitespace) |
| `--column` | | 0-based (or negative) column index to extract | `0` |
| `--output` | `-o` | Output file path | `stdout` |
| `--device` | `-d` | Compute device (`auto`, `metal`, `cpu`) | `auto` |
| `--help` | `-h` | Show usage help | |

### Examples

#### 1. Basic Single-Column Input (with Auto Apple Metal GPU Acceleration)

Given an input file `data.txt`:
```text
# Sample measurements
10.5
20.0
5.0
15.0
```

Run:
```bash
./cdf.py data.txt
```

Output:
```text
0.2500	5.0000
0.5000	10.5000
0.7500	15.0000
1.0000	20.0000
```

#### 2. Multi-Column Whitespace or CSV File

Extract column 1 from whitespace-delimited or comma-separated files:
```bash
# Whitespace-delimited
./cdf.py measurements.txt --column 1

# Comma-delimited CSV
./cdf.py measurements.csv --delimiter ',' --column 1
```

#### 3. Forcing Apple Metal GPU or CPU

```bash
# Explicitly use Apple Metal GPU
./cdf.py large_data.txt --device metal -o cdf_gpu.tsv

# Force CPU (NumPy) computation
./cdf.py large_data.txt --device cpu -o cdf_cpu.tsv
```

#### 4. Save Output or Pipeline into Downstream Tools

```bash
# Write directly to file
./cdf.py data.txt -o cdf_output.tsv

# Or pipe clean stdout to file (warnings and logs go safely to stderr)
./cdf.py data.txt > cdf_output.tsv
```

### Python API

You can import and use `cdf` functions directly in your Python code:

```python
import numpy as np
from cdf import generate_cdf, read_numeric_data, write_output

# Read data from file (supports custom delimiters and column indices)
data = read_numeric_data("data.txt", delimiter=None, column=0)

# Compute CDF with Apple Metal GPU acceleration (or fallback to CPU)
probs, sorted_values = generate_cdf(data, device="auto")

# Output or inspect
for p, val in zip(probs, sorted_values):
    print(f"{p:.4f}\t{val:.4f}")
```

---

## Development & Testing

### Running Tests

Run the complete unit test suite with coverage report:

```bash
# Run pytest with 100% coverage
uv run pytest
```

### Linting & Formatting

Check code quality with `ruff`:

```bash
# Lint check
uv run ruff check .

# Format check
uv run ruff format --check .

# Auto-format
uv run ruff format .
```

---

## Project Structure

```text
CDF_Gen/
├── cdf.py              # Main CDF calculation script, CLI, and Apple Metal engine
├── tests/
│   └── test_cdf.py     # Comprehensive unit tests (47 test cases, 100% coverage)
├── pyproject.toml      # Project configuration, PEP 735 dependency groups, tools
├── .github/workflows/
│   └── ci.yml          # GitHub Actions CI workflow
└── README.md           # Project documentation
```

---

## License

MIT License.
