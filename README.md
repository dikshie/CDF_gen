# CDF_Gen

A lightweight, high-performance command-line utility and Python module to compute the empirical **Cumulative Distribution Function (CDF)** from numeric datasets.

Built with [NumPy](https://numpy.org/) for fast, vectorized calculations and optimized file I/O.

---

## Features

- **High Performance**: Uses NumPy vectorization for sorting and probability calculations, easily scaling to large datasets.
- **Flexible Input Formats**: Supports single-column files, whitespace-delimited files, and multi-column formats (e.g., CSV, TSV) with selectable column indices.
- **Robust Data Parsing**:
  - Automatically ignores comment lines starting with `#` and blank lines.
  - Filters out `NaN` and infinite (`Inf`, `-Inf`) values.
  - Automatically falls back to manual line-by-line parsing if `numpy.loadtxt` encounters malformed or ragged rows.
- **Tab-Separated Output**: Outputs formatted `probability\tvalue` pairs to `stdout` or directly to an output file.
- **Tested & Linted**: Comprehensive test suite with [pytest](https://docs.pytest.org/) and formatted/linted with [Ruff](https://astral.sh/ruff).

---

## Mathematical Background

The empirical cumulative distribution function (eCDF) $F_n(x)$ for a sample of $n$ observations $X_1, X_2, \dots, X_n$ is defined as:

$$F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{X_i \le x}$$

Given sorted observations $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$, the probability associated with each observation $x_{(i)}$ is:

$$P(X \le x_{(i)}) = \frac{i}{n}, \quad \text{for } i = 1, 2, \dots, n$$

The tool outputs these $(P(X \le x_{(i)}), x_{(i)})$ pairs formatted to 4 decimal places.

---

## Requirements & Installation

### Requirements

- Python >= 3.9
- [NumPy](https://numpy.org/)

### Setup

Clone the repository and install dependencies using `pip` or [`uv`](https://github.com/astral-sh/uv):

```bash
# Using pip
pip install numpy

# Or with dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

Or run directly with `uv`:

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
| `--column` | | 0-based column index to extract | `0` |
| `--output` | `-o` | Output file path | `stdout` |
| `--help` | `-h` | Show usage help | |

### Examples

#### 1. Basic Single-Column Input

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

#### 2. Multi-Column CSV File

Extract column 1 (second column) from a comma-separated file:
```bash
./cdf.py measurements.csv --delimiter ',' --column 1
```

#### 3. Save Output to a File

```bash
./cdf.py data.txt -o cdf_output.tsv
```

### Python API

You can also import and use `cdf` functions directly in your Python code:

```python
import numpy as np
from cdf import generate_cdf, read_numeric_data, write_output

# Read data from file
data = read_numeric_data("data.txt")

# Or compute CDF on an existing numpy array
probs, sorted_values = generate_cdf(np.array([12.5, 3.2, 8.7, 1.0]))

# Output or inspect
for p, val in zip(probs, sorted_values):
    print(f"{p:.4f}\t{val:.4f}")
```

---

## Development & Testing

### Running Tests

Run the unit test suite using `pytest`:

```bash
# Using uv
uv run pytest

# Or directly if pytest is installed in your environment
pytest
```

### Linting & Formatting

Check code quality with `ruff`:

```bash
# Lint check
uvx ruff check .

# Format check
uvx ruff format --check .
```

---

## Project Structure

```text
CDF_Gen/
├── cdf.py              # Main CDF calculation script and CLI
├── tests/
│   └── test_cdf.py     # Unit tests (26 test cases)
├── pyproject.toml      # Project configuration, dependencies, and tool settings
└── README.md           # Project documentation
```

---

## License

MIT License.
