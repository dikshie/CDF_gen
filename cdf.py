#!/usr/bin/env python3
"""Generate cumulative distribution function (CDF) from numeric data in a file.

Usage: python cdf.py <filename>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def get_available_device() -> str:
    """Check whether Apple Metal GPU acceleration is available."""
    try:
        import mlx.core as mx

        if mx.default_device() == mx.Device(mx.DeviceType.gpu):
            return "metal"
    except ImportError:
        pass

    try:
        import torch

        if torch.backends.mps.is_available():
            return "metal"
    except ImportError:
        pass

    return "cpu"


def get_metal_device_info() -> str | None:
    """Return descriptive name of Apple Metal GPU device if available."""
    try:
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device:
            return str(device.name())
    except Exception:
        pass

    try:
        import mlx.core as mx

        if mx.default_device() == mx.Device(mx.DeviceType.gpu):
            return "Apple Silicon Metal GPU (MLX)"
    except Exception:
        pass

    try:
        import torch

        if torch.backends.mps.is_available():
            return "Apple Metal Performance Shaders (MPS)"
    except Exception:
        pass

    return None


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CDF from numeric data file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cdf.py data.txt
  python cdf.py measurements.csv --delimiter ','
  python cdf.py large_data.txt --device metal
        """,
    )
    parser.add_argument("filename", help="Input file containing numeric data")
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Column delimiter if file has multiple columns",
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Column index to use (0-based, default: 0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["auto", "metal", "cpu"],
        default="auto",
        help="Compute device: auto (Apple Metal GPU if available), metal, or cpu",
    )

    return parser.parse_args()


def read_numeric_data(filename, delimiter=None, column=0):
    """Read numeric data from file using numpy for faster processing.

    Args:
        filename: Path to input file
        delimiter: Column delimiter for multi-column files
        column: Column index to extract (0-based)

    Returns:
        numpy array of float values

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data cannot be converted to float or file cannot be decoded
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        try:
            data = np.loadtxt(
                file_path,
                delimiter=delimiter,
                usecols=column,
                comments="#",
                encoding="utf-8",
            )
            # Ensure we have a 1D array even for single values
            data = np.atleast_1d(data)
        except (ValueError, IndexError) as e:
            print(f"Error using numpy.loadtxt: {e}", file=sys.stderr)
            print("Falling back to manual parsing...", file=sys.stderr)
            data = _manual_parse(file_path, delimiter, column)
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Error: Unable to decode file {filename}. Try a different encoding: {e}"
        ) from e

    if len(data) == 0:
        raise ValueError("No valid numeric data found in file")

    # Remove any NaN or infinite values
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("No valid finite numeric data found in file")

    return data


def _manual_parse(file_path, delimiter=None, column=0):
    """Fallback manual parsing when numpy.loadtxt fails."""
    data = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                values = line.split(delimiter)
                num_cols = len(values)
                if column >= num_cols or column < -num_cols:
                    print(
                        f"Warning: Line {line_num} has insufficient columns, skipping",
                        file=sys.stderr,
                    )
                    continue
                value = float(values[column].strip())

                if np.isfinite(value):
                    data.append(value)

            except ValueError:
                print(
                    f"Warning: Line {line_num} contains invalid data "
                    f"'{line}', skipping",
                    file=sys.stderr,
                )
                continue

    return np.array(data, dtype=float)


def _generate_cdf_mlx(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate CDF using Apple's native MLX framework on Metal GPU."""
    import mlx.core as mx

    # Zero-copy / transfer to MLX Metal GPU array
    x = mx.array(data)
    sorted_data = mx.sort(x)
    n = len(sorted_data)
    probabilities = mx.arange(1, n + 1, 1, dtype=mx.float32) / n
    mx.eval(sorted_data, probabilities)
    return np.array(probabilities, dtype=float), np.array(sorted_data, dtype=float)


def _generate_cdf_mps(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate CDF using PyTorch with Metal Performance Shaders (MPS)."""
    import torch

    device = torch.device("mps")
    x = torch.as_tensor(data, device=device, dtype=torch.float32)
    sorted_data = torch.sort(x).values
    n = len(sorted_data)
    probabilities = torch.arange(1, n + 1, 1, device=device, dtype=torch.float32) / n
    return (
        probabilities.cpu().numpy().astype(float),
        sorted_data.cpu().numpy().astype(float),
    )


def generate_cdf(data: Any, device: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Generate cumulative distribution function from data.

    Supports Apple Metal GPU acceleration via Apple MLX or PyTorch MPS,
    with seamless fallback to NumPy CPU.

    Args:
        data: Numeric sequence or numpy array of values.
        device: Compute device ('auto', 'metal', or 'cpu'). Default is 'auto'.

    Returns:
        Tuple of (probabilities, sorted_values) as numpy arrays.

    Raises:
        ValueError: If device is invalid.
        RuntimeError: If device='metal' is requested but no Metal backend is available.
    """
    valid_devices = {"auto", "metal", "cpu"}
    if device not in valid_devices:
        raise ValueError(f"Invalid device '{device}'. Expected one of {valid_devices}")

    # Flatten input to 1D array
    arr = np.asarray(data, dtype=float).ravel()
    if len(arr) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    use_metal = False
    if device == "metal":
        available = get_available_device()
        if available != "metal":
            raise RuntimeError(
                "Apple Metal GPU acceleration requested, but neither Apple MLX nor "
                "PyTorch MPS is available."
            )
        use_metal = True
    elif device == "auto":
        use_metal = get_available_device() == "metal"

    if use_metal:
        # Try Apple MLX first
        try:
            return _generate_cdf_mlx(arr)
        except Exception as e:
            # Try PyTorch MPS second
            try:
                return _generate_cdf_mps(arr)
            except Exception:
                if device == "metal":
                    msg = f"Failed to execute on Apple Metal GPU: {e}"
                    raise RuntimeError(msg) from e

    # Fallback / CPU computation using NumPy
    sorted_data = np.sort(arr)
    n = len(sorted_data)
    probabilities = np.arange(1, n + 1, dtype=float) / n
    return probabilities, sorted_data


def write_output(probabilities, values, output_file=None):
    """Write CDF data to file or stdout."""
    if len(probabilities) == 0:
        return

    if output_file:
        try:
            # Write directly to file
            np.savetxt(
                output_file,
                np.column_stack([probabilities, values]),
                fmt="%.4f",
                delimiter="\t",
                encoding="utf-8",
            )
            print(f"CDF written to {output_file}", file=sys.stderr)
        except OSError as e:
            print(f"Error writing to file {output_file}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print to stdout with vectorized string formatting
        prob_strings = np.char.mod("%.4f", probabilities)
        value_strings = np.char.mod("%.4f", values)
        output_lines = np.char.add(np.char.add(prob_strings, "\t"), value_strings)
        print("\n".join(output_lines))


def main():
    """Main function."""
    try:
        args = parse_arguments()

        # Read and process data
        data = read_numeric_data(args.filename, args.delimiter, args.column)

        device_info = get_metal_device_info() or "CPU"
        msg = (
            f"Loaded {len(data)} data points from {args.filename} "
            f"(Device: {device_info})"
        )
        print(msg, file=sys.stderr)

        # Generate CDF
        probabilities, sorted_values = generate_cdf(data, device=args.device)

        # Output results
        write_output(probabilities, sorted_values, args.output)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
