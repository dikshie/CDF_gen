#!/usr/bin/env python3
"""
Generate cumulative distribution function (CDF) from numeric data in a file.
Usage: python script.py <filename>
"""
import sys
import argparse
from pathlib import Path
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CDF from numeric data file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py data.txt
  python script.py measurements.csv --delimiter ','
        """
    )
    parser.add_argument('filename', help='Input file containing numeric data')
    parser.add_argument('--delimiter', default=None, 
                       help='Column delimiter if file has multiple columns')
    parser.add_argument('--column', type=int, default=0,
                       help='Column index to use (0-based, default: 0)')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    return parser.parse_args()


def read_numeric_data(filename, delimiter=None, column=0):
    """
    Read numeric data from file using numpy for faster processing.
    
    Args:
        filename: Path to input file
        delimiter: Column delimiter for multi-column files
        column: Column index to extract (0-based)
        
    Returns:
        numpy array of float values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data cannot be converted to float
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        if delimiter:
            # Use numpy's loadtxt for delimited files - much faster
            try:
                data = np.loadtxt(file_path, delimiter=delimiter, usecols=column, 
                                 comments='#', encoding='utf-8')
                # Ensure we have a 1D array even for single values
                data = np.atleast_1d(data)
            except (ValueError, IndexError) as e:
                print(f"Error using numpy.loadtxt: {e}")
                print("Falling back to manual parsing...")
                data = _manual_parse(file_path, delimiter, column)
        else:
            # Single column file
            try:
                data = np.loadtxt(file_path, comments='#', encoding='utf-8')
                data = np.atleast_1d(data)
            except ValueError as e:
                print(f"Error using numpy.loadtxt: {e}")
                print("Falling back to manual parsing...")
                data = _manual_parse(file_path, delimiter, column)
                
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file {filename}. Try a different encoding.")
        sys.exit(1)
    
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
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                if delimiter:
                    values = line.split(delimiter)
                    if column >= len(values):
                        print(f"Warning: Line {line_num} has insufficient columns, skipping")
                        continue
                    value = float(values[column].strip())
                else:
                    value = float(line)
                
                if np.isfinite(value):
                    data.append(value)
                    
            except ValueError:
                print(f"Warning: Line {line_num} contains invalid data '{line}', skipping")
                continue
    
    return np.array(data)


def generate_cdf(data):
    """
    Generate cumulative distribution function from data using numpy.
    
    Args:
        data: numpy array of numeric values
        
    Returns:
        Tuple of (probabilities, sorted_values) as numpy arrays
    """
    if len(data) == 0:
        return np.array([]), np.array([])
    
    # Sort data using numpy - much faster for large datasets
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Generate probabilities using numpy - vectorized operation
    probabilities = np.arange(1, n + 1) / n  # Empirical CDF
    
    return probabilities, sorted_data


def write_output(probabilities, values, output_file=None):
    """Write CDF data to file or stdout using numpy for formatting."""
    if len(probabilities) == 0:
        return
    
    # Use numpy's vectorized string formatting - much faster
    prob_strings = np.char.mod('%.4f', probabilities)
    value_strings = np.char.mod('%.4f', values)
    
    # Combine with tabs using numpy operations
    output_lines = np.char.add(np.char.add(prob_strings, '\t'), value_strings)
    
    if output_file:
        try:
            # Write all lines at once
            np.savetxt(output_file, np.column_stack([probabilities, values]), 
                      fmt='%.4f', delimiter='\t', encoding='utf-8')
            print(f"CDF written to {output_file}")
        except IOError as e:
            print(f"Error writing to file {output_file}: {e}")
            sys.exit(1)
    else:
        # Print to stdout
        print('\n'.join(output_lines))


def main():
    """Main function."""
    try:
        args = parse_arguments()
        
        # Read and process data
        data = read_numeric_data(args.filename, args.delimiter, args.column)
        print(f"Loaded {len(data)} data points from {args.filename}", file=sys.stderr)
        
        # Generate CDF
        probabilities, sorted_values = generate_cdf(data)
        
        # Output results
        write_output(probabilities, sorted_values, args.output)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
