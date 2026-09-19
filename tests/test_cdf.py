"""Unit tests for cdf.py."""

import sys

import numpy as np
import pytest

from cdf import (
    _manual_parse,
    generate_cdf,
    main,
    parse_arguments,
    read_numeric_data,
    write_output,
)


class TestGenerateCdf:
    """Tests for generate_cdf function."""

    def test_empty_array(self):
        """Test with empty array returns empty arrays."""
        probs, values = generate_cdf(np.array([]))
        assert len(probs) == 0
        assert len(values) == 0
        assert isinstance(probs, np.ndarray)
        assert isinstance(values, np.ndarray)

    def test_single_value(self):
        """Test with single value."""
        data = np.array([42.0])
        probs, values = generate_cdf(data)
        np.testing.assert_array_equal(values, np.array([42.0]))
        np.testing.assert_allclose(probs, np.array([1.0]))

    def test_sorted_input(self):
        """Test with already sorted input."""
        data = np.array([1.0, 2.0, 4.0, 8.0])
        probs, values = generate_cdf(data)
        np.testing.assert_array_equal(values, np.array([1.0, 2.0, 4.0, 8.0]))
        np.testing.assert_allclose(probs, np.array([0.25, 0.5, 0.75, 1.0]))

    def test_unsorted_input(self):
        """Test with unsorted input."""
        data = np.array([30.0, 10.0, 20.0])
        probs, values = generate_cdf(data)
        np.testing.assert_array_equal(values, np.array([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(probs, np.array([1 / 3, 2 / 3, 1.0]))

    def test_duplicates(self):
        """Test with duplicate values."""
        data = np.array([5.0, 1.0, 5.0, 2.0])
        probs, values = generate_cdf(data)
        np.testing.assert_array_equal(values, np.array([1.0, 2.0, 5.0, 5.0]))
        np.testing.assert_allclose(probs, np.array([0.25, 0.5, 0.75, 1.0]))

    def test_negative_and_floats(self):
        """Test with negative numbers and floats."""
        data = np.array([-1.5, 2.5, 0.0, -3.0])
        probs, values = generate_cdf(data)
        np.testing.assert_array_equal(values, np.array([-3.0, -1.5, 0.0, 2.5]))
        np.testing.assert_allclose(probs, np.array([0.25, 0.5, 0.75, 1.0]))


class TestReadNumericData:
    """Tests for read_numeric_data function."""

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent = tmp_path / "missing.txt"
        with pytest.raises(FileNotFoundError, match="File not found"):
            read_numeric_data(non_existent)

    def test_single_column_file(self, tmp_path):
        """Test reading single column file with comments and empty lines."""
        file = tmp_path / "single_col.txt"
        file.write_text("# header comment\n10.5\n\n20.0\n# middle comment\n30.5\n")
        data = read_numeric_data(file)
        np.testing.assert_allclose(data, np.array([10.5, 20.0, 30.5]))

    def test_single_value_file(self, tmp_path):
        """Test reading a file with only one numeric value."""
        file = tmp_path / "single_val.txt"
        file.write_text("42.0\n")
        data = read_numeric_data(file)
        assert data.shape == (1,)
        assert data[0] == 42.0

    def test_multi_column_file_with_delimiter(self, tmp_path):
        """Test reading multi-column file with custom delimiter."""
        file = tmp_path / "multi_col.csv"
        file.write_text("a,1.5,100\nb,2.5,200\nc,3.5,300\n")
        # Column 1
        data1 = read_numeric_data(file, delimiter=",", column=1)
        np.testing.assert_allclose(data1, np.array([1.5, 2.5, 3.5]))
        # Column 2
        data2 = read_numeric_data(file, delimiter=",", column=2)
        np.testing.assert_allclose(data2, np.array([100.0, 200.0, 300.0]))

    def test_filters_nan_and_inf(self, tmp_path):
        """Test that NaN and Inf values are filtered out."""
        file = tmp_path / "nan_inf.txt"
        file.write_text("1.0\nNaN\n2.0\nInf\n-Inf\n3.0\n")
        data = read_numeric_data(file)
        np.testing.assert_allclose(data, np.array([1.0, 2.0, 3.0]))

    def test_empty_file_raises_value_error(self, tmp_path):
        """Test that an empty file raises ValueError."""
        file = tmp_path / "empty.txt"
        file.write_text("")
        with pytest.raises(ValueError, match="No valid numeric data found"):
            read_numeric_data(file)

    def test_comments_only_raises_value_error(self, tmp_path):
        """Test that a file with only comments raises ValueError."""
        file = tmp_path / "comments_only.txt"
        file.write_text("# comment 1\n# comment 2\n")
        with pytest.raises(ValueError, match="No valid numeric data found"):
            read_numeric_data(file)

    def test_only_nan_and_inf_raises_value_error(self, tmp_path):
        """Test that a file with only NaN and Inf raises ValueError."""
        file = tmp_path / "only_nan.txt"
        file.write_text("NaN\nInf\n-Inf\n")
        with pytest.raises(ValueError, match="No valid finite numeric data found"):
            read_numeric_data(file)

    def test_fallback_to_manual_parse_ragged_file(self, tmp_path, capsys):
        """Test fallback to manual parsing when loadtxt fails on ragged lines."""
        file = tmp_path / "ragged.csv"
        # Second line has non-numeric text in col 1; loadtxt will fail
        # and manual parse will recover lines 1 and 3.
        file.write_text("10,1.5\n20,bad_value\n30,3.5\n")
        data = read_numeric_data(file, delimiter=",", column=1)
        np.testing.assert_allclose(data, np.array([1.5, 3.5]))
        captured = capsys.readouterr()
        assert "Falling back to manual parsing" in captured.out


class TestManualParse:
    """Tests for _manual_parse function."""

    def test_manual_parse_skips_invalid_and_comments(self, tmp_path, capsys):
        """Test manual parsing skips comments, blank lines, and invalid values."""
        file = tmp_path / "manual.txt"
        file.write_text("# comment\n1.0\n\ninvalid\n2.0\n")
        data = _manual_parse(file)
        np.testing.assert_allclose(data, np.array([1.0, 2.0]))
        captured = capsys.readouterr()
        expected = "Warning: Line 4 contains invalid data 'invalid', skipping"
        assert expected in captured.out

    def test_manual_parse_insufficient_columns(self, tmp_path, capsys):
        """Test manual parsing handles lines with insufficient columns."""
        file = tmp_path / "short_cols.csv"
        file.write_text("1.0,2.0\n1.0\n3.0,4.0\n")
        data = _manual_parse(file, delimiter=",", column=1)
        np.testing.assert_allclose(data, np.array([2.0, 4.0]))
        captured = capsys.readouterr()
        assert "Warning: Line 2 has insufficient columns, skipping" in captured.out


class TestWriteOutput:
    """Tests for write_output function."""

    def test_write_output_empty(self, capsys):
        """Test write_output does nothing if probabilities array is empty."""
        write_output(np.array([]), np.array([]))
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_write_output_stdout(self, capsys):
        """Test writing output to stdout."""
        probs = np.array([0.25, 0.5, 1.0])
        values = np.array([10.0, 20.0, 30.0])
        write_output(probs, values)
        captured = capsys.readouterr()
        expected = "0.2500\t10.0000\n0.5000\t20.0000\n1.0000\t30.0000\n"
        assert captured.out == expected

    def test_write_output_file(self, tmp_path, capsys):
        """Test writing output to a file."""
        probs = np.array([0.5, 1.0])
        values = np.array([1.23456, 7.89101])
        output_file = tmp_path / "cdf_out.tsv"
        write_output(probs, values, output_file=str(output_file))

        captured = capsys.readouterr()
        assert f"CDF written to {output_file}" in captured.out

        content = output_file.read_text().strip().split("\n")
        assert len(content) == 2
        assert content[0] == "0.5000\t1.2346"
        assert content[1] == "1.0000\t7.8910"

    def test_write_output_file_error(self, capsys):
        """Test write_output exits on file writing error."""
        probs = np.array([1.0])
        values = np.array([1.0])
        invalid_path = "/nonexistent_dir/impossible/out.tsv"
        with pytest.raises(SystemExit) as exc_info:
            write_output(probs, values, output_file=invalid_path)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error writing to file" in captured.out


class TestCliAndMain:
    """Tests for parse_arguments and main function."""

    def test_parse_arguments_defaults(self, monkeypatch):
        """Test CLI arguments parsing with default values."""
        monkeypatch.setattr(sys, "argv", ["cdf.py", "data.txt"])
        args = parse_arguments()
        assert args.filename == "data.txt"
        assert args.delimiter is None
        assert args.column == 0
        assert args.output is None

    def test_parse_arguments_custom(self, monkeypatch):
        """Test CLI arguments parsing with custom options."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cdf.py",
                "data.csv",
                "--delimiter",
                ",",
                "--column",
                "2",
                "-o",
                "res.txt",
            ],
        )
        args = parse_arguments()
        assert args.filename == "data.csv"
        assert args.delimiter == "=" or args.delimiter == ","
        assert args.column == 2
        assert args.output == "res.txt"

    def test_main_success(self, tmp_path, monkeypatch, capsys):
        """Test full main function execution successfully."""
        data_file = tmp_path / "input.txt"
        data_file.write_text("3\n1\n2\n")
        out_file = tmp_path / "result.tsv"

        monkeypatch.setattr(
            sys,
            "argv",
            ["cdf.py", str(data_file), "-o", str(out_file)],
        )
        main()

        captured = capsys.readouterr()
        assert f"Loaded 3 data points from {data_file}" in captured.err
        assert out_file.exists()

    def test_main_file_not_found(self, monkeypatch, capsys):
        """Test main exits with error code 1 when input file is missing."""
        monkeypatch.setattr(sys, "argv", ["cdf.py", "missing_file_xyz.txt"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: File not found:" in captured.err

    def test_main_keyboard_interrupt(self, monkeypatch, capsys):
        """Test main handles KeyboardInterrupt gracefully."""

        def mock_read(*args, **kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr("cdf.read_numeric_data", mock_read)
        monkeypatch.setattr(sys, "argv", ["cdf.py", "some_file.txt"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Operation cancelled by user" in captured.err
