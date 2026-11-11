"""Unit tests for cdf.py."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cdf import (
    _generate_cdf_mlx,
    _generate_cdf_mps,
    _manual_parse,
    generate_cdf,
    get_available_device,
    get_metal_device_info,
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

    def test_multidimensional_flattening(self):
        """Test that 2D input array is flattened before CDF calculation."""
        data_2d = np.array([[30.0, 10.0], [40.0, 20.0]])
        probs, values = generate_cdf(data_2d)
        np.testing.assert_array_equal(values, np.array([10.0, 20.0, 30.0, 40.0]))
        np.testing.assert_allclose(probs, np.array([0.25, 0.5, 0.75, 1.0]))

    def test_python_list_input(self):
        """Test that standard Python list can be passed."""
        probs, values = generate_cdf([3.0, 1.0, 2.0])
        np.testing.assert_array_equal(values, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(probs, np.array([1 / 3, 2 / 3, 1.0]))

    def test_device_cpu(self):
        """Test explicitly requesting CPU calculation."""
        data = np.array([10.0, 5.0, 15.0])
        probs, values = generate_cdf(data, device="cpu")
        np.testing.assert_array_equal(values, np.array([5.0, 10.0, 15.0]))
        np.testing.assert_allclose(probs, np.array([1 / 3, 2 / 3, 1.0]))

    def test_device_invalid(self):
        """Test invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device 'cuda'"):
            generate_cdf([1.0, 2.0], device="cuda")

    def test_device_metal_available(self):
        """Test requesting Metal device when available."""
        if get_available_device() == "metal":
            data = np.array([50.0, 20.0, 40.0, 10.0, 30.0])
            probs, values = generate_cdf(data, device="metal")
            np.testing.assert_array_equal(
                values, np.array([10.0, 20.0, 30.0, 40.0, 50.0])
            )
            np.testing.assert_allclose(probs, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))

    def test_device_metal_unavailable_raises_error(self, monkeypatch):
        """Test requesting Metal device when unavailable raises RuntimeError."""
        monkeypatch.setattr("cdf.get_available_device", lambda: "cpu")
        with pytest.raises(RuntimeError, match="Apple Metal GPU acceleration"):
            generate_cdf([1.0, 2.0], device="metal")

    def test_device_metal_fallback_failure(self, monkeypatch):
        """Test metal failure raises RuntimeError when device='metal'."""
        monkeypatch.setattr("cdf.get_available_device", lambda: "metal")

        def failing_mlx(*args, **kwargs):
            raise RuntimeError("MLX internal fault")

        monkeypatch.setattr("cdf._generate_cdf_mlx", failing_mlx)
        monkeypatch.setattr("cdf._generate_cdf_mps", failing_mlx)

        with pytest.raises(RuntimeError, match="Failed to execute on Apple Metal GPU"):
            generate_cdf([1.0, 2.0], device="metal")


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

    def test_multi_column_whitespace_file(self, tmp_path):
        """Test reading multi-column whitespace file extracting specific column."""
        file = tmp_path / "whitespace_multi.txt"
        file.write_text("10 100 1000\n20 200 2000\n30 300 3000\n")
        # Extract column 1 (default delimiter is None / whitespace)
        data1 = read_numeric_data(file, delimiter=None, column=1)
        np.testing.assert_allclose(data1, np.array([100.0, 200.0, 300.0]))
        # Extract column 2
        data2 = read_numeric_data(file, delimiter=None, column=2)
        np.testing.assert_allclose(data2, np.array([1000.0, 2000.0, 3000.0]))

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

    def test_negative_column_indexing(self, tmp_path):
        """Test reading file with negative column index."""
        file = tmp_path / "negative_col.tsv"
        file.write_text("10\t20\t30\n40\t50\t60\n")
        data = read_numeric_data(file, delimiter="\t", column=-1)
        np.testing.assert_allclose(data, np.array([30.0, 60.0]))

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
        file.write_text("10,1.5\n20,bad_value\n30,3.5\n")
        data = read_numeric_data(file, delimiter=",", column=1)
        np.testing.assert_allclose(data, np.array([1.5, 3.5]))
        captured = capsys.readouterr()
        # Diagnostic messages should be printed to stderr, not stdout
        assert "Falling back to manual parsing" in captured.err
        assert captured.out == ""

    def test_fallback_to_manual_parse_ragged_whitespace(self, tmp_path, capsys):
        """Test fallback to manual parsing on ragged whitespace file."""
        file = tmp_path / "ragged_ws.txt"
        file.write_text("10 20\nbad 40\n50 60\n")
        data = read_numeric_data(file, delimiter=None, column=0)
        np.testing.assert_allclose(data, np.array([10.0, 50.0]))
        captured = capsys.readouterr()
        assert "Falling back to manual parsing" in captured.err

    def test_unicode_decode_error_raises_value_error(self, tmp_path):
        """Test non-UTF8 bytes raise ValueError instead of exiting process."""
        binary_file = tmp_path / "binary.bin"
        # Write invalid UTF-8 sequence
        binary_file.write_bytes(b"\x80\x81\x82\xff")
        with pytest.raises(ValueError, match="Unable to decode file"):
            read_numeric_data(binary_file)


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
        assert expected in captured.err

    def test_manual_parse_insufficient_columns(self, tmp_path, capsys):
        """Test manual parsing handles lines with insufficient columns."""
        file = tmp_path / "short_cols.csv"
        file.write_text("1.0,2.0\n1.0\n3.0,4.0\n")
        data = _manual_parse(file, delimiter=",", column=1)
        np.testing.assert_allclose(data, np.array([2.0, 4.0]))
        captured = capsys.readouterr()
        assert "Warning: Line 2 has insufficient columns, skipping" in captured.err

    def test_manual_parse_negative_column_insufficient(self, tmp_path, capsys):
        """Test manual parsing handles negative column out of bounds."""
        file = tmp_path / "neg_cols.csv"
        file.write_text("1.0,2.0\n")
        data = _manual_parse(file, delimiter=",", column=-5)
        assert len(data) == 0
        captured = capsys.readouterr()
        assert "Warning: Line 1 has insufficient columns, skipping" in captured.err


class TestWriteOutput:
    """Tests for write_output function."""

    def test_write_output_empty(self, capsys):
        """Test write_output does nothing if probabilities array is empty."""
        write_output(np.array([]), np.array([]))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

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
        assert f"CDF written to {output_file}" in captured.err

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
        assert "Error writing to file" in captured.err


class TestMetalAcceleration:
    """Tests for Apple Metal framework detection and execution."""

    def test_get_available_device(self):
        """Test get_available_device returns valid device name."""
        device = get_available_device()
        assert device in {"metal", "cpu"}

    def test_get_available_device_torch_mps(self, monkeypatch):
        """Test get_available_device detects MPS when MLX is absent."""
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = True

        with patch.dict(
            "sys.modules",
            {"mlx": None, "mlx.core": None, "torch": fake_torch},
        ):
            assert get_available_device() == "metal"

    def test_get_available_device_none_returns_cpu(self, monkeypatch):
        """Test get_available_device returns cpu when no metal library is present."""
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None, "torch": None}):
            assert get_available_device() == "cpu"

    def test_get_metal_device_info_metal_framework(self):
        """Test get_metal_device_info using mock Metal framework."""
        fake_metal = MagicMock()
        fake_dev = MagicMock()
        fake_dev.name.return_value = "Apple M2 Max"
        fake_metal.MTLCreateSystemDefaultDevice.return_value = fake_dev

        with patch.dict("sys.modules", {"Metal": fake_metal}):
            assert get_metal_device_info() == "Apple M2 Max"

    def test_get_metal_device_info_torch_mps(self):
        """Test get_metal_device_info fallback to Torch MPS."""
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_available.return_value = True

        with patch.dict(
            "sys.modules",
            {"Metal": None, "mlx": None, "mlx.core": None, "torch": fake_torch},
        ):
            assert get_metal_device_info() == "Apple Metal Performance Shaders (MPS)"

    def test_get_metal_device_info_none(self):
        """Test get_metal_device_info returns None when no backend is available."""
        with patch.dict(
            "sys.modules",
            {"Metal": None, "mlx": None, "mlx.core": None, "torch": None},
        ):
            assert get_metal_device_info() is None

    def test_generate_cdf_mlx_direct(self):
        """Test _generate_cdf_mlx direct call if MLX is installed."""
        try:
            import mlx.core as mx  # noqa: F401

            probs, values = _generate_cdf_mlx(np.array([4.0, 2.0, 1.0, 3.0]))
            np.testing.assert_array_equal(values, np.array([1.0, 2.0, 3.0, 4.0]))
            np.testing.assert_allclose(probs, np.array([0.25, 0.5, 0.75, 1.0]))
        except ImportError:
            pytest.skip("MLX not available on this platform")

    def test_generate_cdf_mps_mock(self, monkeypatch):
        """Test _generate_cdf_mps function with torch mock."""
        fake_torch = MagicMock()
        fake_tensor = MagicMock()
        fake_sorted_val = MagicMock()
        fake_sorted_val.__len__.return_value = 2
        fake_sorted_val.cpu.return_value.numpy.return_value.astype.return_value = (
            np.array([10.0, 20.0])
        )

        fake_sort_result = MagicMock()
        fake_sort_result.values = fake_sorted_val

        fake_probs_tensor = MagicMock()
        divided = fake_probs_tensor.__truediv__.return_value
        divided.cpu.return_value.numpy.return_value.astype.return_value = np.array(
            [0.5, 1.0]
        )

        fake_torch.device.return_value = "mps"
        fake_torch.as_tensor.return_value = fake_tensor
        fake_torch.sort.return_value = fake_sort_result
        fake_torch.arange.return_value = fake_probs_tensor

        with patch.dict("sys.modules", {"torch": fake_torch}):
            probs, values = _generate_cdf_mps(np.array([20.0, 10.0]))
            np.testing.assert_array_equal(values, np.array([10.0, 20.0]))
            np.testing.assert_allclose(probs, np.array([0.5, 1.0]))


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
        assert args.device == "auto"

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
                "--device",
                "metal",
            ],
        )
        args = parse_arguments()
        assert args.filename == "data.csv"
        assert args.delimiter == ","
        assert args.column == 2
        assert args.output == "res.txt"
        assert args.device == "metal"

    def test_main_success(self, tmp_path, monkeypatch, capsys):
        """Test full main function execution successfully."""
        data_file = tmp_path / "input.txt"
        data_file.write_text("3\n1\n2\n")
        out_file = tmp_path / "result.tsv"

        monkeypatch.setattr(
            sys,
            "argv",
            ["cdf.py", str(data_file), "-o", str(out_file), "--device", "cpu"],
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

    def test_main_runtime_error(self, monkeypatch, capsys):
        """Test main handles RuntimeError gracefully."""

        def mock_generate(*args, **kwargs):
            raise RuntimeError("Metal GPU failure")

        monkeypatch.setattr("cdf.generate_cdf", mock_generate)
        monkeypatch.setattr(
            sys, "argv", ["cdf.py", "some_file.txt", "--device", "metal"]
        )
        with patch("cdf.read_numeric_data", return_value=np.array([1.0, 2.0])):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: Metal GPU failure" in captured.err

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
