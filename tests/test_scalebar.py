import pytest
import numpy as np
from topsy.scalebar import BarLengthRecommender


def test_very_small_scales_parsecs():
    """Test scalebar recommendations for very small scales (sub-parsec)."""
    # 10^-3 pc = 1e-6 kpc
    window_width_kpc = 1e-6
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    # Should recommend something reasonable and small
    assert length <= window_width_kpc / 2
    assert "au" in label  # At this scale, should use AU units


def test_small_parsec_scales():
    """Test scalebar recommendations for parsec-scale windows."""
    # 1 pc = 1e-3 kpc
    window_width_kpc = 1e-3
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "pc" in label

    # 10 pc = 0.01 kpc
    window_width_kpc = 0.01
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "pc" in label


def test_kiloparsec_scales():
    """Test scalebar recommendations for kiloparsec-scale windows."""
    # 1 kpc
    window_width_kpc = 1.0
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "pc" in label or "kpc" in label

    # 100 kpc
    window_width_kpc = 100.0
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "kpc" in label


def test_megaparsec_scales():
    """Test scalebar recommendations for megaparsec-scale windows."""
    # 1 Mpc = 1000 kpc
    window_width_kpc = 1100.0
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "kpc" in label or "Mpc" in label

    # 100 Mpc
    window_width_kpc = 110000.0
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "Mpc" in label

    # 2000 Mpc
    window_width_kpc = 2100000.0
    recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
    length = recommender.physical_scalebar_length_kpc
    label = recommender.label

    assert length <= window_width_kpc / 2
    assert "Mpc" in label


def test_quantization_logic():
    """Test that recommended lengths follow 1, 2, 5 × 10^n pattern in their appropriate units."""
    test_windows = [1e-6, 1e-3, 0.01, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    for window_width_kpc in test_windows:
        recommender = BarLengthRecommender(initial_window_width_kpc=window_width_kpc)
        length = recommender._physical_scalebar_length_in_chosen_unit

        power_of_ten = np.floor(np.log10(length))
        mantissa = length / (10 ** power_of_ten)
        assert any(abs(mantissa - target) < 1e-10 for target in [1.0, 2.0, 5.0])

def test_update_window_width():
    """Test that updating the window width recalculates the recommendation correctly."""
    recommender = BarLengthRecommender(initial_window_width_kpc=1.0)
    initial_length = recommender.physical_scalebar_length_kpc
    initial_label = recommender.label

    # Update to a larger window
    recommender.update_window_width(100.0)
    new_length = recommender.physical_scalebar_length_kpc
    new_label = recommender.label

    assert new_length != initial_length
    assert new_label != initial_label
    assert new_length <= 100.0 / 2


def test_label_formatting():
    """Test that labels are formatted correctly for different ranges."""

    # Test very small values that should use scientific notation
    recommender = BarLengthRecommender(initial_window_width_kpc=1e-6)  # 0.001 pc window
    label = recommender.label
    if "pc" in label and recommender.physical_scalebar_length_kpc * 1000 < 0.01:  # If in parsecs and very small
        assert "$" in label and "\\times 10^{" in label

    # Test normal parsec values
    recommender = BarLengthRecommender(initial_window_width_kpc=0.01)  # 10 pc window
    label = recommender.label
    if "pc" in label:
        # Should be normal formatting, not scientific
        value_in_pc = recommender.physical_scalebar_length_kpc * 1000
        if 0.01 <= value_in_pc <= 1000:
            assert "$" not in label

    # Test kpc values
    recommender = BarLengthRecommender(initial_window_width_kpc=10.0)  # 10 kpc window
    label = recommender.label
    if "kpc" in label:
        assert "$" not in label  # Normal formatting

    # Test Mpc values
    recommender = BarLengthRecommender(initial_window_width_kpc=10000.0)  # 10 Mpc window
    label = recommender.label
    if "Mpc" in label:
        assert "$" not in label  # Normal formatting


def test_format_scientific_latex():
    """Test the LaTeX scientific notation formatter."""

    # Test normal range values (no scientific notation)
    result = BarLengthRecommender._format_scientific_latex(0.1, "pc")
    assert result == "0.1 pc"

    result = BarLengthRecommender._format_scientific_latex(1.0, "pc")
    assert result == "1 pc"

    result = BarLengthRecommender._format_scientific_latex(10.5, "kpc")
    assert result == "10.5 kpc"

    # Test very small values (scientific notation)
    result = BarLengthRecommender._format_scientific_latex(0.005, "pc")
    assert result == "$5 \\times 10^{-3}$ pc"

    result = BarLengthRecommender._format_scientific_latex(0.002, "pc")
    assert result == "$2 \\times 10^{-3}$ pc"

    # Test very large values (scientific notation)
    result = BarLengthRecommender._format_scientific_latex(2000, "Mpc")
    assert result == "$2 \\times 10^{3}$ Mpc"

    # Test zero
    result = BarLengthRecommender._format_scientific_latex(0, "pc")
    assert result == "0 pc"


