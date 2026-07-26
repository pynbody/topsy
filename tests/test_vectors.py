"""Tests for the vector-field overlay: its magnitude->pixels scaling, the key
overlay, and the behaviour when the visualiser's vector_name is changed.

The scaling assertions are written as invariants/ratios rather than absolute
pixel values wherever possible. The integration tests drive the (software)
render pipeline, which is exercised in CI via tests/run_tests_in_docker.sh.
"""

import numpy as np
import pytest

import topsy
from topsy.canvas import offscreen
from topsy.drawreason import DrawReason
from topsy.vectors import VectorOverlay
from topsy.util import quantize_to_1_2_5


# --------------------------------------------------------------------------
# Pure-logic tests (no GPU): magnitude -> pixels scaling
# --------------------------------------------------------------------------

def _is_1_2_5_number(x):
    """True if x is (close to) m x 10^N with m in {1, 2, 5}."""
    power = np.floor(np.log10(x))
    mantissa = x / 10.0 ** power
    return bool(np.isclose(mantissa, [1.0, 2.0, 5.0], atol=1e-9).any())


def _bare_overlay():
    """A VectorOverlay instance with just the scaling state, bypassing GPU setup."""
    overlay = object.__new__(VectorOverlay)
    overlay.reference_value = 1.0
    overlay.vector_units_per_dot = 1.0
    return overlay


def test_quantize_to_1_2_5_modes():
    # nearest snaps by ratio (log space): 3.1 is closer to 2, 3.2 closer to 5
    assert quantize_to_1_2_5(3.1, mode="nearest") == 2.0
    assert quantize_to_1_2_5(3.2, mode="nearest") == 5.0
    assert quantize_to_1_2_5(8.0, mode="nearest") == 10.0
    # floor never exceeds the value
    assert quantize_to_1_2_5(9.9, mode="floor") == 5.0
    assert quantize_to_1_2_5(1.0, mode="floor") == 1.0
    # works across decades
    for value in [3e-4, 0.07, 42.0, 7.3e5]:
        assert _is_1_2_5_number(quantize_to_1_2_5(value, mode="nearest"))


def test_quantize_to_1_2_5_rejects_nonpositive():
    with pytest.raises(ValueError):
        quantize_to_1_2_5(0.0)
    with pytest.raises(ValueError):
        quantize_to_1_2_5(-5.0)


def test_reference_value_is_a_nice_number():
    overlay = _bare_overlay()
    for max_magnitude in [0.003, 1.7, 137.0, 4.2e4]:
        overlay._update_scaling(max_magnitude, target_max_length_dots=30.0)
        assert _is_1_2_5_number(overlay.reference_value)


def test_reference_length_matches_target_regardless_of_magnitude():
    """The reference arrow is drawn at exactly the target length, whatever the
    field magnitude or units -- this is the invariant that keeps the on-screen
    arrow (and hence the key) a sensible size."""
    overlay = _bare_overlay()
    target = 42.0
    lengths = []
    for max_magnitude in [1e-3, 1.0, 250.0, 9.9e6]:
        overlay._update_scaling(max_magnitude, target_max_length_dots=target)
        lengths.append(overlay.reference_length_dots)
    assert lengths == pytest.approx([target] * len(lengths))


def test_scaling_reference_value_tracks_magnitude():
    """Scaling the field by 1000x scales the reference value by 1000x (both are
    the same nice number, just a different power of ten)."""
    overlay = _bare_overlay()
    overlay._update_scaling(137.0, target_max_length_dots=30.0)
    small = overlay.reference_value
    overlay._update_scaling(137.0e3, target_max_length_dots=30.0)
    big = overlay.reference_value
    assert big == pytest.approx(small * 1000.0)


def test_degenerate_field_scaling_is_finite():
    overlay = _bare_overlay()
    for bad in [0.0, -1.0, np.nan, np.inf]:
        overlay._update_scaling(bad, target_max_length_dots=30.0)
        assert np.isfinite(overlay.reference_value)
        assert np.isfinite(overlay.reference_length_dots)
        assert overlay.reference_length_dots > 0.0


def test_render_contents_without_field_is_transparent():
    """With no field set, render_contents returns a single transparent pixel and
    must not touch the (absent) visualizer."""
    overlay = object.__new__(VectorOverlay)
    overlay._vector_field = None
    image = VectorOverlay.render_contents(overlay)
    assert image.shape == (1, 1, 4)
    assert image.dtype == np.float32
    assert np.all(image == 0.0)


# --------------------------------------------------------------------------
# Integration tests (drive the render pipeline)
# --------------------------------------------------------------------------

def _make_visualizer():
    return topsy.test(2000, render_resolution=100,
                      canvas_class=offscreen.VisualizerCanvas)


def _recompute_vectors(vis):
    """Render the SPH and (re)compute the overlaid vector field, as the app does."""
    vis.draw(DrawReason.VECTOR_UPDATE)


def test_vector_name_change_rescales_and_relabels():
    """Changing vector_name must refill the GPU buffer, rescale the reference
    value with the new magnitude, and relabel the key with the new units.

    'vel-mps' is exactly 1000x 'vel' with different units, so the reference value
    should scale by 1000 while the on-screen arrow length is unchanged.
    """
    vis = _make_visualizer()

    vis.vector_name = "vel"
    _recompute_vectors(vis)
    ref_kms = vis._vectors.reference_value
    length_kms = vis._vectors.reference_length_dots
    label_kms = vis._vector_key.label

    vis.vector_name = "vel-mps"
    _recompute_vectors(vis)
    ref_mps = vis._vectors.reference_value
    length_mps = vis._vectors.reference_length_dots
    label_mps = vis._vector_key.label

    # the buffer actually changed: reference value tracks the 1000x magnitude
    assert ref_mps == pytest.approx(ref_kms * 1000.0)

    # on-screen arrow length is magnitude-independent
    assert length_mps == pytest.approx(length_kms)

    # the key label carries the units of the currently-selected vector quantity
    assert vis.data_loader.get_quantity_units_string("vel") in label_kms
    assert vis.data_loader.get_quantity_units_string("vel-mps") in label_mps
    assert label_kms != label_mps

def test_cannot_set_vector_to_nonvector_quantity():
    vis = _make_visualizer()
    with pytest.raises(ValueError) as e:
        vis.vector_name = "test-quantity"  # ... a scalar
    assert "must have shape (N, 3)" in str(e.value)
    assert vis.vector_name == "vel"


def test_vector_key_length_matches_field_after_draw():
    """After a draw the key's arrow length is driven from the field's scaling."""
    vis = _make_visualizer()
    vis.vector_name = "vel"
    _recompute_vectors(vis)
    assert vis._vector_key.key_length_dots == pytest.approx(
        vis._vectors.reference_length_dots)


def test_invalid_vector_name_raises():
    vis = _make_visualizer()
    with pytest.raises(KeyError):
        vis.vector_name = "not-a-real-quantity"
