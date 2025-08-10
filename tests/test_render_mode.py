import numpy as np
import pytest
import topsy 

from topsy.canvas import offscreen

def test_render_mode_switching():
    """Test that render mode can be switched on-the-fly without errors"""
    vis = topsy.test(1000, render_resolution=200, canvas_class=offscreen.VisualizerCanvas, 
                     render_mode='univariate')
    vis.scale = 20.0

    mode_sequence = 'univariate', 'bivariate', 'rgb', 'rgb-hdr', 'surface'
    
    for mode in mode_sequence:
        vis.render_mode = mode
        _check_vis_output_matches_mode(vis, mode)


def test_render_mode_invalid():
    """Test that invalid render modes are properly rejected"""
    vis = topsy.test(100, render_resolution=50, canvas_class=offscreen.VisualizerCanvas)
    
    # Test setting invalid render mode
    with pytest.raises(ValueError, match="Invalid render_mode 'invalid'"):
        vis.render_mode = 'invalid'
    
    # Verify the original render mode is unchanged
    assert vis.render_mode == 'univariate'

def test_render_mode_reinitialization():
    """Test that render mode can be set during initialization"""
    modes_to_test = ['univariate', 'bivariate', 'rgb', 'rgb-hdr', 'surface']
    
    for mode in modes_to_test:
        vis = topsy.test(100, render_resolution=50, canvas_class=offscreen.VisualizerCanvas,
                         render_mode=mode)
        assert vis.render_mode == mode
        
        _check_vis_output_matches_mode(vis, mode)

class RestrictedModeOffscreenCanvas(offscreen.VisualizerCanvas):
    """A custom canvas that prevents hdr rendering"""
    def _rc_get_present_methods(self):
        return {
            "bitmap": {
                "formats": ["rgba-u8"],
            }
        }

def test_render_mode_fail():
    """Tests that if a particular render mode fails, the original render mode is restored"""
    vis = topsy.test(100, render_resolution=50, canvas_class=RestrictedModeOffscreenCanvas,
                     render_mode='univariate')
    
    original_mode = vis.render_mode
    
    # Attempt to set an invalid render mode
    with pytest.raises(ValueError):
        vis.render_mode = 'rgb-hdr' # valid, but we are forcing a failure (as will happen in Jupyter currently)
    
    # Verify that the original render mode is still intact
    assert vis.render_mode == original_mode

def _check_vis_output_matches_mode(vis, mode):
    result = vis.get_sph_image()
    result_presentation = vis.get_sph_presentation_image()
        
        # check type based on mode:
    if mode.endswith('hdr'):
        assert result_presentation.dtype == np.float16
    else:
        assert result_presentation.dtype == np.uint8

    res = vis._render_resolution

    assert result_presentation.shape == (res, res, 4)

    if mode in ['rgb', 'rgb-hdr']:
        assert result.shape == (res, res, 3)
    elif mode in ['bivariate', 'surface']:
        assert result.shape == (res, res, 2)
    else:  # univariate
        assert result.shape == (res, res)
