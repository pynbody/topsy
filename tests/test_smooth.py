"""Test the bilateral filtering smoothing operation in ColorAsSurfaceMap."""

import numpy as np
from pathlib import Path
import topsy
from topsy.canvas import offscreen
from topsy.colormap.surface import ColorAsSurfaceMap


def create_test_image(width=256, height=256):
    """Create a two-channel test image with noise, gradient, and discontinuity."""

    np.random.seed(1337)

    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize two-channel image
    test_image = np.zeros((height, width, 2), dtype=np.float32)
    
    # Channel 0: depth/height values
    # Add gradient
    gradient = X * 0.5 + Y * 0.3
    
    # Add discontinuity in the middle
    discontinuity = np.zeros_like(gradient)
    discontinuity[height//4:3*height//4, width//4:3*width//4] = 0.5
    
    # Add gaussian noise
    noise = np.random.normal(0, 0.05, (height, width))
    
    test_image[:, :, 0] = gradient + discontinuity + noise
    
    # Channel 1: density/mass values
    # Similar structure but different values
    gradient2 = Y * 0.4 + X * 0.2
    discontinuity2 = np.zeros_like(gradient2)
    discontinuity2[height//3:2*height//3, width//3:2*width//3] = 0.3
    noise2 = np.random.normal(0, 0.03, (height, width))
    
    test_image[:, :, 1] = gradient2 + discontinuity2 + noise2
    
    # Ensure all values are positive (typical for SPH data)
    test_image = np.abs(test_image) + 0.01
    
    return test_image


def test_smoothing_operation():
    """Test the smoothing operation using ColorAsSurfaceMap._smooth_numpy."""
    # Create output folder
    folder = Path(__file__).parent / "output"
    folder.mkdir(exist_ok=True)

    test_image = create_test_image()

    vis = topsy.test(100, render_resolution=test_image.shape[0], canvas_class=offscreen.VisualizerCanvas)

    vis.colormap.update_parameters({
        'type': 'surface',
        'smoothing_scale': 0.02,
    })
    
    # Get the surface colormap instance
    surface_map = vis.colormap._impl

    smoothed_output = surface_map._smooth_numpy(test_image)
    
    # Save outputs
    np.save(folder / 'test_smooth_input.npy', test_image)
    np.save(folder / 'test_smooth_output.npy', smoothed_output)

    assert False # we want this always to fail for now until we design the right test.



if __name__ == "__main__":
    test_smoothing_operation()