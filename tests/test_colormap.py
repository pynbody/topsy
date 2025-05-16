import topsy
import pytest
import numpy as np

from topsy import colormap
from topsy.canvas import offscreen

@pytest.fixture
def vis(request):
    vis = topsy.test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)
    vis.scale = 200.0
    return vis

def test_colormap(vis):
    test_image = np.zeros((200, 200, 2), dtype=np.float32)
    test_image[:, :, 0] = np.linspace(0, 1, 200)
    test_image[:, :, 1] = np.linspace(0, 1, 200)

    vis.colormap.sph_raw_output_to_image(test_image)