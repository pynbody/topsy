import topsy
import pytest
import pylab as p
import numpy as np
import numpy.testing as npt

from topsy import colormap
from topsy.canvas import offscreen

@pytest.fixture
def vis(request):
    vis = topsy.test(100, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)
    vis.scale = 200.0
    return vis

def test_colormap(vis):
    input_image = np.zeros((200, 200, 2), dtype=np.float32)
    input_image[:, :, 0] = np.linspace(0, 1, 200)
    input_image[:, :, 1] = np.linspace(0, 1, 200)

    vis.colormap.vmin = 0
    vis.colormap.vmax = 1
    vis.colormap.log_scale = False
    image = vis.colormap.sph_raw_output_to_image(input_image)

    assert image.shape == (200, 200, 4)

    mpl_cmap = p.get_cmap(vis.colormap_name)
    image_via_mpl = (mpl_cmap(input_image[:, :, 0])*255).astype(np.uint8)

    npt.assert_allclose(image, image_via_mpl, atol=2)
