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

@pytest.fixture
def input_image():
    input_image = np.empty((200, 200, 2), dtype=np.float32)
    input_image[:, :, 0] = np.logspace(-3, 0, 200)
    input_image[:, :, 1] = np.linspace(0, 1, 200) * input_image[:, :, 0]
    return input_image

@pytest.mark.parametrize("weighted_average", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_colormap(vis, input_image, weighted_average, log_scale):
    cmap = topsy.colormap.Colormap(vis.device, vis.colormap_name, vis._sph.get_output_texture(),
                                   vis.canvas_format, weighted_average)

    cmap.vmin = 0
    cmap.vmax = 1
    cmap.log_scale = log_scale

    image = cmap.sph_raw_output_to_image(input_image)

    assert image.shape == (200, 200, 4)

    mpl_cmap = p.get_cmap(vis.colormap_name)

    # now do it via matplotlib for comparison
    content = cmap.sph_raw_output_to_content(input_image)
    if log_scale:
        content = np.log10(content)
    image_via_mpl = (mpl_cmap(content)*255).astype(np.uint8)

    npt.assert_allclose(image, image_via_mpl, atol=5)

    if weighted_average:
        p.imsave("test_colormap.png", image)
        p.imsave("test_colormap_mpl.png", image_via_mpl)


def test_colormap_holder_instantiation(vis):
    from topsy.colormap import ColormapHolder, RGBColormap, RGBHDRColormap, BivariateColormap, WeightedColormap, Colormap
    specs = [
        {"params": {"type": "rgb", "hdr": True},
         "expected": RGBHDRColormap},
        {"params": {"type": "rgb", "hdr": False},
         "expected": RGBColormap},
        {"params": {"type": "bivariate", "hdr": False},
         "expected": BivariateColormap},
        {"params": {"type": "weighted", "hdr": False},
         "expected": WeightedColormap},
        {"params": {"type": "density", "hdr": False},
         "expected": Colormap}
    ]

    for spec in specs:
        colormap_class = ColormapHolder.instance_from_parameters(spec["params"], vis.device,
                                                                 vis._sph.get_output_texture(),
                                                                 vis.canvas_format)
        assert type(colormap_class) == spec["expected"]


