import topsy
import pytest
import pylab as p
import numpy as np
import numpy.testing as npt

from pathlib import Path
from matplotlib import colors, cm

from topsy import colormap
from topsy.canvas import offscreen

@pytest.fixture
def folder():
    folder = Path(__file__).parent / "output"
    folder.mkdir(exist_ok=True)
    return folder


@pytest.fixture
def vis(request):
    vis = topsy.test(100, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)
    vis.scale = 200.0
    return vis

@pytest.fixture
def input_image():
    """A dummy output from the SPH renderer, where the density varies in the x direction logarithmically between 10^-3
    and 1 and the weighted average varies in the y direction linearly between 0 and 1."""
    input_image = np.empty((200, 200, 2), dtype=np.float32)
    input_image[:, :, 0] = np.logspace(-3, 0, 200)
    input_image[:, :, 1] = np.linspace(0, 1, 200)[:, np.newaxis] * input_image[:, :, 0]
    return input_image

@pytest.mark.parametrize("mode", ['density', 'weighted-average', 'bivariate'])
@pytest.mark.parametrize("log_scale", [True, False], ids=["log", "linear"])
def test_colormap(vis, input_image, mode, log_scale, folder):
    cmap = vis.colormap

    if mode == 'density':
        weighted_average = False
        type = 'density'
        if log_scale:
            vmin, vmax = -3.0, 0.0
        else:
            vmin, vmax = 0.0, 1.0
    elif mode == 'weighted-average':
        weighted_average = True
        type = 'density'
        if log_scale:
            vmin, vmax = -2.0, 0.0
        else:
            vmin, vmax = 0.0, 1.0
    elif mode == 'bivariate':
        weighted_average = True
        type = 'bivariate'
        if log_scale:
            vmin, vmax = -2.0, 0.0
        else:
            vmin, vmax = 0.0, 1.0
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    cmap.update_parameters({
        'type': type,
        'weighted_average': weighted_average,
        'vmin': vmin,
        'vmax': vmax,
        'density_vmin': -3.0,
        'density_vmax': 0.0,
        'log': log_scale,
    })

    image = cmap.sph_raw_output_to_image(input_image)

    assert image.shape == (200, 200, 4)

    p.imsave(folder / f"test_colormap_{mode}_{log_scale}.png", image)

    image_via_mpl = _colormap_in_software(input_image, cmap, log_scale, vmax, vmin)
    p.imsave(folder / f"test_colormap_software_{mode}_{log_scale}.png", image_via_mpl)

    npt.assert_allclose(image, image_via_mpl, atol=5)


def _colormap_in_software(input_image, cmap, log_scale, vmax, vmin):
    if cmap.get_parameter("type") == "bivariate":
        return _bivariate_colormap_in_software(input_image, cmap)
    else:
        return _univariate_colormap_in_software(input_image, cmap)

def _univariate_colormap_in_software(input_image, cmap):
    mpl_cmap_name = cmap.get_parameter("colormap_name")
    vmin = cmap.get_parameter("vmin")
    vmax = cmap.get_parameter("vmax")
    log_scale = cmap.get_parameter("log")

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mpl_cmap = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(mpl_cmap_name)).to_rgba

    content = cmap.sph_raw_output_to_content(input_image)
    if log_scale:
        content = np.log10(content)
    image_via_mpl = (mpl_cmap(content) * 255).astype(np.uint8)
    return image_via_mpl

def _bivariate_colormap_in_software(input_image, cmap):
    den_vmin = cmap.get_parameter("density_vmin")
    den_vmax = cmap.get_parameter("density_vmax")
    vmin = cmap.get_parameter("vmin")
    vmax = cmap.get_parameter("vmax")
    vlog = cmap.get_parameter("log")

    # generate a 1000 x 1000 grid of points. The mapping is a 2D grid of points in the unit square
    mapping = cmap._impl._generate_mapping_rgba_f32(1000)

    # for each point in  the input image, figure out the coordinate in the mapping
    density = input_image[:, :, 0]
    value_times_density = input_image[:, :, 1]
    weighted_value = value_times_density / density

    scaled_density = (np.log10(density) - den_vmin) / (den_vmax - den_vmin)
    if vlog:
        scaled_weighted_value = (np.log10(weighted_value) - vmin) / (vmax - vmin)
    else:
        scaled_weighted_value = (weighted_value - vmin) / (vmax - vmin)

    # set up an interpolator for the mapping on the unit square. Out-of-bounds values should map to nearest
    from scipy.interpolate import RegularGridInterpolator
    points = np.linspace(0, 1, 1000)
    interpolator = RegularGridInterpolator((points, points), mapping, bounds_error=True, method='linear')

    # now interpolate the mapping for each point in the input image
    coords = np.stack((scaled_weighted_value, scaled_density), axis=-1)
    coords = np.clip(coords, 0, 1)  # ensure coordinates are within [0, 1]
    image = np.clip(interpolator(coords), 0, 1)

    # convert to 8-bit RGBA
    image = (image * 255).astype(np.uint8)

    return image




def test_colormap_holder_instantiation(vis):
    from topsy.colormap import ColormapHolder
    from topsy.colormap.implementation import RGBColormap, RGBHDRColormap, BivariateColormap, Colormap
    specs = [
        {"params": {"type": "rgb", "hdr": True},
         "expected": RGBHDRColormap},
        {"params": {"type": "rgb", "hdr": False},
         "expected": RGBColormap},
        {"params": {"type": "bivariate", "hdr": False},
         "expected": BivariateColormap},
        {"params": {"type": "density", "hdr": False},
         "expected": Colormap},
    ]

    for spec in specs:
        colormap_class = ColormapHolder.instance_from_parameters(spec["params"], vis.device,
                                                                 vis._sph.get_output_texture(),
                                                                 vis.canvas_format)
        assert type(colormap_class) == spec["expected"]


def test_colormap_updating(vis):
    """Test that updating the colormap correctly decides whether to create a new implementation or not"""
    cmap = vis.colormap
    cmap.update_parameters({'type': 'density'})
    assert isinstance(cmap._impl, colormap.implementation.Colormap)
    impl_id = id(cmap._impl)

    cmap.update_parameters({'vmin': 0.0, 'vmax': 20.0})
    assert impl_id == id(cmap._impl)  # should not create a new implementation

    
    cmap.update_parameters({'type': 'bivariate'})
    assert isinstance(cmap._impl, colormap.implementation.BivariateColormap)
    assert impl_id != id(cmap._impl)  # should create a new implementation

def test_rgb_colormap_vmin_vmax():
    """Test that RGB colormap can be updated either with vmin/vmax or with min_mag/max_mag"""
    vis = topsy.test(100, render_resolution=200, canvas_class=offscreen.VisualizerCanvas, rgb=True)

    vis.colormap.update_parameters({'vmin': 1.0, 'vmax': 2.0})
    assert vis.colormap.get_parameter('vmin') == 1.0
    assert vis.colormap.get_parameter('vmax') == 2.0
    assert np.allclose(vis.colormap.get_parameter('min_mag'), 31.57212566586528)
    assert np.allclose(vis.colormap.get_parameter('max_mag'), 34.07212566586528)

    vis.colormap.update_parameters({'min_mag': 1.0, 'max_mag': 2.0})
    assert np.allclose(vis.colormap.get_parameter('min_mag'), 1.0)
    assert np.allclose(vis.colormap.get_parameter('max_mag'), 2.0)
    assert np.allclose(vis.colormap.get_parameter('vmin'), 13.828850266346112)
    assert np.allclose(vis.colormap.get_parameter('vmax'), 14.228850266346113)


def test_colormap_dict_access(vis):
    """Test that the colormap can be accessed as a dictionary"""
    cmap = vis.colormap
    cmap.update_parameters({'type': 'density', 'vmin': 0.0, 'vmax': 20.0})

    assert cmap['type'] == 'density'
    assert cmap['vmin'] == 0.0
    assert cmap['vmax'] == 20.0

    cmap['vmin'] = 5.0
    assert cmap['vmin'] == 5.0
