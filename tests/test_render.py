from pathlib import Path

import numpy as np
import numpy.testing as npt

import topsy


from topsy.canvas import offscreen
from matplotlib import pyplot as plt


def setup_module():
    global vis, folder
    np.random.seed(1337)
    vis = topsy._test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)

    folder = Path(__file__).parent / "output"
    folder.mkdir(exist_ok=True)

def test_render():
    result = vis.canvas.draw()
    image = np.frombuffer(result, dtype=np.dtype('u1')).reshape((480,640,4))
    plt.imsave(folder / "test.png", image) # needs manual verification

def test_sph_output():
    vis.canvas.draw()
    result = vis.get_sph_image()
    assert result.shape == (200,200)
    np.save(folder / "test.npy", result) # for debugging
    test = result[::20,::20].flatten()
    expect = np.array([6.6993198e-14, 1.1507333e-13, 1.7919646e-13, 2.4612135e-13,
       2.9066495e-13, 2.8998280e-13, 2.4486413e-13, 1.7858511e-13,
       1.1692387e-13, 7.2200106e-14, 1.2295605e-13, 2.3070537e-13,
       3.9220626e-13, 5.8140309e-13, 7.2356704e-13, 7.2395855e-13,
       5.7850269e-13, 3.8981690e-13, 2.3143754e-13, 1.2701774e-13,
       2.1470363e-13, 4.4307169e-13, 8.4494868e-13, 1.4483490e-12,
       2.0638049e-12, 2.2089444e-12, 1.6153356e-12, 8.8054764e-13,
       4.4565604e-13, 2.1992054e-13, 3.3929980e-13, 7.7498327e-13,
       1.7074509e-12, 3.5581679e-12, 7.1763611e-12, 1.1240477e-11,
       6.7540001e-12, 2.4406371e-12, 8.5188578e-13, 3.5051858e-13,
       4.6225659e-13, 1.1472580e-12, 2.8631052e-12, 7.8389873e-12,
       4.1710101e-11, 1.3738759e-10, 4.3650156e-11, 6.5835722e-12,
       1.5117496e-12, 4.9551118e-13, 5.2268747e-13, 1.3379199e-12,
       3.5099410e-12, 1.2415209e-11, 1.2423079e-10, 4.4262038e-08,
       1.2358496e-10, 9.5642834e-12, 1.8539040e-12, 5.8338073e-13,
       4.8245164e-13, 1.1989600e-12, 2.9855599e-12, 9.4497708e-12,
       4.7168668e-11, 1.2785455e-10, 3.6405660e-11, 5.7495324e-12,
       1.4845271e-12, 5.4782660e-13, 3.6776891e-13, 8.5198087e-13,
       1.8806679e-12, 4.1391638e-12, 9.1587961e-12, 1.2461185e-11,
       6.2175061e-12, 2.2534507e-12, 9.6120670e-13, 4.2535973e-13,
       2.3635561e-13, 5.1153889e-13, 9.9384986e-13, 1.7036724e-12,
       2.4265887e-12, 2.4533247e-12, 1.7540835e-12, 1.0575305e-12,
       5.7956190e-13, 2.8566545e-13, 1.3080068e-13, 2.7080400e-13,
       4.8454578e-13, 7.4089178e-13, 9.3455235e-13, 9.4375126e-13,
       7.7476540e-13, 5.3613396e-13, 3.2055389e-13, 1.6914180e-13],
      dtype=np.float32)
    npt.assert_allclose(test, expect, rtol=1e-4)