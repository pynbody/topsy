from pathlib import Path

import numpy as np
import numpy.testing as npt

import topsy
from topsy.drawreason import DrawReason


from topsy.canvas import offscreen
from matplotlib import pyplot as plt


def setup_module():
    global vis, folder
    np.random.seed(1337)
    vis = topsy._test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)

    folder = Path(__file__).parent / "output"
    folder.mkdir(exist_ok=True)

def test_render():
    result = vis.get_presentation_image()
    assert result.dtype == np.uint8
    # silly test, but it's better than nothing:
    assert result[:,:,0].max() == 255
    assert result[:,:,0].min() <= 5

    plt.imsave(folder / "test.png", result) # needs manual verification


def test_hdr_render():
    vis = topsy._test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, hdr=True)
    result = vis.get_presentation_image()

    assert result.dtype == np.float16
    assert result.max() > 1.0

def test_particle_pos_smooth():
    # this is testing the test data
    xyzw = vis.data_loader.get_pos_smooth()
    npt.assert_allclose(xyzw[::100],
       [[ 1.6189760e+01, -4.0728635e-01, -1.8409515e+01,  2.0848181e+01],
       [-3.6236227e-01,  1.9854842e-02, -3.4908600e+00,  1.2997785e+00],
       [ 5.6721487e+00, -8.8317409e-02, -9.4208164e+00,  1.0804868e+01],
       [-3.6954129e+00, -5.1248574e+00,  1.4329858e+01,  1.5497326e+01],
       [-2.5594389e+01, -9.0724382e+00, -3.3397295e+00,  2.3571991e+01],
       [-3.6231318e-01,  1.6435374e-02,  1.8260944e+00,  1.0799329e+00],
       [ 9.7273951e+00,  1.8408798e-01, -6.7287006e+00,  1.3380475e+01],
       [ 1.4229246e+01,  2.2913518e+00, -1.6219862e+01,  1.8701763e+01],
       [ 1.0776958e+01,  1.6861650e+01,  1.8014458e+01,  2.3113770e+01],
       [ 8.6214191e-01, -1.3920928e-02,  1.7059642e+00,  1.0834730e+00]])



def test_sph_output():
    result = vis.get_sph_image()
    assert result.shape == (200,200)
    np.save(folder / "test.npy", result) # for debugging
    test = result[::20,::20].flatten()

    expect = np.array([1.3719737e-13, 2.2577373e-13, 3.4707128e-13, 4.7913626e-13,
       5.7335636e-13, 5.8166834e-13, 5.0068169e-13, 3.7394534e-13,
       2.5203345e-13, 1.6070023e-13, 2.3526553e-13, 4.3171158e-13,
       7.3597628e-13, 1.1049305e-12, 1.4004169e-12, 1.4294217e-12,
       1.1646425e-12, 7.9952174e-13, 4.8636567e-13, 2.7627079e-13,
       3.9335820e-13, 8.0497160e-13, 1.5378436e-12, 2.6523881e-12,
       3.8650120e-12, 4.2805867e-12, 3.1845073e-12, 1.7736138e-12,
       9.1958971e-13, 4.6606371e-13, 6.0535415e-13, 1.3683649e-12,
       2.9855350e-12, 6.3432588e-12, 1.4027483e-11, 2.2303675e-11,
       1.3719674e-11, 4.8580844e-12, 1.7220782e-12, 7.2667176e-13,
       8.1093319e-13, 1.9735192e-12, 4.8798843e-12, 1.4427965e-11,
       8.4100574e-11, 2.6442615e-10, 8.8517936e-11, 1.3847874e-11,
       3.0435474e-12, 1.0120076e-12, 9.0955282e-13, 2.2799319e-12,
       5.9699832e-12, 2.2838169e-11, 2.3453128e-10, 8.0560795e-08,
       2.3452809e-10, 2.1240560e-11, 3.8278256e-12, 1.1763834e-12,
       8.4215100e-13, 2.0578621e-12, 5.1257293e-12, 1.6460699e-11,
       8.2461420e-11, 2.3309185e-10, 7.4100032e-11, 1.2828785e-11,
       3.0442216e-12, 1.0872116e-12, 6.5038285e-13, 1.4875639e-12,
       3.2932765e-12, 7.1966786e-12, 1.5944173e-11, 2.2467811e-11,
       1.2348589e-11, 4.6734639e-12, 1.8962074e-12, 8.2295775e-13,
       4.2770960e-13, 9.1109151e-13, 1.7781172e-12, 3.0819160e-12,
       4.4282217e-12, 4.6020904e-12, 3.4024370e-12, 2.0485410e-12,
       1.1009980e-12, 5.4363285e-13, 2.4588935e-13, 4.9328543e-13,
       8.7989062e-13, 1.3605021e-12, 1.7387084e-12, 1.7740940e-12,
       1.4578791e-12, 1.0015811e-12, 5.9854254e-13, 3.2601019e-13],
      dtype=np.float32)

    # the following test will seem very weak, but the problem is that different handling of e.g.
    # mipmaps by different renderers means that individual pixels can easily be very different
    # from any reference image. So we just check that the overall image is similar.
    npt.assert_allclose(test, expect, rtol=5e-1)

    # now let's also check that the distribution is sharply peaked around the right value
    assert abs((test/expect).mean()-1.0)<0.001
    assert (test/expect).std() < 0.015

def test_periodic_sph_output():
    vis2 = topsy._test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, periodic_tiling=True)
    result = vis2.get_sph_image()
    result_untiled = vis.get_sph_image()
    assert result.std() > 3*result_untiled.std()

def test_rotated_sph_output():
    unrotated_output = vis.get_sph_image()
    vis.rotation_matrix = np.array([[0.0, 1.0, 0.0],
                                    [-1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0]], dtype=np.float32)
    try:
        vis.draw(reason=DrawReason.EXPORT)
        rotated_output = vis.get_sph_image()
        npt.assert_allclose(unrotated_output.T[:,::-1], rotated_output, rtol=5e-2)


    finally:
        vis.rotation_matrix = np.eye(3, dtype=np.float32)

