from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

import topsy
from topsy.drawreason import DrawReason


from topsy.canvas import offscreen
from matplotlib import pyplot as plt

@pytest.fixture
def folder():
    folder = Path(__file__).parent / "output"
    folder.mkdir(exist_ok=True)
    return folder

@pytest.fixture(params=[False, True])
def vis(request):
    vis = topsy.test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, with_cells=request.param)
    vis.scale = 200.0
    return vis


def test_render(vis, folder):
    result = vis.get_presentation_image()
    assert result.dtype == np.uint8
    # silly test, but it's better than nothing:
    assert result[:,:,0].max() == 255
    assert result[:,:,0].min() <= 5

    plt.imsave(folder / "test.png", result) # needs manual verification

def test_hdr_rgb_render(vis):
    vis = topsy.test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, hdr=True, rgb=True)
    vis.colormap.update_parameters({"min_mag": 38.0, "max_mag": 40.0})
    result = vis.get_presentation_image()[..., :3]

    assert result.dtype == np.float16
    assert result.max() > 1.0

def test_particle_pos_smooth(vis):
    # this is testing the test data
    if hasattr(vis.data_loader, '_cell_layout'):
        return # skip this test if we are using cells
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

def test_sph_weighted_output(vis, folder):
    vis.quantity_name = "test-quantity"
    vis.scale = 20.0
    vis.rotate(0.0,0.4)
    vis.render_sph(DrawReason.EXPORT)
    result = vis.get_sph_image()
    assert result.shape == (200,200)

    np.save(folder / "test_weighted.npy", result)

    test = result[::20,::20].flatten()
    expect =  [ 5.43033502e-06,  5.24370580e-06,  4.10593202e-06,  2.02327237e-06,
       -4.18346104e-07, -2.62014078e-06, -4.05347464e-06, -5.11882399e-06,
       -5.35779054e-06, -4.99097268e-06,  5.45265766e-06,  5.09395159e-06,
        3.48505409e-06,  1.16395825e-06, -8.43042187e-07, -1.45847355e-06,
       -7.42325199e-07, -2.18958917e-06, -4.82037649e-06, -4.59195871e-06,
        5.41132204e-06,  4.35603579e-06,  2.17515367e-06,  4.91636740e-07,
       -5.65283109e-08, -1.40854684e-06, -3.74410342e-06, -5.81178483e-06,
       -5.09375786e-06, -3.57636532e-06,  4.87861234e-06,  2.97656561e-06,
        1.66810560e-06,  2.19740036e-06,  9.82135475e-07, -8.12036262e-07,
       -2.89094419e-06, -7.02606530e-06, -3.94840890e-06, -1.54778229e-06,
        3.99404962e-06,  2.29679290e-06,  1.75102468e-06, -2.59838384e-06,
       -7.99023746e-06,  6.96154939e-06,  1.22214078e-05,  4.38679444e-06,
        5.60235185e-06,  1.50137737e-06,  3.14368890e-06,  1.71596139e-06,
       -1.40251836e-06, -4.41343991e-06, -3.76780258e-06,  2.00669024e-06,
       -2.16206718e-06,  1.01332953e-05,  1.07507904e-05,  3.09911479e-06,
        2.31677564e-06,  8.55237090e-07, -6.15154192e-07,  2.68512372e-06,
        2.10978737e-06, -5.41282191e-07,  1.23767513e-05,  1.22587708e-05,
        8.23176015e-06,  1.06019718e-06,  1.33470564e-06, -3.61378696e-08,
       -1.61409139e-06, -2.52063046e-06, -1.01564396e-07,  3.33199682e-06,
        3.70180169e-06,  2.76784954e-06,  2.63906173e-07, -2.26574616e-06,
       -3.76287801e-09, -6.21360527e-07, -2.10721282e-06, -3.51426252e-06,
       -4.18957279e-06, -4.20794186e-06, -4.03539661e-06, -3.86382771e-06,
       -4.00788394e-06, -4.25639519e-06, -1.58968851e-06, -1.60171351e-06,
       -2.46739432e-06, -3.65177402e-06, -4.68898270e-06, -5.36195603e-06,
       -5.62509740e-06, -5.53361224e-06, -5.27684415e-06, -5.15632610e-06]

    npt.assert_allclose(test, expect, atol=1.5e-7)

def test_sph_output(vis, folder):
    vis.render_sph(DrawReason.EXPORT)
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
    assert abs((test/expect).mean()-1.0)<0.0015
    assert (test/expect).std() < 0.015

def test_periodic_sph_output(vis, folder):
    vis2 = topsy.test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, periodic_tiling=True)
    vis2.scale = 200.0
    vis2.render_sph(DrawReason.EXPORT)
    result = vis2.get_sph_image()

    np.save(folder / "test_periodic.npy", result)

    expect = np.array([8.9322626e-08, 3.1703462e-10, 9.2687735e-10, 1.1192928e-09,
       3.1655625e-10, 8.9333170e-08, 3.2010358e-10, 9.2814262e-10,
       1.1169485e-09, 3.1108691e-10, 3.2055755e-10, 1.6879378e-10,
       2.5913338e-10, 2.5908647e-10, 1.7273429e-10, 3.3135905e-10,
       1.7230860e-10, 2.6059924e-10, 2.5635122e-10, 1.6636349e-10,
       1.3825600e-09, 2.7530075e-10, 6.2645300e-10, 6.9163519e-10,
       2.8291827e-10, 1.3944854e-09, 2.7912050e-10, 6.2802880e-10,
       6.8869560e-10, 2.7608971e-10, 8.7882529e-10, 2.4545646e-10,
       5.1961285e-10, 5.8564026e-10, 2.4835095e-10, 8.9120167e-10,
       2.4955563e-10, 5.2136878e-10, 5.8244543e-10, 2.4107497e-10,
       3.5990838e-10, 1.8307128e-10, 2.9074523e-10, 2.9106720e-10,
       1.8820685e-10, 3.7185485e-10, 1.8720442e-10, 2.9251249e-10,
       2.8785399e-10, 1.8088944e-10, 8.9335380e-08, 3.3031011e-10,
       9.4112340e-10, 1.1338572e-09, 3.3094341e-10, 8.9348021e-08,
       3.3437902e-10, 9.4284891e-10, 1.1307086e-09, 3.2375427e-10,
       3.2530026e-10, 1.7408402e-10, 2.6467059e-10, 2.6495947e-10,
       1.7861751e-10, 3.3720080e-10, 1.7819340e-10, 2.6643898e-10,
       2.6174474e-10, 1.7130720e-10, 1.3846500e-09, 2.7765840e-10,
       6.2892319e-10, 6.9430756e-10, 2.8559652e-10, 1.3971296e-09,
       2.8180488e-10, 6.3067879e-10, 6.9111544e-10, 2.7830191e-10,
       8.7712987e-10, 2.4356558e-10, 5.1764731e-10, 5.8348204e-10,
       2.4619712e-10, 8.8906421e-10, 2.4739205e-10, 5.1924093e-10,
       5.8052546e-10, 2.3932253e-10, 3.5584827e-10, 1.7859708e-10,
       2.8607963e-10, 2.8608999e-10, 1.8324889e-10, 3.6689662e-10,
       1.8223464e-10, 2.8758435e-10, 2.8330593e-10, 1.7673871e-10],
      dtype=np.float32)

    npt.assert_allclose(result[::20,::20].flatten(), expect, rtol=1e-1)

def test_rotated_sph_output(vis):
    vis.draw(reason=DrawReason.EXPORT)
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


def test_rgb_sph_output():
    vis = topsy.test(1000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas, rgb=True)
    result = vis.get_sph_image()
    assert result.shape == (200,200,3)

def test_depth_output(vis, folder):
    vis = topsy.test(1000, render_resolution=200, canvas_class=offscreen.VisualizerCanvas)
    vis.scale = 20.0
    vis.rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [0.0, -1.0, 0.0]], dtype=np.float32)
    vis.render_sph(DrawReason.EXPORT)

    np.save(folder/"test_depth_context.npy", vis.get_sph_image())

    result = vis._sph.get_depth_image(DrawReason.EXPORT)

    np.save(folder / "test_depth.npy", result)  # for debugging

    expect = np.array([-6.1412215e-01, -1.8200874e-02,  4.4874430e-01,  6.8237543e-01,
        6.7839861e-01,  4.8642159e-01,  1.8122435e-01, -1.5840769e-01,
       -4.8412681e-01, -7.8494072e-01, -7.2757006e-01, -1.7276883e-01,
        2.2809744e-01,  3.7608147e-01,  2.7173996e-01, -4.3261051e-03,
       -3.3126712e-01, -5.7852268e-01, -7.1909428e-01, -8.2638502e-01,
       -9.6994162e-01, -4.7681451e-01, -1.4445305e-01, -9.2931986e-02,
       -3.0393243e-01, -6.7068458e-01, -1.2243545e+00, -1.4892983e+00,
       -1.2386465e+00, -9.9033833e-01, -1.2633693e+00, -8.1685185e-01,
       -5.0349355e-01, -4.2420983e-01, -4.1854501e-01, -7.6590419e-01,
       -2.4770129e+00, -3.5506952e+00, -2.4098432e+00, -1.2935197e+00,
       -1.5107369e+00, -1.0751343e+00, -6.8476319e-01, -3.7224770e-01,
       -1.9266486e-01, -5.1854849e-01, -2.8899932e+00, -5.6981363e+00,
       -4.1787076e+00, -1.8048847e+00, -1.6587865e+00, -1.2280405e+00,
       -7.5689793e-01, -3.1625152e-01, -1.5581965e-01, -4.8974395e-01,
       -3.0715656e+00, -6.4643850e+00, -4.9952149e+00, -2.1794629e+00,
       -1.7302644e+00, -1.3548887e+00, -9.3402386e-01, -5.6250334e-01,
       -3.4593463e-01, -7.3299527e-01, -3.0981314e+00, -5.8333817e+00,
       -4.2573762e+00, -1.8872452e+00, -1.7559803e+00, -1.5045440e+00,
       -1.1963260e+00, -9.8268032e-01, -7.4352264e-01, -1.0188949e+00,
       -2.1862769e+00, -3.7116265e+00, -2.6124454e+00, -1.2816906e+00,
       -1.7749834e+00, -1.6487813e+00, -1.4966643e+00, -1.3906789e+00,
       -1.3710022e+00, -1.5978980e+00, -1.9147623e+00, -1.9956529e+00,
       -1.5321469e+00, -9.3367457e-01, -1.8167615e+00, -1.7895186e+00,
       -1.7594409e+00, -1.7537165e+00, -1.7831147e+00, -1.8056393e+00,
       -1.7693257e+00, -1.6105938e+00, -1.3042843e+00, -9.7951174e-01],
      dtype=np.float32)

    npt.assert_allclose(result[::20,::20].ravel(), expect, atol=1e-1)
