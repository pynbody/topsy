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
    vis.draw(reason=DrawReason.EXPORT)
    result = vis.get_presentation_image()
    plt.imsave(folder / "test.png", result) # needs manual verification

def test_sph_output():
    vis.draw(reason=DrawReason.EXPORT)
    result = vis.get_sph_image()
    assert result.shape == (200,200)
    np.save(folder / "test.npy", result) # for debugging
    test = result[::20,::20].flatten()

    print(repr(test))

    expect = np.array([1.3152095e-13, 2.1683958e-13, 3.3428411e-13, 4.6267482e-13,
       5.5460758e-13, 5.6280165e-13, 4.8377491e-13, 3.6066842e-13,
       2.4283730e-13, 1.5503958e-13, 2.2529823e-13, 4.1573822e-13,
       7.1293768e-13, 1.0756402e-12, 1.3646146e-12, 1.3923820e-12,
       1.1335476e-12, 7.7611799e-13, 4.6997383e-13, 2.6606067e-13,
       3.7777916e-13, 7.7990191e-13, 1.5024050e-12, 2.6028316e-12,
       3.7975599e-12, 4.1846604e-12, 3.0966542e-12, 1.7241240e-12,
       8.9327623e-13, 4.5005753e-13, 5.8374025e-13, 1.3346154e-12,
       2.9382217e-12, 6.2685222e-12, 1.3794959e-11, 2.1916914e-11,
       1.3405932e-11, 4.7136535e-12, 1.6706981e-12, 7.0424227e-13,
       7.8486653e-13, 1.9346662e-12, 4.8317461e-12, 1.4285845e-11,
       8.3038985e-11, 2.6427677e-10, 8.7739101e-11, 1.3530749e-11,
       2.9550841e-12, 9.8121402e-13, 8.8169749e-13, 2.2386858e-12,
       5.9199824e-12, 2.2553770e-11, 2.3434868e-10, 8.0104826e-08,
       2.3312002e-10, 2.0885210e-11, 3.7275925e-12, 1.1423055e-12,
       8.1536145e-13, 2.0178529e-12, 5.0764935e-12, 1.6284897e-11,
       8.2104219e-11, 2.3366509e-10, 7.3335504e-11, 1.2568673e-11,
       2.9692683e-12, 1.0564956e-12, 6.2675744e-13, 1.4526925e-12,
       3.2491332e-12, 7.1263360e-12, 1.5710695e-11, 2.2008189e-11,
       1.2040994e-11, 4.5604913e-12, 1.8561702e-12, 7.9805769e-13,
       4.0857720e-13, 8.8350003e-13, 1.7419240e-12, 3.0378955e-12,
       4.3733836e-12, 4.5352992e-12, 3.3447255e-12, 2.0113982e-12,
       1.0732789e-12, 5.2568838e-13, 2.3264011e-13, 4.7279840e-13,
       8.5326163e-13, 1.3296205e-12, 1.7072913e-12, 1.7436266e-12,
       1.4297575e-12, 9.7699518e-13, 5.7949695e-13, 3.1307056e-13],
      dtype=np.float32)
    npt.assert_allclose(test, expect, rtol=1e-2)