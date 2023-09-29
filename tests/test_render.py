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
    vis = topsy._test(10000, render_resolution=200, canvas_class = offscreen.VisualizerCanvas)

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

    expect = np.array([3.9287383e-13, 5.5050257e-13, 7.6353989e-13, 1.0141182e-12,
       1.2393138e-12, 1.3240675e-12, 1.2094989e-12, 9.7496121e-13,
       7.3494753e-13, 5.3885184e-13, 5.5531962e-13, 8.6450085e-13,
       1.3481805e-12, 2.0611694e-12, 2.8464824e-12, 3.1656967e-12,
       2.7328623e-12, 1.9321500e-12, 1.2500829e-12, 8.1329621e-13,
       7.8355118e-13, 1.3631815e-12, 2.5299955e-12, 4.9601534e-12,
       8.3184406e-12, 9.9214205e-12, 7.7984988e-12, 4.4260927e-12,
       2.2768317e-12, 1.2318675e-12, 1.0535535e-12, 2.0777362e-12,
       5.0366386e-12, 1.4153312e-11, 3.6467555e-11, 5.4102028e-11,
       3.1363929e-11, 1.1786298e-11, 4.2883514e-12, 1.8357449e-12,
       1.2924806e-12, 2.9085404e-12, 9.0102639e-12, 4.0556905e-11,
       2.7220667e-10, 1.2469481e-09, 2.3019615e-10, 2.9873597e-11,
       7.1707609e-12, 2.5135124e-12, 1.3953204e-12, 3.3737598e-12,
       1.1420813e-11, 6.7440817e-11, 1.1875348e-09, 1.7286030e-07,
       1.0353859e-09, 5.0008213e-11, 8.8475668e-12, 2.8494353e-12,
       1.2956567e-12, 3.0306443e-12, 9.2138432e-12, 3.9570687e-11,
       2.7873756e-10, 8.7612456e-10, 2.3548541e-10, 3.2350695e-11,
       7.3600839e-12, 2.5689867e-12, 1.0495618e-12, 2.1662457e-12,
       5.3645061e-12, 1.4507441e-11, 3.7246033e-11, 5.5721056e-11,
       3.3367049e-11, 1.2087110e-11, 4.4923006e-12, 1.9295540e-12,
       7.7805151e-13, 1.3838001e-12, 2.6602041e-12, 5.2666170e-12,
       9.0259146e-12, 1.0758684e-11, 8.2174345e-12, 4.7123286e-12,
       2.4574930e-12, 1.3119297e-12, 5.5245814e-13, 8.6899753e-13,
       1.3761681e-12, 2.1119391e-12, 2.9193512e-12, 3.2884075e-12,
       2.8896930e-12, 2.0867015e-12, 1.3551895e-12, 8.6546113e-13],
      dtype=np.float32)
    npt.assert_allclose(test, expect, rtol=1e-4)