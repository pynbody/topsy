"""Test the bilateral filtering smoothing operation in ColorAsSurfaceMap."""

import numpy as np
import numpy.testing as npt
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

    # no smoothing on channel 0
    npt.assert_allclose(test_image[..., 0], smoothed_output[..., 0], atol=1e-7)

    # channel 1 is smoothed but hard edges are still there
    expected_global_samples = [0.04350269, 0.03492163, 0.03985117, 0.06765869, 0.09567888,
       0.08533357, 0.10505654, 0.10955958, 0.12479778, 0.14756411,
       0.15113021, 0.193778  , 0.17523961, 0.03672419, 0.04862463,
       0.06353201, 0.07946779, 0.10003348, 0.11344124, 0.1363642 ,
       0.14499053, 0.14822552, 0.18607135, 0.19857994, 0.21028055,
       0.2280225 , 0.04706344, 0.08421917, 0.11555166, 0.10795519,
       0.13088886, 0.14922458, 0.16380104, 0.18379168, 0.21592017,
       0.21860206, 0.22268587, 0.26771224, 0.25449008, 0.11944734,
       0.10907468, 0.14002527, 0.14063616, 0.16814029, 0.199773  ,
       0.1959068 , 0.21331303, 0.230805  , 0.2333972 , 0.25765777,
       0.2804561 , 0.279846  , 0.12494649, 0.1561663 , 0.17562656,
       0.18248414, 0.19247337, 0.21708317, 0.23262993, 0.24562259,
       0.25853062, 0.28892142, 0.28311318, 0.29793793, 0.33640784,
       0.17010446, 0.18930109, 0.21303204, 0.23668505, 0.2234434 ,
       0.53724605, 0.5470026 , 0.59243613, 0.58226967, 0.30523527,
       0.32288015, 0.34913924, 0.3622239 , 0.19013162, 0.22471175,
       0.23146637, 0.24505465, 0.24869259, 0.56404704, 0.577596  ,
       0.60139513, 0.63722885, 0.33896458, 0.3310112 , 0.36426947,
       0.37940887, 0.24438018, 0.23389104, 0.27845004, 0.27066812,
       0.30080408, 0.61517453, 0.6194309 , 0.6452768 , 0.65323645,
       0.36640742, 0.4021577 , 0.39369363, 0.40901196, 0.2726592 ,
       0.27804396, 0.27932608, 0.327759  , 0.32077065, 0.6381103 ,
       0.65829104, 0.6525516 , 0.67309   , 0.40183058, 0.4239184 ,
       0.42731968, 0.4529117 , 0.29238752, 0.31701306, 0.33301896,
       0.3448711 , 0.34038064, 0.37747476, 0.38808158, 0.39679116,
       0.4076333 , 0.43214443, 0.46319935, 0.4567135 , 0.47022748,
       0.3328091 , 0.34471515, 0.35241255, 0.38199195, 0.40559343,
       0.3941699 , 0.41470632, 0.4342157 , 0.44876787, 0.45069033,
       0.4712601 , 0.4940555 , 0.49560383, 0.3539791 , 0.35014838,
       0.374198  , 0.41407102, 0.40864894, 0.4354274 , 0.45889753,
       0.4607982 , 0.48360315, 0.49325395, 0.5230214 , 0.5123342 ,
       0.54918534, 0.3842421 , 0.40161213, 0.41294298, 0.4332912 ,
       0.4545948 , 0.47875527, 0.47641772, 0.4919833 , 0.52749985,
       0.53773326, 0.53975433, 0.55892164, 0.56311625]
    
    global_check = smoothed_output[::20, ::20, 1].ravel()

    expected_edge_check = [0.19247337, 0.19723694, 0.1849934 , 0.19642891, 0.19979529,
       0.1993649 , 0.20303836, 0.2173118 , 0.1889234 , 0.20993778,
       0.19258004, 0.22300835, 0.20747848, 0.20263639, 0.2036718 ,
       0.20567518, 0.21924324, 0.20507486, 0.19026951, 0.20912749,
       0.19608739, 0.20039833, 0.19389133, 0.19785273, 0.19580497,
       0.20818928, 0.20516331, 0.20875177, 0.21691433, 0.18723641,
       0.21353379, 0.19767466, 0.20573969, 0.1855796 , 0.19924074,
       0.21442725, 0.1919996 , 0.17996098, 0.20208283, 0.21387406,
       0.24663831, 0.20913196, 0.19462162, 0.21180561, 0.16858205,
       0.21117128, 0.20315671, 0.20511323, 0.21663508, 0.20262529,
       0.20380434, 0.19074719, 0.1645996 , 0.2216465 , 0.2202986 ,
       0.51710486, 0.5235424 , 0.51853245, 0.51737845, 0.5172093 ,
       0.20266144, 0.19300404, 0.19269636, 0.20673202, 0.20537308,
       0.5252804 , 0.5193447 , 0.5337641 , 0.51419616, 0.5214026 ,
       0.20528564, 0.1887608 , 0.22220144, 0.20611644, 0.2162794 ,
       0.5282587 , 0.5235715 , 0.5250429 , 0.532619  , 0.53551286,
       0.20328672, 0.20438206, 0.20458573, 0.2203121 , 0.22026433,
       0.5077367 , 0.5264619 , 0.52011055, 0.5161084 , 0.5056762 ,
       0.2132591 , 0.21822827, 0.19445635, 0.21045099, 0.22532488,
       0.5191735 , 0.530404  , 0.52163655, 0.5298376 , 0.5205669 ]
    
    edge_check = smoothed_output[80:90, 80:90, 1].ravel()

    npt.assert_allclose(global_check, expected_global_samples, atol=1e-6)
    npt.assert_allclose(edge_check, expected_edge_check, atol=1e-6)
    


if __name__ == "__main__":
    test_smoothing_operation()