import numpy as np

from topsy.recorder import interpolator

def test_step_interpolator():
    timestream = [(0.4, None), (1.0, "Hello"), (2.0, "World")]
    interp = interpolator.StepInterpolator(timestream)
    assert interp(0.0) is interp.no_value
    assert interp(0.5) is None
    assert interp(1.0) == "Hello"
    assert interp(1.5) is interp.no_value
    assert interp(1.9) is interp.no_value
    assert interp(2.1) == "World"

def test_linear_interpolator():
    timestream = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (4.0, 0.0)]
    interp = interpolator.LinearInterpolator(timestream)
    assert interp(0.0) == 0.0
    assert interp(0.5) == 0.5
    assert interp(1.0) == 1.0
    assert interp(1.5) == 2.5
    assert interp(2.0) == 4.0
    assert interp(3.0) == 2.0
    assert interp(4.0) == 0.0
    assert interp(4.01) is interp.no_value

def test_smoothed_step_interpolation_with_none():
    timestream = [(0.0, 0.0), (1.0, None), (2.0, 4.0), (4.0, 0.0)]
    interp = interpolator.SmoothedStepInterpolator(timestream)
    assert interp(0.0) == 0.0
    assert interp(0.5) is interp.no_value
    assert interp(1.99) is interp.no_value
    assert interp(2.01) == 4.0
    assert interp(3.0) is interp.no_value
    assert interp(4.0) == 4.0
    assert interp(4.125) == 2.0
    assert interp(4.25) == 0.0
    assert interp(4.5) is interp.no_value

def test_rotation_interpolator():
    timestream = [(0.0, np.eye(3)), (1.0, np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))]
    interp = interpolator.RotationInterpolator(timestream)
    assert np.allclose(interp(0.0), np.eye(3))
    midway = interp(0.5)
    assert np.allclose(midway @ midway.T, np.eye(3))
    assert midway[0,0]<1.0 and midway[0,0]>0.0
    assert midway[0,1]<1.0 and midway[0,1]>0.0
    assert np.allclose(interp(1.0), np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

def test_smoothed_linear_interpolator():
    timestream = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (4.0, 0.0)]
    interp = interpolator.SmoothedLinearInterpolator(timestream, smoothing=0.5)
    assert np.allclose(interp(0.0), 0.18728488638447055)
    assert np.allclose(interp(0.5), 0.5833226799824336)
    assert np.allclose(interp(1.0), 1.3206482380039408)
    assert np.allclose(interp(1.5), 2.3157963036465143)
    assert np.allclose(interp(3.9), 0.5695905129936958)
    assert np.allclose(interp(4.0), 0.4616432412285651)
    assert interp(4.1) is interp.no_value
    # check smoothness
    assert abs(np.diff(np.diff([interp(x) for x in np.arange(0.0,4.0,0.05)]))).max()<0.02

def test_smoothed_rotation_interpolator():
    timestream = [(0.0, np.eye(3)), (1.0, np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))]
    interp = interpolator.SmoothedRotationInterpolator(timestream, smoothing=0.5)
    for x in np.arange(0.0,1.0,0.1):
        assert np.allclose(interp(x) @ interp(x).T, np.eye(3))

def test_smoothed_step_interpolator():
    timestream = [(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]
    interp = interpolator.SmoothedStepInterpolator(timestream, smoothing=0.5)
    assert interp(0.1) == 0.0
    assert interp(0.5) is interp.no_value
    assert interp(1.0) == 0.0
    assert interp(1.125) == 1.25
    assert interp(1.25) == 2.5
    assert interp(1.5) == 5.0
    assert interp(1.75) is interp.no_value
    assert interp(1.99) is interp.no_value
    assert interp(2.0) == 5.0
    assert interp(2.25) == 2.5
