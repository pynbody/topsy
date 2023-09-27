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

def test_rotation_interpolator():
    timestream = [(0.0, np.eye(3)), (1.0, np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))]
    interp = interpolator.RotationInterpolator(timestream)
    assert np.allclose(interp(0.0), np.eye(3))
    midway = interp(0.5)
    assert np.allclose(midway @ midway.T, np.eye(3))
    assert midway[0,0]<1.0 and midway[0,0]>0.0
    assert midway[0,1]<1.0 and midway[0,1]>0.0
    assert np.allclose(interp(1.0), np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
