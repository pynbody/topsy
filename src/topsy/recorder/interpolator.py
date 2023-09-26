"""Interpolation between frames for the motion recorder"""

import numpy as np

from abc import ABC, abstractmethod

class Interpolator(ABC):
    """ABC for interpolating a timestream.

    The timestream is a list of (time, value) pairs, where time is a float and value is any type."""

    def __init__(self, timestream):
        self._timestream = timestream

    @abstractmethod
    def __call__(self, t):
        pass


class LinearInterpolator(Interpolator):
    """Returns the linearly interpolated value, or None if no value is available"""

    def __call__(self, t):
        stream = self._timestream
        for i, (t_ev, val_ev) in enumerate(stream):
            if t_ev >= t:
                if i == 0:
                    return val_ev
                else:
                    t0, val0 = stream[i - 1]
                    assert t0 < t
                    return val0 + (val_ev - val0) * (t - t0) / (t_ev - t0)
        return None


class RotationInterpolator(LinearInterpolator):
    """Returns an interpolated rotation matrix"""

    def __call__(self, t):
        matr = super().__call__(t)
        if matr is None:
            return None

        # orthogonalise matr:
        u, s, vh = np.linalg.svd(matr)
        return u @ vh


class StepInterpolator(Interpolator):
    """Only returns a value when it has changed. Assumes it is being acccessed sequentially"""

    def __init__(self, timestream):
        super().__init__(timestream)
        self._last_value = None

    def __call__(self, t):
        stream = self._timestream
        for (t_ev, val_ev) in stream[::-1]:
            if t_ev <= t:
                if val_ev != self._last_value:
                    self._last_value = val_ev
                    return self._last_value
                else:
                    return None
        return None
