"""Interpolation between frames for the motion recorder"""

import math
import numpy as np

from abc import ABC, abstractmethod

class Interpolator(ABC):
    """ABC for interpolating a timestream.

    The timestream is a list of (time, value) pairs, where time is a float and value is any type."""

    no_value = object()

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
        return self.no_value

class SmoothedInterpolatorMixin:
    def __init__(self, timestream, smoothing=1.0, fps=30):
        """Create a linear interpolator with gaussian smoothing over the specified period

        Args:
            timestream: the timestream to interpolate
            smoothing: the standard deviation of the gaussian smoothing kernel, in seconds
            fps: the number of samples per second in the smoothed timestream (doesn't have to match video fps)
        """
        super().__init__(timestream)
        tmax = timestream[-1][0]
        self._smoothing = smoothing

        interpolated_timestream = []
        for i in range(math.floor(tmax*fps)):
            interpolated_timestream.append(super().__call__(i/fps))


        kernel = np.exp(-np.arange(-3*smoothing*fps, 3*smoothing*fps)**2/(2*smoothing**2*fps**2))
        kernel/=kernel.sum()
        interpolated_timestream = np.vstack(
            ([interpolated_timestream[0]]*(len(kernel)//2),
             interpolated_timestream,
             [interpolated_timestream[-1]]*(len(kernel)//2))
        )

        if len(interpolated_timestream.shape)==1:
            smoothed_timestream = np.convolve(interpolated_timestream, kernel, mode='valid')
        else:
            smoothed_timestream = None
            for index in np.ndindex(interpolated_timestream.shape[1:]):
                result = np.convolve(interpolated_timestream[:,*index], kernel, mode='valid')
                if smoothed_timestream is None:
                    smoothed_timestream = np.empty((len(result),)+interpolated_timestream.shape[1:])
                smoothed_timestream[:,*index] = result

        print(len(smoothed_timestream), len(interpolated_timestream))
        self._timestream = [(i/fps, val) for i, val in enumerate(smoothed_timestream)]


class SmoothedLinearInterpolator(SmoothedInterpolatorMixin, LinearInterpolator):
    pass



class RotationInterpolator(LinearInterpolator):
    """Returns an interpolated rotation matrix"""

    def __call__(self, t):
        matr = super().__call__(t)
        if matr is self.no_value:
            return matr

        # orthogonalise matr:
        u, s, vh = np.linalg.svd(matr)
        return u @ vh

class SmoothedRotationInterpolator(SmoothedInterpolatorMixin, RotationInterpolator):
    pass


class StepInterpolator(Interpolator):
    """Only returns a value when it has changed. Assumes it is being acccessed sequentially"""

    def __init__(self, timestream):
        super().__init__(timestream)
        self._last_value = self.no_value

    def __call__(self, t):
        stream = self._timestream
        for (t_ev, val_ev) in stream[::-1]:
            if t_ev <= t:
                if val_ev != self._last_value:
                    self._last_value = val_ev
                    return self._last_value
                else:
                    return self.no_value
        return self.no_value

