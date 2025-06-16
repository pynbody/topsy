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
    def __init__(self, timestream, smoothing=0.25, fps=30):
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
        interpolated_timestream = np.concatenate(
            ([interpolated_timestream[0]]*(len(kernel)//2),
             interpolated_timestream,
             [interpolated_timestream[-1]]*(len(kernel)//2))
        )

        if len(interpolated_timestream.shape)==1:
            smoothed_timestream = np.convolve(interpolated_timestream, kernel, mode='valid')
        else:
            smoothed_timestream = None
            for index in np.ndindex(interpolated_timestream.shape[1:]):
                index_c = (slice(None),)+index # py3.11+ supports [:, *index] but not py3.10-
                result = np.convolve(interpolated_timestream[index_c], kernel, mode='valid')
                if smoothed_timestream is None:
                    smoothed_timestream = np.empty((len(result),)+interpolated_timestream.shape[1:])
                smoothed_timestream[index_c] = result

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
        self._last_t = None

    def __call__(self, t):
        if self._last_t is not None and t<self._last_t:
            raise ValueError("StepInterpolator must be accessed sequentially")

        self._last_t = t

        stream = self._timestream
        for (t_ev, val_ev) in stream[::-1]:
            if t_ev <= t:
                if val_ev != self._last_value:
                    self._last_value = val_ev
                    return self._last_value
                else:
                    return self.no_value
        return self.no_value

class SmoothedStepInterpolator(StepInterpolator):
    def __init__(self, timestream, smoothing=0.25):
        self._start_value = None
        self._target_value = None
        self._transition_start = None
        self._transition_end = None
        self._smoothing = smoothing
        super().__init__(timestream)

    def __call__(self, t):
        if self._target_value is not None:
            if t>=self._transition_end:
                tv = self._target_value
                self._start_value = None
                self._target_value = None
                self._transition_start = None
                self._transition_end = None
                return tv
            else:
                return self._start_value + (self._target_value-self._start_value)*(t-self._transition_start)/(self._transition_end-self._transition_start)
        else:
            last_value = self._last_value
            new_value = super().__call__(t)
            if new_value is self.no_value or new_value is None or new_value == last_value:
                return self.no_value
            elif last_value is self.no_value or last_value is None:
                return new_value
            else:
                self._start_value = last_value
                self._target_value = new_value
                self._transition_start = t
                self._transition_end = t + self._smoothing
                return last_value

