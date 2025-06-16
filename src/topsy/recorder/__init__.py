from __future__ import annotations

import copy
import time
import tqdm
import wgpu
import numpy as np
import logging
import pickle

from . interpolator import (Interpolator, StepInterpolator, LinearInterpolator, RotationInterpolator,
                            SmoothedRotationInterpolator, SmoothedLinearInterpolator, SmoothedStepInterpolator)
from ..drawreason import DrawReason
from ..view_synchronizer import ViewSynchronizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..visualizer import Visualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class VisualizationRecorder:
    _record_properties = ['colormap[type]', 'quantity_name', 'colormap[log]', 'colormap[vmin]', 'colormap[vmax]', 'colormap[gamma]', # NB ordering is important to prevent triggering auto-scaling
                          'colormap[density_vmin]', 'colormap[density_vmax]', 'rotation_matrix', 'scale', 'position_offset']
    _record_interpolation_class_smoothed = [StepInterpolator, StepInterpolator, StepInterpolator, SmoothedStepInterpolator, SmoothedStepInterpolator,
                                            SmoothedStepInterpolator, SmoothedStepInterpolator, SmoothedStepInterpolator, SmoothedRotationInterpolator, SmoothedLinearInterpolator, SmoothedLinearInterpolator]
    _record_interpolation_class_unsmoothed = [StepInterpolator, StepInterpolator, StepInterpolator, StepInterpolator, StepInterpolator,
                                              StepInterpolator, StepInterpolator, StepInterpolator, RotationInterpolator, LinearInterpolator, LinearInterpolator]


    def __init__(self, visualizer: Visualizer):
        vs = ViewSynchronizer(synchronize=self._record_properties)
        vs.add_view(visualizer)
        vs.add_view(self, setter = VisualizationRecorder._add_event)
        self._recording = False
        self._playback = False
        self._recording_ends_at = None
        self._visualizer = visualizer
        self._reset_timestream()

    def _add_event(self, key, value):
        if key in self._record_properties:
            self._view_synchronizer.update_completed(self)  # this marks the update as done
            if self._recording:
                self._timestream[key].append((self._time_elapsed(), copy.copy(value)))

    def _time_elapsed(self):
        return time.time() - self._t0

    def _reset_timestream(self):
        self._timestream = {r: [(0.0, copy.copy(
            self._view_synchronizer._default_getter(self._visualizer, r)))] for r in self._record_properties}

    def record(self):
        self._t0 = time.time()
        self._reset_timestream()
        self._recording = True
        self._playback = False

    def stop(self):
        if self._recording:
            self._recording_ends_at = self._time_elapsed()
        self._recording = False
        self._playback = False


    def _get_value_at_time(self, property, time):
        return self._interpolators[property](time)

    def _progress_iterator(self, ntot):
        """Return an iterator that displays progress in an appropriate way

        Overriden for the qt gui"""
        return tqdm.tqdm(range(ntot), unit="frame")

    def _replay(self, fps=30.0, resolution=(1920, 1080), show_colorbar=True,
                show_scalebar=True, smooth=True, set_vmin_vmax=True,
                set_quantity=True):
        if self._recording:
            self.stop()
        if self._recording_ends_at is None:
            raise RuntimeError("Can't playback before recording")

        self._recording = False
        self._playback = True

        exclude = []

        if not set_vmin_vmax:
            exclude.extend(['vmin', 'vmax'])
        if not set_quantity:
            exclude.append('quantity_name')


        try:
            self._visualizer.show_colorbar = show_colorbar
            self._visualizer.show_scalebar = show_scalebar
            if smooth:
                self._interpolators = {r: c(self._timestream[r])
                                       for c, r in zip(self._record_interpolation_class_smoothed,
                                                       self._record_properties)
                                       if r not in exclude}
            else:
                self._interpolators = {r: c(self._timestream[r])
                                       for c, r in zip(self._record_interpolation_class_unsmoothed,
                                                       self._record_properties)
                                       if r not in exclude}

            device = self._visualizer.device

            render_texture: wgpu.GPUTexture = device.create_texture(
                size=(resolution[0], resolution[1], 1),
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                      wgpu.TextureUsage.COPY_SRC,
                format=self._visualizer.canvas_format,
                label="output_texture",
            )

            num_frames = int(self._recording_ends_at * fps)
            for i in self._progress_iterator(num_frames):
                t = i / fps
                for p in self._record_properties:
                    if p not in exclude:
                        val = self._get_value_at_time(p, t)
                        if val is not Interpolator.no_value:
                            self._view_synchronizer._default_setter(self._visualizer, p, val)

                self._visualizer.display_status("github.com/pynbody/topsy/", timeout=1e6)
                self._visualizer.draw(DrawReason.EXPORT, render_texture.create_view())
                im = device.queue.read_texture({'texture': render_texture, 'origin': (0, 0, 0)},
                                               {'bytes_per_row': 4 * resolution[0]},
                                               (resolution[0], resolution[1], 1))
                im_npy = np.frombuffer(im, dtype=np.uint8).reshape((resolution[1], resolution[0], 4))
                im_npy = im_npy[:, :,:3]
                yield im_npy

            self.playback = False
        finally:
            self._visualizer.show_colorbar = True
            self._visualizer.show_scalebar = True
            self._visualizer.display_status("Complete", timeout=1.0)

    def save_mp4(self, filename, fps, resolution, *args, **kwargs):
        import cv2
        writer = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*'mp4v'), fps,
                                 resolution)

        for image in self._replay(fps, resolution, *args, **kwargs):
            writer.write(image)

        writer.release()

    def save_timestream(self, fname):
        pickle.dump((self._timestream, self._recording_ends_at), open(fname, 'wb'))

    def load_timestream(self, fname):
        self._timestream, self._recording_ends_at = pickle.load(open(fname, 'rb'))


    @property
    def recording(self):
        return self._recording

