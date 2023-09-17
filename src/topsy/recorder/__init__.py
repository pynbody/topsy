from __future__ import annotations

import time
import tqdm
import wgpu
import numpy as np

from . interpolator import StepInterpolator, LinearInterpolator, RotationInterpolator
from ..drawreason import DrawReason
from ..view_synchronizer import ViewSynchronizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..visualizer import Visualizer


class VisualizationRecorder:
    _record_properties = ['vmin', 'vmax', 'rotation_matrix', 'scale', 'position_offset']
    _record_interpolation_class = [StepInterpolator, StepInterpolator, RotationInterpolator, LinearInterpolator,
                                   LinearInterpolator]

    def __init__(self, visualizer: Visualizer):
        vs = ViewSynchronizer(synchronize=self._record_properties)
        vs.add_view(visualizer)
        vs.add_view(self)
        self._recording = False
        self._playback = False
        self._recording_ends_at = None
        self._visualizer = visualizer
        self._reset_timestream()

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in self._record_properties:
            self._view_synchronizer.update_completed(self)  # this marks the update as done
            if self._recording:
                self._timestream[key].append((self._time_elapsed(), value))

    def _time_elapsed(self):
        return time.time() - self._t0

    def _reset_timestream(self):
        self._timestream = {r: [(0.0, getattr(self._visualizer, r))] for r in self._record_properties}

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
        self._interpolators = {r: c(self._timestream[r])
                               for c, r in zip(self._record_interpolation_class, self._record_properties)}

    def _get_value_at_time(self, property, time):
        return self._interpolators[property](time)

    def _replay(self, fps=30.0, resolution=(1920, 1080)):
        if self._recording:
            self.stop()
        if self._recording_ends_at is None:
            raise RuntimeError("Can't playback before recording")
        self._playback = True
        self._recording = False

        device = self._visualizer.device

        render_texture: wgpu.GPUTexture = device.create_texture(
            size=(resolution[0], resolution[1], 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.COPY_SRC,
            format=self._visualizer.canvas_format,
            label="output_texture",
        )

        num_frames = int(self._recording_ends_at * fps)
        for i in tqdm.tqdm(range(num_frames), unit="frame"):
            t = i / fps
            for p in self._record_properties:
                val = self._get_value_at_time(p, t)
                if val is not None:
                    setattr(self._visualizer, p, val)
            self._visualizer.display_status("github.com/pynbody/topsy/")
            self._visualizer.draw(DrawReason.REFINE, render_texture.create_view())
            im = device.queue.read_texture({'texture': render_texture, 'origin': (0, 0, 0)},
                                           {'bytes_per_row': 4 * resolution[0]},
                                           (resolution[0], resolution[1], 1))
            im_npy = np.frombuffer(im, dtype=np.uint8).reshape((resolution[1], resolution[0], 4))
            im_npy = im_npy[:, :, 2::-1]
            e = time.time()
            yield im_npy
            self._visualizer.display_status(f"Rendering video frame {i+1} of {num_frames}")
            self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)
            self._visualizer.context.present()

        self.playback = False

    def save_mp4(self, filename, fps=30.0, resolution=(1920, 1080)):
        import cv2
        writer = cv2.VideoWriter(filename, cv2.VideoWriter.fourcc(*'mp4v'), fps,
                                 resolution)

        for image in self._replay(fps, resolution):
            writer.write(image)

        writer.release()

