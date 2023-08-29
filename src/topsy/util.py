import numpy as np
import time
import wgpu


def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()

class TimeGpuOperation:
    def __init__(self, device, n_frames_smooth=10):
        self.device = device
        self.n_frames_smooth = n_frames_smooth
        self._recent_times = []
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        # Now, we want to measure how much time the render has taken so that we can adapt
        # for the next frame if needed. However, the GPU is asynchronous. In the long term
        # there should be facilities like callbacks or querysets to help with this, but
        # right now these don't seem to be implemented. So we need to make something block
        # until the current queue is complete. The hack here is to do a trivial read
        # operation
        dummy_buffer = self.device.create_buffer(
            size=16,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        self.device.queue.read_buffer(dummy_buffer, 0)
        end = time.time()

        self.end = time.time()
        self.last_duration = self.end - self.start
        self._recent_times.append(self.last_duration)
        if len(self._recent_times) > self.n_frames_smooth:
            self._recent_times.pop(0)
    @property
    def running_mean_duration(self):
        return np.mean(self._recent_times)
