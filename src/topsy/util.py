import numpy as np
import time
import wgpu
import re
import os

def load_shader(name):
    from importlib import resources
    with open(resources.files("topsy.shaders") / name, "r") as f:
        return f.read()

def preprocess_shader(shader_code, active_flags):
    """A hacky preprocessor for WGSL shaders.

    Any line in shader_code containing [[FLAG]] will be removed if FLAG is not in active_flags.
    Otherwise, the string [[FLAG]] will be removed, leaving just valid syntax.

    This is needed because we can't use
    const values in the shader yet, so we need to use something like #ifdefs instead.
    In final version of webgpu doesn't look like this will be needed"""
    for f in active_flags:
        shader_code = re.sub(f"^.*\[\[{f}]](.*)$", r"\1", shader_code, flags=re.MULTILINE)
    shader_code = re.sub(r"^.*\[\[[A-Z_]+]].*$", "", shader_code, flags=re.MULTILINE)

    # process #ifdef / #else / #endif
    lines = shader_code.splitlines()
    output_lines: list[str] = []
    include_stack: list[tuple[bool, bool]] = []
    current_include = True

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('#ifdef'):
            _, flag = stripped.split(None, 1)
            should_include = current_include and (flag in active_flags)
            include_stack.append((current_include, should_include))
            current_include = should_include

        elif stripped.startswith('#else'):
            parent_include, prev_include = include_stack[-1]
            new_include = parent_include and not prev_include
            include_stack[-1] = (parent_include, new_include)
            current_include = new_include

        elif stripped.startswith('#endif'):
            parent_include, _ = include_stack.pop()
            current_include = parent_include

        elif current_include:
            output_lines.append(line)

    result = "\n".join(output_lines)

    return result

def is_inside_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def is_inside_jupyter_notebook():
    return "JPY_SESSION_NAME" in os.environ

def is_ipython_running_qt_event_loop():
    if not is_inside_ipython():
        return False
    import IPython.lib.guisupport
    return IPython.lib.guisupport.is_event_loop_running_qt4()

def determine_backend():
    if is_inside_ipython():
        pass

class TimeGpuOperation:
    """Context manager for timing GPU operations"""
    def __init__(self, device, n_frames_smooth=10):
        self.device = device
        self.n_frames_smooth = n_frames_smooth
        self._recent_times = []
        self._current_frame_duration = 0.0

    def __enter__(self):
        self.device.queue.on_submitted_work_done_sync()
        self.__block_start = time.time()
        return self

    def __exit__(self, *args):
        # Now, we want to measure how much time the render has taken so that we can adapt
        # for the next frame if needed. However, the GPU is asynchronous. In the long term
        # there should be facilities like callbacks or querysets to help with this, but
        # right now these don't seem to be implemented. So we need to make something block
        # until the current queue is complete. The hack here is to do a trivial read
        # operation

        self.device.queue.on_submitted_work_done_sync()
        block_end = time.time()
        self._current_frame_duration += block_end - self.__block_start

    def end_frame(self):
        self.last_duration = self._current_frame_duration
        self._current_frame_duration = 0.0

        self._recent_times.append(self.last_duration)
        if len(self._recent_times) > self.n_frames_smooth:
            self._recent_times.pop(0)

    def total_time_in_frame(self):
        """Return the time elapsed in GPU operations since the last call to end_frame"""
        return self._current_frame_duration

    @property
    def running_mean_duration(self):
        return np.mean(self._recent_times)
