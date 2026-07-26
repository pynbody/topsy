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

def quantize_to_1_2_5(value, mode="nearest"):
    """Snap a positive value to a "nice" number of the form {1, 2, 5} x 10^N.

    Such numbers make natural reference lengths for scalebars and vector keys.

    :param value: the positive value to quantize
    :param mode: how to choose between the candidate nice numbers:
        - "nearest": the nice number closest to ``value`` in log space (i.e. by
          ratio), so that e.g. 3.1 -> 2 but 3.2 -> 5.
        - "floor": the largest nice number that is <= ``value``.
    :return: the chosen nice number, as a float
    """
    if value <= 0.0:
        raise ValueError("quantize_to_1_2_5 requires a strictly positive value")

    power_of_ten = np.floor(np.log10(value))
    # candidates span this decade plus the bottom of the next, so that values
    # sitting just below a power of ten can round up to it
    candidates = np.array([1.0, 2.0, 5.0, 10.0]) * 10.0 ** power_of_ten

    if mode == "floor":
        # tiny tolerance so that value == candidate isn't excluded by rounding
        eligible = candidates[candidates <= value * (1.0 + 1e-9)]
        return float(eligible.max())
    elif mode == "nearest":
        closest = np.argmin(np.abs(np.log10(candidates) - np.log10(value)))
        return float(candidates[closest])
    else:
        raise ValueError(f"Unknown quantization mode {mode!r}")

def format_scientific_latex(value, unit=None):
    """Format a number in scientific notation with LaTeX rendering."""
    if value == 0:
        result = "0"
    # Only use scientific notation for very small or very large numbers
    elif 0.01 <= abs(value) <= 1000:
        if value == int(value):
            result = f"{int(value)}"
        else:
            result = f"{value:.2f}".rstrip('0').rstrip('.')
    else:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exponent)
        result = f"${mantissa:.0f} \\times 10^{{{exponent}}}$"

    if unit is None:
        return result
    else:
        return f"{result} {unit}"

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
