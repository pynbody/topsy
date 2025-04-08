from __future__ import annotations

import numpy as np
import rendercanvas.jupyter, rendercanvas.auto

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..visualizer import Visualizer



class VisualizerCanvasBase:
    def __init__(self, *args, **kwargs):
        self._visualizer : Visualizer = kwargs.pop("visualizer")

        self._last_x = 0
        self._last_y = 0
        # The below are dummy values that will be updated by the initial resize event
        self.width_physical, self.height_physical = 640,480
        self.pixel_ratio = 1

        super().__init__(*args, **kwargs)

        self.add_event_handler(self.handle_event, "*")

    def handle_event(self, event):
        if event['event_type']=='pointer_move':
            if len(event['buttons'])>0:
                if len(event['modifiers'])==0:
                    self.drag(event['x']-self._last_x, event['y']-self._last_y)
                else:
                    self.shift_drag(event['x']-self._last_x, event['y']-self._last_y)
            self._last_x = event['x']
            self._last_y = event['y']
        elif event['event_type']=='wheel':
            self.mouse_wheel(event['dx'], event['dy'])
        elif event['event_type']=='key_up':
            self.key_up(event['key'])
        elif event['event_type']=='resize':
            self.resize_complete(event['width'], event['height'], event['pixel_ratio'])
        elif event['event_type']=='double_click':
            self.double_click(event['x'], event['y'])
        elif event['event_type']=='pointer_up':
            self.release_drag()
        else:
            pass


    def drag(self, dx, dy):
        self._visualizer.rotate(dx*0.01, dy*0.01)

    def shift_drag(self, dx, dy):
        biggest_dimension = max(self.width_physical, self.height_physical)

        displacement = 2.*self.pixel_ratio*np.array([dx, -dy, 0], dtype=np.float32) / biggest_dimension * self._visualizer.scale
        self._visualizer.position_offset += self._visualizer.rotation_matrix.T @ displacement

        self._visualizer.display_status("centre = [{:.2f}, {:.2f}, {:.2f}]".format(*self._visualizer._sph.position_offset))

        self._visualizer.crosshairs_visible = True


    def key_up(self, key):
        if key=='s':
            self._visualizer.save()
        elif key=='r':
            self._visualizer.vmin_vmax_is_set = False
            self._visualizer.invalidate()
        elif key=='h':
            self._visualizer.reset_view()

    def mouse_wheel(self, delta_x, delta_y):
        if isinstance(self, rendercanvas.jupyter.JupyterRenderCanvas):
            # scroll events are much smaller from the web browser, for
            # some reason, compared with native windowing
            delta_y *= 10
            delta_x *= 10

        self._visualizer.scale*=np.exp(delta_y/1000)

    def release_drag(self):
        if self._visualizer.crosshairs_visible:
            self._visualizer.crosshairs_visible = False
            self._visualizer.invalidate()


    def resize(self, *args):
        # putting this here as a reminder that the resize method must be passed to the base class
        super().resize(*args)

    def resize_complete(self, width, height, pixel_ratio=1):
        self.width_physical = int(width*pixel_ratio)
        self.height_physical = int(height*pixel_ratio)
        self.pixel_ratio = pixel_ratio

    def double_click(self, x, y):
        pass

    @classmethod
    def call_later(cls, delay, fn, *args):
        raise NotImplementedError()





# Now we are going to select a specific backend
#
# we don't use rendercanvas.auto directly because it prefers the glfw backend over qt
# whereas we want to use qt
#
# Note also that is_jupyter as implemented fails to distinguish correctly if we are
# running inside a kernel that isn't attached to a notebook. There doesn't seem to
# be any way to distinguish this, so we live with it for now.


def is_jupyter():
    """Determine whether the user is executing in a Jupyter Notebook / Lab.

    This has been pasted from an old version of wgpu.gui.auto.is_jupyter; the function was removed"""
    from IPython import get_ipython
    try:
        ip = get_ipython()
        if ip is None:
            return False
        if ip.has_trait("kernel"):
            return True
        else:
            return False
    except NameError:
        return False


if is_jupyter():
    from .jupyter import VisualizerCanvas
else:
    from .qt import VisualizerCanvas

