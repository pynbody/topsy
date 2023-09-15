from __future__ import annotations

import numpy as np
import wgpu.gui.jupyter

from wgpu.gui.qt import WgpuCanvas
from .drawreason import DrawReason

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

class VisualizerCanvas(WgpuCanvas):
    def __init__(self, *args, **kwargs):
        self._visualizer : Visualizer = kwargs.pop("visualizer")

        self._last_x = 0
        self._last_y = 0
        # The below are dummy values that will be updated by the initial resize event
        self.width_physical, self.height_physical = 640,480
        self.pixel_ratio = 1

        super().__init__(*args, **kwargs)

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
        super().handle_event(event)

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
        if isinstance(self, wgpu.gui.jupyter.JupyterWgpuCanvas):
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
        if width==0.0 or height==0.0:
            return
            # qt seems to make a call with zero, which then leads to textures being initialized
            # with zero size if we take it seriously

        self.width_physical = width*pixel_ratio
        self.height_physical = height*pixel_ratio
        self.pixel_ratio = pixel_ratio

    def request_draw(self, function, *args, **kwargs):

        # As a side effect, wgpu gui layer stores our function call, to enable it to be
        # repainted later. But we want to distinguish such repaints and handle them
        # differently, so we need to replace the function with our own
        def function_wrapper():
            function()
            self._subwidget.draw_frame = lambda: self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)

        super().request_draw(function_wrapper,*args, **kwargs)



