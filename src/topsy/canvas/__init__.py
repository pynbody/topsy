from __future__ import annotations

import numpy as np
import rendercanvas.jupyter, rendercanvas.auto
import time
import copy

from .. import config

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

        self.add_event_handler(self.event_handler, "*")

    def event_handler(self, event):
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
            self._visualizer.colormap_autorange()
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
        original_position = copy.copy(self._visualizer.position_offset)

        biggest_dimension = max(self.width_physical, self.height_physical)


        centre_physical_x = self.width_physical / (2*self.pixel_ratio)
        centre_physical_y = self.height_physical / (2*self.pixel_ratio)

        xy_displacement = 2. * self.pixel_ratio * np.array([centre_physical_x-x,
                                                            y-centre_physical_y,
                                                            0], dtype=np.float32) / biggest_dimension * self._visualizer.scale


        self._visualizer.position_offset += self._visualizer.rotation_matrix.T @ xy_displacement


        depth_im = self._visualizer.get_depth_image()
        central_depth = depth_im[depth_im.shape[0]//2, depth_im.shape[1]//2]

        if ~np.isnan(central_depth):
            z_displacement = np.array([0, 0, -central_depth], dtype=np.float32)
            self._visualizer.position_offset += self._visualizer.rotation_matrix.T @ z_displacement

        final_position = self._visualizer.position_offset

        # the actual work is done - now animate it so it looks understandable
        self._visualizer.position_offset = original_position

        #def interpolate_position(t):
        #    return original_position + (final_position - original_position) * t

        def interpolate_position(t):
            w1 = np.arctan(5*(t*2-1))/np.pi+0.5
            w2 = 1-w1
            return w2 * original_position + w1 * final_position

        start = time.time()

        def glide():
            t = (time.time()-start)/config.GLIDE_TIME
            if t>1:
                self._visualizer.position_offset = final_position
            else:
                self.call_later(0.0, glide)
                self._visualizer.position_offset = interpolate_position(t)



        self.call_later(1. / config.TARGET_FPS, glide)


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


from .. import is_jupyter

if is_jupyter():
    from .jupyter import VisualizerCanvas
else:
    from .qt import VisualizerCanvas

