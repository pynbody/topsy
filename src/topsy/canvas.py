from __future__ import annotations

import numpy as np

from wgpu.gui.auto import WgpuCanvas

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer_wgpu import Visualizer

class VisualizerCanvas(WgpuCanvas):
    def __init__(self, *args, **kwargs):
        self._visualizer : Visualizer = kwargs.pop("visualizer")
        super().__init__(*args, **kwargs)
        self._last_x = 0
        self._last_y = 0

    def handle_event(self, event):
        if event['event_type']=='pointer_move':
            if len(event['buttons'])>0:
                self.drag(event['x']-self._last_x, event['y']-self._last_y)
            self._last_x = event['x']
            self._last_y = event['y']
        elif event['event_type']=='wheel':
            self.mouse_wheel(event['dx'], event['dy'])
        elif event['event_type']=='key_up':
            self.key_up(event['key'])
        elif event['event_type']=='resize':
            self.resize(event['width'], event['height'], event['pixel_ratio'])
        else:
            pass
            # print(event)

    def drag(self, dx, dy):
        self._visualizer.rotate(dx, dy)

    def key_up(self, key):
        if key=='s':
            self._visualizer.save()
        elif key=='r':
            self._visualizer.vmin_vmax_is_set = False
            self._visualizer.invalidate()

    def mouse_wheel(self, delta_x, delta_y):
        self._visualizer.scale*=np.exp(delta_y/1000)

    def resize(self, width, height, pixel_ratio):
        pass