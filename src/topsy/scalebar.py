from __future__ import annotations

import numpy as np

from . import text
from . import overlay


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

class BarOverlay(overlay.Overlay):
    """Overlay that implements a bar."""

    def __init__(self, *args, x0=0.1, y0=0.1, height_pixels=20, color=(1, 1, 1, 1), initial_length=0.2, **kwargs):
        self.x0 = x0
        self.y0 = y0
        self.height_pixels = height_pixels
        self.color = color
        self.length = initial_length # update just by directly setting self.length in clipspace coords - simple :-)

        super().__init__(*args, **kwargs)

    def render_contents(self) -> np.ndarray:
        # just a single pixel of the right color
        pixel = np.ones((1, 1, 4), dtype=np.float32)
        pixel[0,0,:] = self.color
        return pixel

    def get_clipspace_coordinates(self, window_pixel_width, window_pixel_height):
        height_clipspace = 2.0 * self.height_pixels / window_pixel_height
        return self.x0, self.y0, self.length, height_clipspace


class ScalebarOverlay:
    def __init__(self, visualizer: Visualizer):
        self._label = text.TextOverlay(visualizer, "Scalebar", (-0.9, -0.85), 40, color=(1, 1, 1, 1))
        self._bar = BarOverlay(visualizer, x0=-0.9, y0=-0.9, height_pixels=10, color=(1, 1, 1, 1))
        self._visualizer = visualizer

    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder, target_texture_view: wgpu.GPUTextureView):
        physical_scalebar_length = self._recommend_physical_scalebar_length()
        self._bar.length = physical_scalebar_length / self._visualizer.scale
        # note that the visualizer scale refers to a square rendering target
        # however only part of this is shown in the final window if the window
        # aspect ratio isn't 1:1. So we now need to correct for this effect.
        # The full x extent is shown if the width is greater than the height, so
        # no correction is needed then. If the height is greater than the width,
        # then the x extent is scaled by the ratio of the height to the width.

        if self._visualizer.canvas.width_physical < self._visualizer.canvas.height_physical:
            self._bar.length *= self._visualizer.canvas.height_physical / self._visualizer.canvas.width_physical

        self._update_scalebar_label(physical_scalebar_length)

        self._label.encode_render_pass(command_encoder, target_texture_view)
        self._bar.encode_render_pass(command_encoder, target_texture_view)

    def _get_scalebar_label_text(self, physical_scalebar_length_kpc):
        if physical_scalebar_length_kpc < 1:
            return f"{physical_scalebar_length_kpc * 1000:.0f} pc"
        if physical_scalebar_length_kpc < 1000:
            return f"{physical_scalebar_length_kpc:.0f} kpc"
        else:
            return f"{physical_scalebar_length_kpc / 1000:.0f} Mpc"

    def _update_scalebar_label(self, physical_scalebar_length):
        if getattr(self, "_scalebar_label_is_for_length", None) != physical_scalebar_length:
            self._label.text = self._get_scalebar_label_text(physical_scalebar_length)
            self._scalebar_label_is_for_length = physical_scalebar_length
            self._label.update()

    def _recommend_physical_scalebar_length(self):
        # target is for the scalebar to be no more than 1/2 the viewport
        # (but not too much less either); however the length is to be 10^n or 5*10^n
        # in world coordinates the viewport is 2 * self.scale kpc wide
        # so the maximum scalebar length should be self.scale kpc

        physical_scalebar_length =  self._visualizer.scale
        # now quantize it:
        power_of_ten = np.floor(np.log10(physical_scalebar_length))
        mantissa = physical_scalebar_length / 10 ** power_of_ten
        if mantissa < 2.0:
            physical_scalebar_length = 10.0 ** power_of_ten
        elif mantissa < 5.0:
            physical_scalebar_length = 2.0 * 10.0 ** power_of_ten
        else:
            physical_scalebar_length = 5.0 * 10.0 ** power_of_ten
        return physical_scalebar_length

