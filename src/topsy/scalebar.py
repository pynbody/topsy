from __future__ import annotations

import numpy as np
import pynbody

from . import text
from . import overlay


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

class BarLengthRecommender:
    """Class to recommend a physical length for a scalebar, given the window width in kpc.
    
    The recommended length will be a "nice" number (1, 2, or 5 times a power of ten), along with a unit.

    The unit will be chosen from km, AU, pc, kpc, or Mpc depending on the length itself and the window width.
    
    """

    acceptable_units = "km", "au", "pc", "kpc", "Mpc"
    unit_conversion_to_kpc = np.array([
        pynbody.units.Unit(u).in_units("kpc") for u in acceptable_units
    ])

    def __init__(self, initial_window_width_kpc=1.0):
        self._window_width_kpc = initial_window_width_kpc
        self._update_recommendation()
        self._update_label()

    def _update_recommendation(self):
        magnitude_in_each_unit = abs(np.log10(self._window_width_kpc / self.unit_conversion_to_kpc) - 0.5)
        chosen_unit_index = np.argmin(magnitude_in_each_unit)
        chosen_unit = self.acceptable_units[chosen_unit_index]
        chosen_unit_conversion = self.unit_conversion_to_kpc[chosen_unit_index]
        target_scalebar_length_in_chosen_unit = (self._window_width_kpc / 2.0) / chosen_unit_conversion
        quantized_length_in_chosen_unit = self._quantize_length(target_scalebar_length_in_chosen_unit)
        self._physical_scalebar_length_in_chosen_unit = quantized_length_in_chosen_unit
        self._physical_scalebar_length_unit_name = chosen_unit
        self._physical_scalebar_length_kpc = quantized_length_in_chosen_unit * chosen_unit_conversion

    @classmethod
    def _quantize_length(cls, physical_scalebar_length):
        """Find a length less than or equal to physical_scalebar_length, that is 1, 2, or 5 times a power of ten."""
        power_of_ten = np.floor(np.log10(physical_scalebar_length))
        mantissa = physical_scalebar_length / 10 ** power_of_ten
        if mantissa < 2.0:
            physical_scalebar_length = 10.0 ** power_of_ten
        elif mantissa < 5.0:
            physical_scalebar_length = 2.0 * 10.0 ** power_of_ten
        else:
            physical_scalebar_length = 5.0 * 10.0 ** power_of_ten
        return physical_scalebar_length

    @classmethod
    def _format_scientific_latex(cls, value, unit):
        """Format a number in scientific notation with LaTeX rendering."""
        if value == 0:
            return f"0 {unit}"

        # Only use scientific notation for very small or very large numbers
        if 0.01 <= abs(value) <= 1000:
            if value == int(value):
                return f"{int(value)} {unit}"
            else:
                return f"{value:.2f}".rstrip('0').rstrip('.') + f" {unit}"

        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10 ** exponent)

        return f"${mantissa:.0f} \\times 10^{{{exponent}}}$ {unit}"

    def _update_label(self):
        self._label = self._format_scientific_latex(self._physical_scalebar_length_in_chosen_unit,
                                                   self._physical_scalebar_length_unit_name)
        self._label_is_for = (self._physical_scalebar_length_in_chosen_unit, self._physical_scalebar_length_unit_name)

    def update_window_width(self, window_width_kpc):
        """Update the window width in kpc, and recalculate the recommended scalebar length if it has changed."""
        if window_width_kpc != self._window_width_kpc:
            self._window_width_kpc = window_width_kpc
            self._update_recommendation()

    @property
    def label(self):
        """Get the label for the current recommended scalebar length."""
        if self._label_is_for != (self._physical_scalebar_length_in_chosen_unit,
                                 self._physical_scalebar_length_unit_name):
            self._update_label()
        return self._label

    @property
    def physical_scalebar_length_kpc(self):
        """Get the recommended physical scalebar length in kpc."""
        return self._physical_scalebar_length_kpc
    
    
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
        self._recommender = BarLengthRecommender(1.0) # will be updated immediately
        self._visualizer = visualizer

    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder, target_texture_view: wgpu.GPUTextureView):
        self._update_length()
        self._bar.length = self._physical_scalebar_length / self._visualizer.scale
        # note that the visualizer scale refers to a square rendering target
        # however only part of this is shown in the final window if the window
        # aspect ratio isn't 1:1. So we now need to correct for this effect.
        # The full x extent is shown if the width is greater than the height, so
        # no correction is needed then. If the height is greater than the width,
        # then the x extent is scaled by the ratio of the height to the width.

        if self._visualizer.canvas.width_physical < self._visualizer.canvas.height_physical:
            self._bar.length *= self._visualizer.canvas.height_physical / self._visualizer.canvas.width_physical

        self._label.encode_render_pass(command_encoder, target_texture_view)
        self._bar.encode_render_pass(command_encoder, target_texture_view)

    def _update_scalebar_label(self, physical_scalebar_length):
        if getattr(self, "_scalebar_label_is_for_length", None) != physical_scalebar_length:
            self._label.text = self._recommender.label
            self._scalebar_label_is_for_length = physical_scalebar_length
            self._label.update()

    def _update_length(self):
        # target is for the scalebar to be no more than 1/2 the viewport
        # (but not too much less either); however the length is to be 10^n or 5*10^n
        # in world coordinates the viewport is 2 * self.scale kpc wide
        # so the maximum scalebar length should be self.scale kpc
        window_width_kpc = 2.0 * self._visualizer.scale
        self._recommender.update_window_width(window_width_kpc)
        self._physical_scalebar_length = self._recommender.physical_scalebar_length_kpc
        self._update_scalebar_label(self._physical_scalebar_length)


