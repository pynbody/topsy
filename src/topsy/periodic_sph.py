from __future__ import annotations

from . import sph
from . import overlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visualizer import Visualizer

import numpy as np
import wgpu

class PeriodicSPHAccumulationOverlay(overlay.Overlay):
    _blending = (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.one,
                    wgpu.BlendOperation.add,
                 )
    def __init__(self, visualizer: Visualizer, source_texture: wgpu.GPUTexture, target_texture: wgpu.GPUTexture):
        self._texture = source_texture
        self.num_repetitions = 0
        self.panel_scale = 1.0
        self.rotation_matrix = np.eye(3)
        super().__init__(visualizer, target_texture)

    def _setup_texture(self):
        pass

    def get_clipspace_coordinates(self, width, height) -> tuple[float, float, float, float]:
        return -1.0, -1.0, 2.0, 2.0

    def get_instance_offsets_and_weights(self):
        offsets = []
        weights = []

        for xoff in range(-self.num_repetitions, self.num_repetitions + 1):
            for yoff in range(-self.num_repetitions, self.num_repetitions + 1):
                for zoff in range(-self.num_repetitions, self.num_repetitions + 1):
                    offset = self.rotation_matrix @ np.array([xoff,yoff,zoff], dtype=np.float32)
                    if abs(offset[2]) < 1.0:
                        offsets.append(offset[:2])
                        z = abs(offset[2])
                        # the weight be 1 for 0<z<0.5, and smoothly decrease to 0 over the range 0.5<z<1.0
                        if z>0.5:
                            weight = 1.0 - 2.0*(z-0.5)
                        else:
                            weight = 1.0
                        weights.append(weight)

        return np.array(offsets, dtype=np.float32) * self.panel_scale, np.array(weights, dtype=np.float32)

    def render_contents(self) -> np.ndarray:
        # must be implemented, but should never be called because texture is provided externally
        raise RuntimeError("SPHAccumulationOverlay.render_contents() should never be called")
class PeriodicSPH(sph.SPH):
    def __init__(self, visualizer, render_texture):
        self._final_render_texture = render_texture
        proxy_render_texture =  visualizer.device.create_texture(
                size=render_texture.size,
                format=render_texture.format,
                usage=render_texture.usage,
                label=f"proxy_sph"
            )
        self._accumulator = PeriodicSPHAccumulationOverlay(visualizer, proxy_render_texture, render_texture)
        super().__init__(visualizer, proxy_render_texture, wrapping=True)

    def encode_render_pass(self, command_encoder):
        super().encode_render_pass(command_encoder)
        self._accumulator.num_repetitions = 2
        self._accumulator.rotation_matrix = self.rotation_matrix
        self._accumulator.panel_scale = self._visualizer.periodicity_scale/self._visualizer.scale

        self._accumulator.encode_render_pass(command_encoder, True)

