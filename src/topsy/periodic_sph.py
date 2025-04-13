from __future__ import annotations

from topsy.drawreason import DrawReason

from . import sph
from . import overlay

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visualizer import Visualizer

import numpy as np
import wgpu

class PeriodicSPHAccumulationOverlay(overlay.Overlay):
    _blending = {
        "src_factor": wgpu.BlendFactor.one,
        "dst_factor": wgpu.BlendFactor.one,
        "operation": wgpu.BlendOperation.add
    }

    def __init__(self, visualizer: Visualizer, source_texture: wgpu.GPUTexture):
        self._texture = source_texture
        self.num_repetitions = 0
        self.panel_scale = 1.0
        self.rotation_matrix = np.eye(3)
        super().__init__(visualizer, source_texture.format)

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
    def __init__(self, visualizer, render_size):
        super().__init__(visualizer, render_size, wrapping=True)
        self._periodic_texture =  visualizer.device.create_texture(
                size=self._render_texture.size,
                format=self._render_texture.format,
                usage=self._render_texture.usage,
                label=f"proxy_sph"
            )
        self._accumulator = PeriodicSPHAccumulationOverlay(visualizer, self._render_texture)

    def get_output_texture(self) -> wgpu.Texture:
        return self._periodic_texture

    def render(self, draw_reason=DrawReason.CHANGE):
        if draw_reason == DrawReason.PRESENTATION_CHANGE:
            return

        super().render(draw_reason)

        command_encoder = self._visualizer.device.create_command_encoder(label="PeriodicSPH")
        self._accumulator.num_repetitions = 2
        self._accumulator.rotation_matrix = self.rotation_matrix
        self._accumulator.panel_scale = self._visualizer.periodicity_scale / self._visualizer.scale

        self._accumulator.encode_render_pass(command_encoder, self._periodic_texture.create_view(), True)

        encoded_render_pass = command_encoder.finish()
        self._device.queue.submit([encoded_render_pass])

