from __future__ import annotations

import numpy as np
import wgpu
from . import sph
from . import overlay
from logging import getLogger

logger = getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer_wgpu import Visualizer

class SPHAccumulationOverlay(overlay.Overlay):
    _blending = (
                    wgpu.BlendFactor.one,
                    wgpu.BlendFactor.one,
                    wgpu.BlendOperation.add,
                 )
    def __init__(self, visualizer: Visualizer, source_texture: wgpu.GPUTexture, target_texture: wgpu.GPUTexture):
        self._texture = source_texture
        super().__init__(visualizer, target_texture)


    def _setup_texture(self):
        pass

    def get_clipspace_coordinates(self, width, height) -> tuple[float, float, float, float]:
        return -1.0, -1.0, 2.0, 2.0

    def render_contents(self) -> np.ndarray:
        # must be implemented, but should never be called because texture is provided externally
        raise RuntimeError("SPHAccumulationOverlay.render_contents() should never be called")

class MultiresolutionSPH:
    """A drop-in replacement for the SPH class, which renders to multiple resolutions and then combines them."""

    def __init__(self, visualizer: Visualizer, render_texture: wgpu.GPUTexture, max_pixels=10.0):
        self._resolution_final = render_texture.width
        assert render_texture.width == render_texture.height
        self._pixel_scaling_factors = [1]
        self._resolutions = [self._resolution_final // factor for factor in self._pixel_scaling_factors]
        self._textures = [render_texture]
        for i,r in enumerate(self._resolutions[1:],1):
            self._textures.append(visualizer.device.create_texture(
                size=(r, r, 1),
                format=self._textures[0].format,
                usage=self._textures[0].usage,
                label=f"multires_sph_level_{i}"
            ))

        self._renderers: list[sph.SPH] = [sph.SPH(visualizer, texture) for texture in self._textures]

        self._accumulators = [SPHAccumulationOverlay(visualizer, self._textures[i], self._textures[0])
                              for i in range(len(self._textures)-1,0,-1)]

        layer_counts = {}
        for ds in self._pixel_scaling_factors:
            layer_counts[ds] = layer_counts.get(ds, 0) + 1

        logger.info(f"Multi-resolution SPH initialized: {len(self._renderers)} layers, with pixel downsample factors as follows:")

        current_offsets = {ds: 0 for ds in layer_counts.keys()}

        if self._pixel_scaling_factors[0]!=1:
            raise RuntimeError("Factor 1 must be first entry in the pixel scaling factors list")

        for i in range(len(self._renderers)):
            r = self._renderers[i]
            downsamp_fac = self._pixel_scaling_factors[i]
            if downsamp_fac==1:
                r.min_pixels = 0.0
            else:
                next_highest_downsamp_fac = max([ds for ds in self._pixel_scaling_factors if ds < downsamp_fac])
                r.min_pixels = max_pixels * next_highest_downsamp_fac / self._pixel_scaling_factors[i]

            if downsamp_fac==max(self._pixel_scaling_factors):
                r.max_pixels = np.inf
            else:
                r.max_pixels = max_pixels

            r.downsample_offset = current_offsets[self._pixel_scaling_factors[i]]
            current_offsets[self._pixel_scaling_factors[i]] += 1
            r.downsample_factor = layer_counts[self._pixel_scaling_factors[i]]
            logger.info(f"  Layer {i}: pixel_width={self._resolutions[i]} min_pixels={r.min_pixels}, max_pixels={r.max_pixels}, downsample_factor={r.downsample_factor}, downsample_offset={r.downsample_offset}")

        self._visualizer = visualizer

    @property
    def scale(self):
        return self._renderers[0].scale

    @scale.setter
    def scale(self, value):
        for s in self._renderers:
            s.scale = value

    @property
    def rotation_matrix(self):
        return self._renderers[0].rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        for s in self._renderers:
            s.rotation_matrix = value

    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder):
        for s in self._renderers:
            s.encode_render_pass(command_encoder)

        for a in self._accumulators:
            a.encode_render_pass(command_encoder)


