from __future__ import annotations

import numpy as np
import wgpu
from . import sph
from . import overlay

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
        self._downsample_factors = [1, 8]
        self._resolutions = [self._resolution_final//factor for factor in self._downsample_factors]
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


        for i in range(len(self._renderers)):
            r = self._renderers[i]
            if i==0:
                r.min_pixels = 0.0
            else:
                r.min_pixels = max_pixels / self._downsample_factors[i]

            if i==len(self._renderers)-1:
                r.max_pixels = np.inf
            else:
                r.max_pixels = max_pixels

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


