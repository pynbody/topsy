from . import sph
from .multiresolution_sph import SPHAccumulationOverlay
class PeriodicSPH(sph.SPH):
    def __init__(self, visualizer, render_texture):
        self._final_render_texture = render_texture
        proxy_render_texture =  visualizer.device.create_texture(
                size=render_texture.size,
                format=render_texture.format,
                usage=render_texture.usage,
                label=f"proxy_sph"
            )
        self._accumulator = SPHAccumulationOverlay(visualizer, proxy_render_texture, render_texture)
        super().__init__(visualizer, proxy_render_texture, wrapping=True)

    def encode_render_pass(self, command_encoder):
        super().encode_render_pass(command_encoder)
        self._accumulator.num_repetitions = 2
        self._accumulator.panel_scale = self._visualizer.periodicity_scale/self._visualizer.scale

        self._accumulator.encode_render_pass(command_encoder, True)

