from __future__ import annotations

import logging
import numpy as np
import time
import wgpu
import wgpu.backends.rs # noqa: F401, Select Rust backend

from . import config
from . import canvas
from . import colormap
from . import multiresolution_sph, sph
from . import colorbar
from . import text
from . import scalebar
from . import loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Visualizer:
    colorbar_label = r"$\mathrm{log}_{10}$ density / $M_{\odot} / \mathrm{kpc}^2$"
    colormap_name = config.DEFAULT_COLORMAP
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = ()):


        self.canvas = canvas.VisualizerCanvas(visualizer=self, title="topsy")
        self.adapter: wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device: wgpu.GPUDevice = self.adapter.request_device()
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()
        self.canvas_format = self.context.get_preferred_format(self.adapter)
        self._recent_frame_times = []
        if self.canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self.canvas_format = self.canvas_format[:-5]

        self.context.configure(device=self.device, format=self.canvas_format)

        self._render_resolution = 512
        self.render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.r32float,
            label="sph_render_texture",
        )

        self.data_loader = data_loader_class(self.device, *data_loader_args)

        self._colormap = colormap.Colormap(self)
        #self._sph = sph.SPH(self, self.render_texture)
        self._sph = multiresolution_sph.MultiresolutionSPH(self, self.render_texture)

        self._last_status_update = 0.0

        self._colorbar = colorbar.ColorbarOverlay(self, 0.0, 1.0, self.colormap_name, self.colorbar_label)
        self._scalebar = scalebar.ScalebarOverlay(self)
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 80, color=(1, 1, 1, 1))

        self.vmin_vmax_is_set = False

        self.invalidate()



    def invalidate(self):
        self.canvas.request_draw(self.draw)

    def rotate(self, dx, dy):
        dx_rotation_matrix = self._x_rotation_matrix(dx*0.01)
        dy_rotation_matrix = self._y_rotation_matrix(dy*0.01)
        self._sph.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self._sph.rotation_matrix
        self.invalidate()

    @property
    def scale(self):
        """Return the scalefactor from kpc to viewport coordinates. Viewport will therefore be 2*scale wide."""
        return self._sph.scale
    @scale.setter
    def scale(self, value):
        self._sph.scale = value
        self.invalidate()

    @staticmethod
    def _y_rotation_matrix(angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def _x_rotation_matrix(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def draw(self):
        start = time.time()
        command_encoder = self.device.create_command_encoder()

        self._sph.encode_render_pass(command_encoder)

        if not self.vmin_vmax_is_set:
            logger.info("Setting vmin/vmax")
            self.device.queue.submit([command_encoder.finish()]) # have to render the image to get the min/max
            self._colormap.set_vmin_vmax()
            command_encoder = self.device.create_command_encoder() # new command encoder needed
            self.vmin_vmax_is_set = True
            self._colorbar.vmin = self._colormap.vmin
            self._colorbar.vmax = self._colormap.vmax
            self._colorbar.update()

        self._colormap.encode_render_pass(command_encoder)

        self._colorbar.encode_render_pass(command_encoder)
        self._scalebar.encode_render_pass(command_encoder)

        self.device.queue.submit([command_encoder.finish()])
        end = time.time()

        self._recent_frame_times.append(end-start)
        if len(self._recent_frame_times)>20:
            self._recent_frame_times.pop(0)

        if end - self._last_status_update>0.2:
            self._last_status_update = end
            self._status.text = f"1/render_time = {1.0/np.mean(self._recent_frame_times):.0f} s$^{{-1}}$"
            self._status.update()

        command_encoder = self.device.create_command_encoder()
        self._status.encode_render_pass(command_encoder)
        self.device.queue.submit([command_encoder.finish()])

    def get_rendered_image(self) -> np.ndarray:
        im = self.device.queue.read_texture({'texture':self.render_texture, 'origin':(0, 0, 0)},
                                            {'bytes_per_row':4*self._render_resolution},
                                            (self._render_resolution, self._render_resolution, 1))
        im = np.frombuffer(im, dtype=np.float32).reshape((self._render_resolution, self._render_resolution))
        return im

    def save(self):
        logger.info("Saving to pdf")
        mybuffer = self.get_rendered_image()
        import pylab as p
        fig = p.figure()
        p.clf()
        p.set_cmap(self.colormap_name)
        extent = np.array([-1., 1., -1., 1.])*self.scale
        p.imshow(np.log10(mybuffer), vmin=self._colormap.vmin, vmax=self._colormap.vmax, extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self.colorbar_label)
        p.savefig("output.pdf")
        p.close(fig)

    def run(self):
        wgpu.gui.auto.run()
