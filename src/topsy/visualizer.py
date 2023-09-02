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
from . import util

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Visualizer:
    colormap_name = config.DEFAULT_COLORMAP
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO

    show_status = True
    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = ()):


        self.canvas = canvas.VisualizerCanvas(visualizer=self, title="topsy")
        self.adapter: wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas, power_preference="high-performance")
        self.device: wgpu.GPUDevice = self.adapter.request_device()
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()
        self.canvas_format = self.context.get_preferred_format(self.adapter)

        self._n_sph_channels = 1
        self._recent_frame_times = []
        if self.canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self.canvas_format = self.canvas_format[:-5]

        self.context.configure(device=self.device, format=self.canvas_format)

        self._render_resolution = 1024

        self.render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.rg32float,
            label="sph_render_texture",
        )

        self.data_loader = data_loader_class(self.device, *data_loader_args)

        self._colormap = colormap.Colormap(self, weighted_average = False)
        self._sph = sph.SPH(self, self.render_texture)
        #self._sph = multiresolution_sph.MultiresolutionSPH(self, self.render_texture)

        self._last_status_update = 0.0

        self._colorbar = colorbar.ColorbarOverlay(self, 0.0, 1.0, self.colormap_name, "TODO")
        self._scalebar = scalebar.ScalebarOverlay(self)
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 80, color=(1, 1, 1, 1))
        self._display_fullres_render_status = False # when True, customises info text to refer to full-res render

        self._render_timer = util.TimeGpuOperation(self.device)

        self.vmin_vmax_is_set = False

        self.invalidate()



    def invalidate(self):
        self.canvas.request_draw(self.draw)

    def rotate(self, dx, dy):
        dx_rotation_matrix = self._x_rotation_matrix(dx*0.01)
        dy_rotation_matrix = self._y_rotation_matrix(dy*0.01)
        self._sph.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self._sph.rotation_matrix
        self.invalidate()

    def reset_view(self):
        self._sph.rotation_matrix = np.eye(3)
        self.scale = config.DEFAULT_SCALE

    @property
    def scale(self):
        """Return the scalefactor from kpc to viewport coordinates. Viewport will therefore be 2*scale wide."""
        return self._sph.scale
    @scale.setter
    def scale(self, value):
        self._sph.scale = value
        self.invalidate()

    @property
    def quantity_name(self):
        """The name of the quantity being visualised, or None if density projection."""
        return self.data_loader.quantity_name

    @property
    def averaging(self):
        """True if the quantity being visualised is a weighted average, False if it is a mass projection."""
        return self.data_loader.quantity_name is not None

    @quantity_name.setter
    def quantity_name(self, value):
        self.data_loader.quantity_name = value
        self.vmin_vmax_is_set = False
        self._colormap = colormap.Colormap(self, weighted_average = value is not None)
        self._colorbar = colorbar.ColorbarOverlay(self, 0.0, 1.0, self.colormap_name,
                                                  r"$\log_{10}$ "+self.data_loader.get_quantity_label())
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

    def _check_whether_inactive(self):
        if time.time()-self._last_lores_draw_time>config.FULL_RESOLUTION_RENDER_AFTER*0.999:
            self._sph.downsample_factor = 1
            self._last_lores_draw_time = np.inf
            self._display_fullres_render_status = True
            self.invalidate()



    def draw(self):

        ce_label = "sph_render"
        # labelling this is useful for understanding performance in macos instruments
        if self._sph.downsample_factor>1:
            ce_label += f"_ds{self._sph.downsample_factor:d}"
        else:
            ce_label += "_fullres"

        command_encoder : wgpu.GPUCommandEncoder = self.device.create_command_encoder(label=ce_label)
        self._sph.encode_render_pass(command_encoder)

        with self._render_timer:
            self.device.queue.submit([command_encoder.finish()])

        # in principle, we can often use the same command encoder for the drawing into the final
        # buffer, but when vmin/vmax needs to be set, we need to render the image first.
        # So we just always use a new command encoder for the final render. It is unclear whether
        # this has any performance impact.

        if not self.vmin_vmax_is_set:
            logger.info("Setting vmin/vmax")
            self._colormap.set_vmin_vmax()
            self.vmin_vmax_is_set = True
            self._colorbar.vmin = self._colormap.vmin
            self._colorbar.vmax = self._colormap.vmax
            self._colorbar.update()

        command_encoder = self.device.create_command_encoder(label="render_to_screen")
        self._colormap.encode_render_pass(command_encoder)
        self._colorbar.encode_render_pass(command_encoder)
        self._scalebar.encode_render_pass(command_encoder)

        if self.show_status:
            self._update_and_display_status(command_encoder)

        self.device.queue.submit([command_encoder.finish()])

        if self._sph.downsample_factor>1:
            self._last_lores_draw_time = time.time()
            wgpu.gui.auto.call_later(config.FULL_RESOLUTION_RENDER_AFTER, self._check_whether_inactive)
        elif self._render_timer.last_duration>1/config.TARGET_FPS and self._sph.downsample_factor==1:
            # this will affect the NEXT frame, not this one!
            self._sph.downsample_factor = int(np.floor(float(config.TARGET_FPS)*self._render_timer.last_duration))


    def _update_and_display_status(self, command_encoder):
        now = time.time()
        if now - self._last_status_update > config.STATUS_LINE_UPDATE_INTERVAL:
            self._last_status_update = now
            self._status.text = f"${1.0 / self._render_timer.running_mean_duration:.0f}$ fps"
            if self._sph.downsample_factor > 1:
                self._status.text += f", downsample={self._sph.downsample_factor:d}"
            if self._display_fullres_render_status:
                self._status.text = f"Full-res render took {self._render_timer.last_duration:.2f} s"
                self._display_fullres_render_status = False
            self._status.update()

        self._status.encode_render_pass(command_encoder)

    def get_rendered_image(self) -> np.ndarray:
        im = self.device.queue.read_texture({'texture':self.render_texture, 'origin':(0, 0, 0)},
                                            {'bytes_per_row':8*self._render_resolution},
                                            (self._render_resolution, self._render_resolution, 1))
        np_im = np.frombuffer(im, dtype=np.float32).reshape((self._render_resolution, self._render_resolution, 2))
        if self.averaging:
            im = np_im[:,:,1]/np_im[:,:,0]
        else:
            im = np_im[:,:,0]
        return im

    def save(self, filename='output.pdf'):
        image = self.get_rendered_image()
        import pylab as p
        fig = p.figure()
        p.clf()
        p.set_cmap(self.colormap_name)
        extent = np.array([-1., 1., -1., 1.])*self.scale
        if self._colormap.log_scale:
            image = np.log10(image)

        p.imshow(image,
                 vmin=self._colormap.vmin,
                 vmax=self._colormap.vmax,
                 extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self.colorbar_label)
        p.savefig(filename)
        p.close(fig)

    def run(self):
        wgpu.gui.auto.run()
