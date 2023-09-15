from __future__ import annotations

import logging
import numpy as np
import time
import wgpu
import wgpu.backends.rs # noqa: F401, Select Rust backend

from . import config
from . import canvas
from . import colormap
from . import multiresolution_sph, sph, periodic_sph
from . import colorbar
from . import text
from . import scalebar
from . import loader
from . import util
from . import line
from . import simcube
from . import view_synchronizer
from .drawreason import DrawReason

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class VisualizerBase:
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    show_status = True

    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = (),
                 *, render_resolution = config.DEFAULT_RESOLUTION, periodic_tiling = False,
                 colormap_name = config.DEFAULT_COLORMAP):
        self.colormap_name = colormap_name
        self._draw_pending = False
        self._render_resolution = render_resolution
        self.crosshairs_visible = False
        self._display_fullres_render_status = False  # when True, customises info text to refer to full-res render
        self.vmin_vmax_is_set = False

        self.canvas = canvas.VisualizerCanvas(visualizer=self, title="topsy")

        self._setup_wgpu()

        self.data_loader = data_loader_class(self.device, *data_loader_args)
        self.periodicity_scale = self.data_loader.get_periodicity_scale()

        self._colormap = colormap.Colormap(self, weighted_average = False)
        if periodic_tiling:
            self._sph = periodic_sph.PeriodicSPH(self, self.render_texture)
        else:
            self._sph = sph.SPH(self, self.render_texture)
        #self._sph = multiresolution_sph.MultiresolutionSPH(self, self.render_texture)

        self._last_status_update = 0.0
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 80, color=(1, 1, 1, 1))

        self._colorbar = colorbar.ColorbarOverlay(self, 0.0, 1.0, self.colormap_name, "TODO")
        self._scalebar = scalebar.ScalebarOverlay(self)

        self._crosshairs = line.Line(self,
                                     [(-1, 0,0,0), (1, 0,0,0),
                                      (200,200,0,0),
                                      (0, 1, 0, 0), (0, -1, 0, 0)],
                                     (1, 1, 1, 0.3) # color
                                     , 10.0)
        self._cube = simcube.SimCube(self, (1, 1, 1, 0.3), 10.0)

        self._render_timer = util.TimeGpuOperation(self.device)

        self.invalidate(DrawReason.INITIAL_UPDATE)

    def _setup_wgpu(self):
        self.adapter: wgpu.GPUAdapter = wgpu.request_adapter(canvas=self.canvas,
                                                             power_preference="high-performance")
        self.device: wgpu.GPUDevice = self.adapter.request_device()
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()
        self.canvas_format = self.context.get_preferred_format(self.adapter)
        if self.canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self.canvas_format = self.canvas_format[:-5]
        self.context.configure(device=self.device, format=self.canvas_format)
        self.render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.rg32float,
            label="sph_render_texture",
        )

    def invalidate(self, reason=DrawReason.CHANGE):
        if not self._draw_pending:
            self.canvas.request_draw(lambda: self.draw(reason))
            self._draw_pending = True

    def rotate(self, x_angle, y_angle):
        dx_rotation_matrix = self._x_rotation_matrix(x_angle)
        dy_rotation_matrix = self._y_rotation_matrix(y_angle)
        self.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self.rotation_matrix

    @property
    def rotation_matrix(self):
        return self._sph.rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        self._sph.rotation_matrix = value
        self.invalidate()

    @property
    def position_offset(self):
        return self._sph.position_offset

    @position_offset.setter
    def position_offset(self, value):
        self._sph.position_offset = value
        self.invalidate()

    def reset_view(self):
        self._sph.rotation_matrix = np.eye(3)
        self.scale = config.DEFAULT_SCALE
        self._sph.position_offset = np.zeros(3)

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



    def draw(self, reason):
        if reason!=DrawReason.PRESENTATION_CHANGE:
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

        if not self.vmin_vmax_is_set:
            logger.info("Setting vmin/vmax")
            self._colormap.autorange_vmin_vmax()
            self.vmin_vmax_is_set = True
            self._refresh_colorbar()


        command_encoder = self.device.create_command_encoder(label="render_to_screen")
        self._colormap.encode_render_pass(command_encoder)
        self._colorbar.encode_render_pass(command_encoder)
        self._scalebar.encode_render_pass(command_encoder)
        if self.crosshairs_visible:
            self._crosshairs.encode_render_pass(command_encoder)
        self._cube.encode_render_pass(command_encoder)

        if self.show_status:
            self._update_and_display_status(command_encoder)

        self.device.queue.submit([command_encoder.finish()])

        if self._sph.downsample_factor>1:
            self._last_lores_draw_time = time.time()
            wgpu.gui.qt.call_later(config.FULL_RESOLUTION_RENDER_AFTER, self._check_whether_inactive)
        elif self._render_timer.last_duration>1/config.TARGET_FPS and self._sph.downsample_factor==1:
            # this will affect the NEXT frame, not this one!
            self._sph.downsample_factor = int(np.floor(float(config.TARGET_FPS)*self._render_timer.last_duration))

        self._draw_pending = False

    @property
    def vmin(self):
        return self._colormap.vmin

    @property
    def vmax(self):
        return self._colormap.vmax

    @vmin.setter
    def vmin(self, value):
        self._colormap.vmin = value
        self.vmin_vmax_is_set = True
        self._refresh_colorbar()
        self.invalidate()

    @vmax.setter
    def vmax(self, value):
        self._colormap.vmax = value
        self.vmin_vmax_is_set = True
        self._refresh_colorbar()
        self.invalidate()

    def _refresh_colorbar(self):
        self._colorbar.vmin = self._colormap.vmin
        self._colorbar.vmax = self._colormap.vmax
        self._colorbar.update()

    def sph_clipspace_to_screen_clipspace_matrix(self):
        aspect_ratio = self.canvas.width_physical / self.canvas.height_physical
        if aspect_ratio>1:
            y_squash = aspect_ratio
            x_squash = 1.0
        elif aspect_ratio<1:
            y_squash = 1.0
            x_squash = 1.0/aspect_ratio

        matr = np.eye(4, dtype=np.float32)
        matr[0,0] = x_squash
        matr[1,1] = y_squash
        return matr



    def display_status(self, text, timeout=0.5):
        self._override_status_text = text
        self._override_status_text_until = time.time()+timeout

    def _update_and_display_status(self, command_encoder):
        now = time.time()
        if hasattr(self, "_override_status_text_until") and now<self._override_status_text_until:
            if self._status.text!=self._override_status_text and now-self._last_status_update>config.STATUS_LINE_UPDATE_INTERVAL_RAPID:
                self._status.text = self._override_status_text
                self._last_status_update = now
                self._status.update()

        elif now - self._last_status_update > config.STATUS_LINE_UPDATE_INTERVAL:
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

    def show(self, force=False):
        from wgpu.gui import jupyter, qt
        if isinstance(self.canvas, jupyter.WgpuCanvas):
            return self.canvas
        elif isinstance(self.canvas, qt.WgpuCanvas):
            self.canvas.show()
            event_loop_running = False
            if not force:
                try:
                    __IPYTHON__
                    import IPython.lib.guisupport
                    event_loop_running = IPython.lib.guisupport.is_event_loop_running_qt4()
                    if not event_loop_running:
                        print("You appear to be running from inside ipython, but the gui event loop is not running.")
                        print("Please run '%gui qt' in ipython before calling show().")
                        print("Alternatively, if you do not want to continue interacting with ipython while the"
                              "visualizer is running, you can call show(force=True) to force the gui to take"
                              "precedence over the ipython command line.")
                        return

                except (ImportError, NameError):
                    pass

            if not event_loop_running:
                qt.run()


class Visualizer(view_synchronizer.SynchronizationMixin, VisualizerBase):
    pass