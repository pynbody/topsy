from __future__ import annotations

import logging
import numpy as np
import time
import wgpu

from contextlib import contextmanager

from . import config
from . import canvas
from . import colormap
from . import sph, periodic_sph
from . import colorbar
from . import text
from . import scalebar
from . import loader
from . import util
from . import line
from . import simcube
from . import view_synchronizer
from . import particle_buffers
from .drawreason import DrawReason

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class VisualizerBase:
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    show_status = True
    device = None # device will be shared across all instances

    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = (), data_loader_kwargs={},
                 *, render_resolution = config.DEFAULT_RESOLUTION, periodic_tiling = False,
                 colormap_name = config.DEFAULT_COLORMAP, canvas_class = canvas.VisualizerCanvas,
                 hdr = False, rgb=False):
        self._hdr = hdr
        self._rgb = rgb
        self._colormap_name = colormap_name
        self._render_resolution = render_resolution
        self._sph_class = sph.SPH
        self._colorbar = None
        self._colormap = None

        self.crosshairs_visible = False

        self._prevent_sph_rendering = False # when True, prevents the sph from rendering, to ensure quick screen updates

        self.show_colorbar = True
        self.show_scalebar = True

        self.canvas = canvas_class(visualizer=self, title="topsy")

        self._setup_wgpu()

        self.data_loader = data_loader_class(self.device, *data_loader_args, **data_loader_kwargs)
        self.particle_buffers = particle_buffers.ParticleBuffers(self.data_loader, self.device)

        self.periodicity_scale = self.data_loader.get_periodicity_scale()

        self._periodic_tiling = periodic_tiling

        if periodic_tiling:
            self._sph = periodic_sph.PeriodicSPH(self, self._render_resolution)
        elif self._rgb:
            logger.info("Using RGB renderer")
            self._sph = sph.RGBSPH(self, self._render_resolution)
        else:
            self._sph = sph.SPH(self, self._render_resolution)

        self.render_texture = self._sph.get_output_texture()

        self.reset_view()

        self._reinitialize_colormap_and_bar()

        self._last_status_update = 0.0
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 40, color=(1, 1, 1, 1))

        self._scalebar = scalebar.ScalebarOverlay(self)

        self._crosshairs = line.Line(self,
                                     [(-1, 0,0,0), (1, 0,0,0),
                                      (200,200,0,0),
                                      (0, 1, 0, 0), (0, -1, 0, 0)],
                                     (1, 1, 1, 0.3) # color
                                     , 10.0)
        self._cube = simcube.SimCube(self, (1, 1, 1, 0.3), 10.0)

    def _setup_wgpu(self):
        self.adapter: wgpu.GPUAdapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        logger.info(f"GPU device: {self.adapter.info['adapter_type']} ({self.adapter.info['device']})")
        if self.device is None:
            max_buffer_size = self.adapter.limits['max-buffer-size']
            # on some systems, this is 2^64 which can lead to overflows
            if max_buffer_size > 2**63:
                max_buffer_size = 2**63
            type(self).device: wgpu.GPUDevice = self.adapter.request_device_sync(
                required_features=["texture-adapter-specific-format-features", "float32-filterable",
                                   ],
                required_limits={"max_buffer_size": max_buffer_size})
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context("wgpu")

        if self._hdr:
            self.canvas_format = "rgba16float"
        else:
            self.canvas_format = self.context.get_preferred_format(self.adapter)
            if self.canvas_format.endswith("-srgb"):
                # matplotlib colours aren't srgb. It might be better to convert
                # but for now, just stop the canvas being srgb
                self.canvas_format = self.canvas_format[:-5]

        self.context.configure(device=self.device, format=self.canvas_format)

        logger.info(f"Canvas format {self.canvas_format}")

    def invalidate(self, reason=DrawReason.CHANGE):
        # NB no need to check if we're already pending a draw - rendercanvas does that for us
        self.canvas.request_draw(lambda: self.draw(reason))

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
    def colormap_name(self):
        return self._colormap_name

    @property
    def colormap(self):
        return self._colormap

    @colormap_name.setter
    def colormap_name(self, value):
        self._colormap_name = value
        self._reinitialize_colormap_and_bar(keep_scale=True)
        self.invalidate(reason=DrawReason.PRESENTATION_CHANGE)

    @property
    def position_offset(self):
        return self._sph.position_offset

    @position_offset.setter
    def position_offset(self, value):
        self._sph.position_offset = value
        self.invalidate()

    def reset_view(self):
        self._sph.rotation_matrix = np.eye(3)
        period_scale = self.data_loader.get_periodicity_scale()
        if period_scale is not None:
            self.scale = period_scale / 2
        else:
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
        return self.particle_buffers.quantity_name

    @property
    def averaging(self):
        """True if the quantity being visualised is a weighted average, False if it is a mass projection."""
        return self.particle_buffers.quantity_name is not None

    @quantity_name.setter
    def quantity_name(self, value):
        if value == self.particle_buffers.quantity_name:
            return

        if value is not None:
            # see if we can get it. Assume it'll be cached, so this won't waste time.
            try:
                self.data_loader.get_named_quantity(value)
            except Exception as e:
                raise ValueError(f"Unable to get quantity named '{value}'") from e

        self.particle_buffers.quantity_name = value
        
        self._reinitialize_colormap_and_bar()
        

    def _reinitialize_colormap_and_bar(self, keep_scale=False):
        """Reinitialize the colormap and colorbar.
        
        If keep_scale is False, render and figure out the min/max values for the colormap too.
        """
        if self._colormap is not None:
            old_vmin, old_vmax, old_log = self._colormap.vmin, self._colormap.vmax, self._colormap.log_scale
        else:
            if keep_scale:
                raise ValueError("Cannot keep scale if colormap not yet initialized")

        if self._rgb:
            if self._hdr:
                self._colormap = colormap.RGBHDRColormap(self)
            else:
                self._colormap = colormap.RGBColormap(self)
        elif self._hdr:
            self._colormap = colormap.HDRColormap(self, weighted_average=self.quantity_name is not None)

        else:
            self._colormap = colormap.Colormap(self, weighted_average=self.quantity_name is not None)
            
        if not keep_scale:
            self._sph.render(DrawReason.EXPORT)
            self._colormap.autorange_vmin_vmax()
        else:
            self._colormap.vmin = old_vmin
            self._colormap.vmax = old_vmax
            self._colormap.log_scale = old_log

        if not self._rgb and not self._hdr:
            self._colorbar = colorbar.ColorbarOverlay(self, self.vmin, self.vmax, self.colormap_name, self._get_colorbar_label())

        self.invalidate(DrawReason.PRESENTATION_CHANGE)


    def _get_colorbar_label(self):
        label = self.data_loader.get_quantity_label(self.quantity_name)
        if self._colormap.log_scale:
            label = r"$\log_{10}$ " + label
        return label

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

    @contextmanager
    def prevent_sph_rendering(self):
        self._prevent_sph_rendering = True
        try:
            yield
        finally:
            self._prevent_sph_rendering = False

    def draw(self, reason, target_texture_view=None):

        if not self._prevent_sph_rendering:
            self.render_sph(reason)

        command_encoder = self.device.create_command_encoder()

        if target_texture_view is None:
            target_texture_view = self.canvas.get_context("wgpu").get_current_texture().create_view()

        color_prescale = self._sph.last_render_mass_scale if not self.averaging else 1.0
        self._colormap.encode_render_pass(command_encoder, target_texture_view, color_prescale)
        if self.show_colorbar and self._colorbar is not None:
            self._colorbar.encode_render_pass(command_encoder, target_texture_view)
        if self.show_scalebar:
            self._scalebar.encode_render_pass(command_encoder, target_texture_view)
        if self.crosshairs_visible:
            self._crosshairs.encode_render_pass(command_encoder, target_texture_view)
        if self._periodic_tiling:
            self._cube.encode_render_pass(command_encoder, target_texture_view)

        if self.show_status:
            self._update_and_display_status(command_encoder, target_texture_view)

        self.device.queue.submit([command_encoder.finish()])

        if reason != DrawReason.EXPORT and (not self._prevent_sph_rendering):
            if self._sph.needs_refine():
                self.invalidate(DrawReason.REFINE)

    def render_sph(self, draw_reason = DrawReason.CHANGE):
        self._sph.render(draw_reason)

    @property
    def vmin(self):
        return self._colormap.vmin

    @property
    def vmax(self):
        return self._colormap.vmax

    @vmin.setter
    def vmin(self, value):
        self._colormap.vmin = value
        self._refresh_colorbar()
        self.invalidate(DrawReason.PRESENTATION_CHANGE)

    @vmax.setter
    def vmax(self, value):
        self._colormap.vmax = value
        self._refresh_colorbar()
        self.invalidate(DrawReason.PRESENTATION_CHANGE)

    @property
    def gamma(self):
        if hasattr(self._colormap, 'gamma'):
            return self._colormap.gamma
        else:
            return None

    @gamma.setter
    def gamma(self, value):
        if hasattr(self._colormap, 'gamma'):
            self._colormap.gamma = value
            self.invalidate(DrawReason.PRESENTATION_CHANGE)

    @property
    def log_scale(self):
        return self._colormap.log_scale

    @log_scale.setter
    def log_scale(self, value):
        self._colormap.log_scale = value
        self._refresh_colorbar()
        self.invalidate(DrawReason.PRESENTATION_CHANGE)

    def _refresh_colorbar(self):
        if self._colorbar is None:
            return
        self._colorbar.vmin = self._colormap.vmin
        self._colorbar.vmax = self._colormap.vmax
        self._colorbar.label = self._get_colorbar_label()
        self._colorbar.update()

    def sph_clipspace_to_screen_clipspace_matrix(self):
        aspect_ratio = self.canvas.width_physical / self.canvas.height_physical
        if aspect_ratio>1:
            y_squash = aspect_ratio
            x_squash = 1.0
        elif aspect_ratio<1:
            y_squash = 1.0
            x_squash = 1.0/aspect_ratio
        else:
            x_squash = 1.0
            y_squash = 1.0

        matr = np.eye(4, dtype=np.float32)
        matr[0,0] = x_squash
        matr[1,1] = y_squash
        return matr



    def display_status(self, text, timeout=0.5):
        self._override_status_text = text
        self._override_status_text_until = time.time()+timeout

    def _update_and_display_status(self, command_encoder, target_texture_view):
        now = time.time()
        if hasattr(self, "_override_status_text_until") and now<self._override_status_text_until:
            if self._status.text!=self._override_status_text and now-self._last_status_update>config.STATUS_LINE_UPDATE_INTERVAL_RAPID:
                self._status.text = self._override_status_text
                self._last_status_update = now
                self._status.update()

        elif now - self._last_status_update > config.STATUS_LINE_UPDATE_INTERVAL:
            self._last_status_update = now
            self._status.text = f"${self._sph.last_render_fps:.0f}$ fps"
            factor = np.round(self._sph.last_render_mass_scale, 1)
            if factor>1.1:
                self._status.text += f" /{factor:.1f}ds"
            geom_factor = self._sph._render_progression.get_fraction_volume_selected()
            if geom_factor<0.9:
                self._status.text += f" /{1./geom_factor:.1f}gf"

            self._status.update()

        self._status.encode_render_pass(command_encoder, target_texture_view)

    def get_sph_image(self) -> np.ndarray:
        return self._sph.get_image(self.averaging)

    def get_depth_image(self) -> np.ndarray:
        return self._sph.get_depth_image()


    def get_presentation_image(self, resolution=(640,480)) -> np.ndarray:
        texture: wgpu.GPUTexture = self.device.create_texture(
            size=(resolution[0], resolution[1], 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.COPY_SRC,
            format=self.canvas_format,
            label="output_texture",
        )
        self.draw(DrawReason.EXPORT, texture.create_view())

        size = texture.size

        if texture.format.endswith("8unorm"):
            bytes_per_pixel = 4
            np_type = np.uint8
        elif texture.format.endswith("16float"):
            bytes_per_pixel = 8
            np_type = np.float16
        else:
            raise ValueError(f"Unsupported texture format {texture.format}")

        data = self.device.queue.read_texture(
            {
                "texture": texture,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )

        return np.frombuffer(data, np_type).reshape(size[1], size[0], 4)


    def save(self, filename='output.pdf'):
        image = self.get_sph_image()
        import matplotlib.pyplot as p
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
        p.colorbar().set_label(self._colorbar.label)
        p.savefig(filename)
        p.close(fig)

    def show(self, force=False):
        from rendercanvas import jupyter
        if isinstance(self.canvas, jupyter.RenderCanvas):
            return self.canvas
        else:
            from rendercanvas import qt # can only safely import this if we think we're running in a qt environment
            assert isinstance(self.canvas, qt.RenderCanvas)
            self.canvas.show()
            if force or not util.is_inside_ipython():
                qt.loop.run()
            elif not util.is_ipython_running_qt_event_loop():
                # is_inside_ipython_console must be True; if it were False, the previous branch would have run
                # instead.
                print("\r\nYou appear to be running from inside ipython, but the gui event loop is not running.\r\n"
                      "Please run %gui qt in ipython before calling show().\r\n"
                      "\r\n"
                      "Alternatively, if you do not want to continue interacting with ipython while the\r\n"
                      "visualizer is running, you can call show(force=True) to run the gui without access\r\n"
                      "to the ipython console until you close the visualizer window.\r\n\r\n"
                      )
        #else:
        #    raise RuntimeError("The wgpu library is using a gui backend that topsy does not recognize")

    def _ipython_display_(self):
        if isinstance(self.canvas, canvas.jupyter.VisualizerCanvas):
            self.canvas.ipython_display_with_widgets()
        else:
            from IPython.display import display
            display(repr(self))

class Visualizer(view_synchronizer.SynchronizationMixin, VisualizerBase):
    pass