from __future__ import annotations

import logging
import numpy as np
import time
import wgpu

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

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

    _sph : Optional[sph.SPH]
    _colormap: Optional[colormap.ColormapHolder]
    _colorbar: Optional[colorbar.ColorbarOverlay]

    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = (), data_loader_kwargs={},
                 *, render_resolution = config.DEFAULT_RESOLUTION, periodic_tiling = False,
                 colormap_name = config.DEFAULT_COLORMAP, canvas_class = canvas.VisualizerCanvas,
                 render_mode='univariate'):
        
        self._render_resolution = render_resolution
        self._colorbar = None
        self._sph = None
        self._colormap = None
        self._encoder_executor = ThreadPoolExecutor(max_workers=1) # 1 worker to prevent GIL contention

        self.crosshairs_visible = False

        self._prevent_sph_rendering = False # when True, prevents the sph from rendering, to ensure quick screen updates

        self.show_colorbar = True
        self.show_scalebar = True

        self.canvas = canvas_class(visualizer=self, title="topsy")
        self._validate_render_mode(render_mode)
        self._render_mode = render_mode
        
        self._setup_wgpu()
        self._configure_canvas_context()
        self._initialize_data_loader_and_buffers(data_loader_class, data_loader_args, data_loader_kwargs)
        self._initialize_overlays()

        self._periodic_tiling = periodic_tiling

        self._initialize_sph_and_colormap_and_bar(colormap_name)

        self._last_status_update = 0.0

    def _initialize_data_loader_and_buffers(self, data_loader_class, data_loader_args, data_loader_kwargs):
        self.data_loader = data_loader_class(self.device, *data_loader_args, **data_loader_kwargs)
        self.particle_buffers = particle_buffers.ParticleBuffers(self.data_loader, self.device,
                                                                 self.data_loader.get_render_progression().get_max_particle_regions_per_block())

        self.periodicity_scale = self.data_loader.get_periodicity_scale()
        

    def _initialize_overlays(self):
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 40, color=(1, 1, 1, 1))

        self._scalebar = scalebar.ScalebarOverlay(self)

        self._crosshairs = line.Line(self,
                                     [(-1, 0,0,0), (1, 0,0,0),
                                      (200,200,0,0),
                                      (0, 1, 0, 0), (0, -1, 0, 0)],
                                     (1, 1, 1, 0.3) # color
                                     , 10.0)
        self._cube = simcube.SimCube(self, (1, 1, 1, 0.3), 10.0)

    def _get_sph_class_for_render_mode(self, render_mode):
        """Map render mode to appropriate SPH class."""
        if render_mode == 'rgb' or render_mode == 'rgb-hdr':
            return sph.RGBSPH
        elif render_mode == 'surface':
            return sph.DepthSPHWithOcclusion
        else:  # 'univariate', 'bivariate'
            return sph.SPH
    
    def _get_colormap_parameters_for_render_mode(self, render_mode):
        """Generate colormap parameters for the given render mode."""
        colormap_params = {'weighted_average': self.quantity_name is not None}
        
        if render_mode == 'rgb':
            colormap_params.update({'type': 'rgb', 'hdr': False, 'log': True})
        elif render_mode == 'rgb-hdr':
            colormap_params.update({'type': 'rgb', 'hdr': True, 'log': True})
        elif render_mode == 'bivariate':
            colormap_params.update({'type': 'bivariate'})
        elif render_mode == 'surface':
            colormap_params.update({'type': 'surface'})
        else:  # 'univariate'
            colormap_params.update({'type': 'density'})
            
        return colormap_params

    def _initialize_sph_and_colormap_and_bar(self, colormap_name = None):
        """(Re-)initialize SPH, colormap and colorbar"""

        if self._sph is not None:
            old_rotation = self._sph.rotation_matrix
            old_position = self._sph.position_offset
            old_scale = self._sph.scale
        else:
            old_rotation = None
            old_position = None
            old_scale = None

        if self._periodic_tiling:
            self._sph = periodic_sph.PeriodicSPH(self, self._render_resolution)
        else:
            # Use render mode to select SPH class
            sph_class = self._get_sph_class_for_render_mode(self._render_mode)
            logger.info(f"Using {sph_class.__name__} renderer for render mode '{self._render_mode}'")
            self._sph = sph_class(self, self._render_resolution)

        self.reset_view(rotation_matrix = old_rotation, position_offset = old_position, scale = old_scale)
        self.invalidate()

        if colormap_name is None:
            colormap_name = self._colormap.get_parameter('colormap_name')

        self.render_texture = self._sph.get_output_texture()

        self._colormap: colormap.ColormapHolder = colormap.ColormapHolder(self.device, self.render_texture,
                                                                          self.canvas_format)
        self._colormap.update_parameters({'colormap_name': colormap_name})

        self._initialize_colormap_and_bar()

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
                                  "multi-draw-indirect"],
                required_limits={"max_buffer_size": max_buffer_size})
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context("wgpu")

    def _render_mode_to_canvas_format(self, render_mode):
        if render_mode is None:
            return None
        elif render_mode.endswith('hdr'):
            format = "rgba16float"
        else:
            format = self.context.get_preferred_format(self.adapter)
            if format.endswith("-srgb"):
                # matplotlib colours aren't srgb. It might be better to convert
                # but for now, just stop the canvas being srgb
                format = format[:-5]
        return format 
        
    def _configure_canvas_context(self):
        self.canvas_format = self._render_mode_to_canvas_format(self._render_mode)
        self.context.configure(device=self.device, format=self.canvas_format)
        logger.info(f"Canvas format {self.canvas_format}")

    def invalidate(self, reason=DrawReason.CHANGE):
        self._sph.invalidate(reason)

        # NB no need to check if we're already pending a draw - rendercanvas does that for us
        self.canvas.request_draw(lambda: self.draw(reason))

    def rotate(self, x_angle, y_angle):
        dx_rotation_matrix = self._x_rotation_matrix(x_angle)
        dy_rotation_matrix = self._y_rotation_matrix(y_angle)
        self.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self.rotation_matrix

    @property
    def colormap(self):
        return self._colormap

    def _update_render_mode(self, new_render_mode):
        self._validate_render_mode(new_render_mode)

        old_render_mode = getattr(self, "_render_mode", None)   
        self._render_mode = new_render_mode

        logger.info(f"Initializing pipeline for render mode '{self._render_mode}'")
        if self._render_mode_to_canvas_format(old_render_mode) != self._render_mode_to_canvas_format(self._render_mode):
            # If canvas format changes, reconfigure the canvas context:
            self._configure_canvas_context()

            # when canvas format changes we need to re-initialize the overlays:
            self._initialize_overlays()

        self._initialize_sph_and_colormap_and_bar()
        
        self.invalidate(DrawReason.CHANGE)

    def _validate_render_mode(self, new_render_mode):
        valid_modes = {'univariate', 'bivariate', 'rgb', 'rgb-hdr', 'surface'}
        if new_render_mode not in valid_modes:
            raise ValueError(f"Invalid render_mode '{new_render_mode}'. Valid modes: {valid_modes}")
    
 
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

    @property
    def render_mode(self):
        return self._render_mode 
    
    @render_mode.setter 
    def render_mode(self, value):
        self._update_render_mode(value)

    def reset_view(self, rotation_matrix=None, position_offset=None, scale=None):
        """Reset to the default view, or a specified rotation/position/scale if provided."""
        if rotation_matrix is None:
            rotation_matrix = np.eye(3)
        if position_offset is None:
            position_offset = -self.data_loader.get_initial_center()
            logger.info(f"Position offset: {position_offset}")
        if scale is None:
            period_scale = self.data_loader.get_periodicity_scale()
            if period_scale is not None:
                scale = period_scale / 2
            else:
                scale = config.DEFAULT_SCALE

        self._sph.rotation_matrix = rotation_matrix
        self._sph.scale = scale
        self._sph.position_offset = position_offset

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
        return self.quantity_name is not None

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
        self.invalidate(DrawReason.CHANGE)
        self._colormap.update_parameters({'vmin': None, 'vmax': None, 'log': None})
        self._initialize_colormap_and_bar()


    def colormap_autorange(self):
        self._colormap.autorange(self._sph.get_image())
        self.invalidate(DrawReason.PRESENTATION_CHANGE)

    def _initialize_colormap_and_bar(self):
        """Reinitialize the colormap and colorbar.

        """
        # Get colormap parameters based on render mode
        colormap_params = self._get_colormap_parameters_for_render_mode(self._render_mode)

        changed_type = self._colormap.update_parameters(colormap_params)

        colormap_params = self._colormap.get_parameters()

        if changed_type or colormap_params['vmin'] is None or colormap_params['vmax'] is None:
            logger.info("Autorange colormap parameters")
            self._colormap.autorange(self._sph.get_image())

        if colormap_params['type'] not in ('rgb', 'surface'):
            self._colorbar = colorbar.ColorbarOverlay(self, colormap_params['vmin'], colormap_params['vmax'],
                                                      colormap_params['colormap_name'], self._get_colorbar_label())
        else:
            self._colorbar = None


    def _get_colorbar_label(self):
        label = self.data_loader.get_quantity_label(self.quantity_name)
        if self._colormap.get_parameter('log'):
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

    def _encode_draw(self, target_texture_view):
        command_encoder = self.device.create_command_encoder()

        self._colormap.encode_render_pass(command_encoder, target_texture_view)
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

        result = command_encoder.finish()
        return result

    def draw(self, reason, target_texture_view=None):

        if target_texture_view is None:
            target_texture_view = self.canvas.get_context("wgpu").get_current_texture().create_view()

        command_buffer_future = self._encoder_executor.submit(self._encode_draw, target_texture_view)

        if not self._prevent_sph_rendering:
            self.render_sph(reason)

        self._colormap.set_scaling(*target_texture_view.size[:2], self._sph.last_render_mass_scale)

        self.device.queue.submit([command_buffer_future.result()])

        if reason != DrawReason.EXPORT and (not self._prevent_sph_rendering):
            if self._sph.needs_refine():
                self.invalidate(DrawReason.REFINE)

    def render_sph(self, draw_reason = DrawReason.CHANGE):
        self._sph.render(draw_reason)

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

        elif now - self._last_status_update > config.STATUS_LINE_UPDATE_INTERVAL and hasattr(self._sph,'last_render_fps'):
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
        """Get the logical content of the SPH image, possibly with post-processing but no colormaps"""
        return self._colormap.sph_raw_output_to_content(self._sph.get_image())
    
    def get_sph_presentation_image(self) -> np.ndarray:
        """Get the SPH image as a presentation image, i.e. with colormap applied but no additional layers such as colorbar."""
        texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                    wgpu.TextureUsage.COPY_SRC,
            format=self.canvas_format,
            label="output_texture",
        )

        self.render_sph(DrawReason.EXPORT)

        self._colormap.set_scaling(self._render_resolution, self._render_resolution, self._sph.last_render_mass_scale)

        command_encoder = self.device.create_command_encoder()
        self._colormap.encode_render_pass(command_encoder, texture.create_view())
        self.device.queue.submit([command_encoder.finish()])

        return self._texture_to_rgba_numpy(texture)


    def get_depth_image(self) -> np.ndarray:
        return self._sph.get_depth_image()

    def get_presentation_image(self, resolution=(640,480)) -> np.ndarray:
        """Get the full presentation image, complete with layers such as colorbar, scalebar and status line"""
        texture: wgpu.GPUTexture = self.device.create_texture(
            size=(resolution[0], resolution[1], 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.COPY_SRC,
            format=self.canvas_format,
            label="output_texture",
        )
        self.draw(DrawReason.EXPORT, texture.create_view())

        return self._texture_to_rgba_numpy(texture)

    def _texture_to_rgba_numpy(self, texture):
        size = texture.size

        is_bgr = texture.format.startswith("bgr")

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

        result = np.frombuffer(data, np_type).reshape(size[1], size[0], 4)

        if is_bgr:
            result = result[..., [2, 1, 0, 3]]

        return result


    def save(self, filename='output.pdf'):
        """Save the current view to a file, either as a numpy array or as a matplotlib image.
        
        If a numpy array, the logical content of the SPH image is saved, i.e. without colormaps.
        
        If a matplotlib image, the SPH image is colormapped and saved as an image using matplotlib to add
        the colorbar and axes so that these are rendered to vectors if a pdf is requested.

        Note that matplotlib is never used to perform the colormapping, so that bivariate and surface rendering
        outputs correctly.
        """
        self._sph.render(DrawReason.EXPORT)
        if filename.endswith(".npy"):
            image = self.get_sph_image()
            np.save(filename, image)
        else:
            import matplotlib.pyplot as p
            colormap_params = self._colormap.get_parameters()

            fig = p.figure()
            p.clf()
            p.set_cmap(colormap_params['colormap_name'])

            image = self.get_sph_presentation_image()

            extent = np.array([-1., 1., -1., 1.])*self.scale

            p.imshow(image, extent=extent)
            p.xlabel("$x$/kpc")

            cb_vmin = self._colormap.get_parameter('vmin')
            cb_vmax = self._colormap.get_parameter('vmax')

            if self._colorbar is not None:
                p.colorbar(
                    p.cm.ScalarMappable(
                        norm=p.Normalize(vmin=cb_vmin, vmax=cb_vmax),
                        cmap=colormap_params['colormap_name'],
                    ),
                    ax = p.gca()
                ).set_label(self._colorbar.label)
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