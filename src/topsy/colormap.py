from __future__ import annotations

import logging
import numpy as np
import re
import wgpu
import matplotlib

from . import config

from .drawreason import DrawReason
from .util import load_shader, preprocess_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Colormap:
    input_channels = 2
    fragment_shader = "fragment_main"
    percentile_scaling = [1.0, 99.9]
    map_dimension = wgpu.TextureViewDimension.d1

    parameter_dtype = np.dtype([("vmin", np.float32, (1,)),
                               ("vmax", np.float32, (1,)),
                               ("density_vmin", np.float32, (1,)),
                               ("density_vmax", np.float32, (1,)),
                               ("window_aspect_ratio", np.float32, (1,)),
                               ("gamma", np.float32, (1,))])

    def __init__(self, visualizer: Visualizer, weighted_average: bool = False):
        self._visualizer = visualizer
        self._device = visualizer.device
        self._colormap_name = visualizer.colormap_name
        self._input_texture = visualizer.render_texture
        self._output_format = visualizer.canvas_format
        self._weighted_average = weighted_average

        self.vmin, self.vmax = 0,1
        self._log_scale = True
        # all three of these will be reset by set_vmin_vmax

        self._setup_texture()
        self._setup_shader_module()
        self._setup_render_pipeline()

    @property
    def log_scale(self):
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value):
        old_value = self._log_scale
        self._log_scale = value
        if value != old_value:
            self._setup_shader_module()
            self._setup_render_pipeline()

    def _setup_shader_module(self, active_flags = None):
        shader_code = load_shader("colormap.wgsl")

        if active_flags is None:
            mode = "WEIGHTED_MEAN" if self._weighted_average else "DENSITY"
            active_flags = [mode]

        if self.log_scale:
            active_flags.append("LOG_SCALE")

        shader_code = preprocess_shader(shader_code, active_flags)

        self._shader = self._device.create_shader_module(code=shader_code, label="colormap")

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        """Map from raw image to the logical content that the colormap will use

        For example, drop unneeded channel if density is being displayed; perform ratio if column average is being
        displayed.
        """
        if self._weighted_average:
            numpy_image = numpy_image[..., 1] / numpy_image[..., 0]
        else:
            numpy_image = numpy_image[..., 0]

        return numpy_image

    def sph_raw_output_to_image(self, numpy_image: np.ndarray):
        """Map from SPH output to the colored image"""
        # check that input image has correct number of channels

        if len(numpy_image.shape) != 3:
            raise ValueError(f"Expected a 3D array, but got shape {numpy_image.shape}")
        if numpy_image.shape[2] != self.input_channels:
            raise ValueError(f"Expected the last dimension to have size {self.input_channels}, but got {numpy_image.shape[2]}")
        if numpy_image.dtype != np.float32:
            raise ValueError(f"Expected dtype to be np.float32, but got {numpy_image.dtype}")

        if self._output_format == wgpu.TextureFormat.rgba8unorm:
            output_dtype = np.uint8
        elif self._output_format == wgpu.TextureFormat.rgba32float:
            output_dtype = np.float32
        else:
            raise ValueError(f"Unsupported output format: {self._output_format}")

        # create a texture to hold the logical image:
        source_texture = self._device.create_texture(
            size=(numpy_image.shape[1], numpy_image.shape[0], 1),
            format=self._input_texture.format,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            label="colormap_input_texture"
        )

        destination_texture = self._device.create_texture(
            size=(numpy_image.shape[1], numpy_image.shape[0], 1),
            format=self._output_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            label="colormap_output_texture"
        )

        self.set_scaling(destination_texture, 1.0)

        # copy the image data to the texture
        self._device.queue.write_texture(
            {
                "texture": source_texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            numpy_image.tobytes(),
            {
                "bytes_per_row": 4 * self.input_channels * numpy_image.shape[1],
                "offset": 0,
            },
            (numpy_image.shape[1], numpy_image.shape[0], 1)
        )

        # create a bind group for the input texture
        bind_group = self._create_bind_group(source_texture)

        # create a render pass to apply the colormap
        command_encoder = self._device.create_command_encoder(label="colormap_command_encoder")

        self.encode_render_pass(command_encoder, destination_texture.create_view(), bind_group)

        # submit the command encoder
        self._device.queue.submit([command_encoder.finish()])

        # read back the result
        result = np.frombuffer(
            self._device.queue.read_texture({'texture': destination_texture, 'origin': (0, 0, 0)},
                                            {'bytes_per_row': 4 * output_dtype().itemsize * numpy_image.shape[1]},
                                            (numpy_image.shape[1], numpy_image.shape[0], 1)),
            dtype=output_dtype
        ).reshape((numpy_image.shape[0], numpy_image.shape[1], 4))

        return result



    def _setup_texture(self, num_points=config.COLORMAP_NUM_SAMPLES):
        rgba = self._generate_mapping_rgba_f32(num_points)

        size = rgba.shape[:2] + (1,)

        self._texture = self._device.create_texture(
            label="colormap_texture",
            size=size,
            dimension=self.map_dimension,
            format=wgpu.TextureFormat.rgba32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
        )

        self._device.queue.write_texture(
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            rgba.tobytes(),
            {
                "bytes_per_row": 4 * 4 * num_points,
                "offset": 0,
            },
            size
        )

    def _generate_mapping_rgba_f32(self, num_points):
        cmap = matplotlib.colormaps[self._colormap_name]
        rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
        return rgba

    def _setup_render_pipeline(self):
        self._parameter_buffer = self._device.create_buffer(size = self.parameter_dtype.itemsize,
                                                            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._bind_group_layout = \
            self._device.create_bind_group_layout(
                label="colormap_bind_group_layout",
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": self.map_dimension},
                    },
                    {
                        "binding": 3,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 4,
                        "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.uniform}
                    }
                ]
            )

        self._input_interpolation = self._device.create_sampler(label="colormap_sampler",
                                                                mag_filter=wgpu.FilterMode.linear, )

        self._bind_group = self._create_bind_group(self._input_texture)

        self._pipeline_layout = \
            self._device.create_pipeline_layout(
                label="colormap_pipeline_layout",
                bind_group_layouts=[self._bind_group_layout]
            )


        self._pipeline = \
            self._device.create_render_pipeline(
                layout=self._pipeline_layout,
                label="colormap_pipeline",
                vertex={
                    "module": self._shader,
                    "entry_point": "vertex_main",
                    "buffers": []
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_strip,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": self._shader,
                    "entry_point": self.fragment_shader,
                    "targets": [
                        {
                            "format": self._output_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            }
                        }
                    ]
                }
            )

    def _create_bind_group(self, input_texture: wgpu.GPUTexture):
        return self._device.create_bind_group(
            label="colormap_bind_group",
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0,
                 "resource": input_texture.create_view(),
                 },
                {"binding": 1,
                 "resource": self._input_interpolation,
                 },
                {"binding": 2,
                 "resource": self._texture.create_view(),
                 },
                {"binding": 3,
                 "resource": self._input_interpolation,
                 },
                {"binding": 4,
                 "resource": {"buffer": self._parameter_buffer,
                              "offset": 0,
                              "size": self._parameter_buffer.size}
                 }
            ]
        )

    def encode_render_pass(self, command_encoder, target_texture_view, bind_group = None):
        colormap_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_texture_view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        colormap_render_pass.set_pipeline(self._pipeline)
        colormap_render_pass.set_bind_group(0, bind_group or self._bind_group, [], 0, 99)
        colormap_render_pass.draw(4, 1, 0, 0)
        colormap_render_pass.end()

    def set_scaling(self, target_texture_view, scaling):
        self._update_parameter_buffer(*target_texture_view.size[:2], scaling)

    def get_ui_range(self):
        """Get a range for vmin->vmax suitable for user interface sliders"""
        if not hasattr(self, "_vals_min"):
            self.autorange_vmin_vmax()
        if self.log_scale:
            return self._log_vals_min, self._log_vals_max
        else:
            return self._vals_min, self._vals_max

    @classmethod
    def _finite_range(cls, values):
        valid = np.isfinite(values)
        valid_values = values[valid]
        if len(valid_values) > 0:
            return np.min(valid_values), np.max(valid_values)
        else:
            return np.nan, np.nan

    def autorange_vmin_vmax(self):
        """Set the vmin and vmax values for the colomap based on the most recent SPH render"""

        # This can and probably should be done on-GPU using a compute shader, but for now
        # we'll do it on the CPU
        vals = self._visualizer.get_sph_image().ravel()

        self._log_vals_min, self._log_vals_max = self._finite_range(np.log10(vals))
        self._vals_min, self._vals_max = self._finite_range(vals)

        if self._log_vals_max == self._log_vals_min:
            self._log_vals_max += 1.0
            self._log_vals_min -= 1.0

        if self._vals_max == self._vals_min:
            self._vals_max += 1.0
            self._vals_min -= 1.0

        if (vals<0).any():
            self.log_scale = False
        else:
            self.log_scale = True
        # NB above switching of log scale will automatically rebuild the pipeline if needed

        if self.log_scale:
            vals = np.log10(vals)

        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self.vmin, self.vmax = np.percentile(vals, self.percentile_scaling)
        elif len(vals)>2:
            self.vmin, self.vmax = np.min(vals), np.max(vals)
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmin, self.vmax = 0.0, 1.0

        self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)
        logger.info(f"Autoscale: log_scale={self.log_scale}, vmin={self.vmin}, vmax={self.vmax}")


    def _update_parameter_buffer(self, width, height, mass_scale):
        parameters = np.zeros((), dtype=self.parameter_dtype)
        parameters["vmin"] = self.vmin
        parameters["vmax"] = self.vmax
        if self.log_scale:
            parameters["vmin"] -= np.log10(mass_scale)
            parameters["vmax"] -= np.log10(mass_scale)
        else:
            parameters["vmin"] /= mass_scale
            parameters["vmax"] /= mass_scale

        parameters["window_aspect_ratio"] = float(width)/height
        parameters["gamma"] = self.gamma if hasattr(self, "gamma") else 1.0

        parameters["density_vmin"] = self.density_vmin - np.log10(mass_scale) if hasattr(self, "density_vmin") else 0.0
        parameters["density_vmax"] = self.density_vmax - np.log10(mass_scale) if hasattr(self, "density_vmax") else 1.0

        self._device.queue.write_buffer(self._parameter_buffer, 0, parameters)

class RGBColormap(Colormap):
    input_channels = 3
    fragment_shader = "fragment_main_tri"
    max_percentile = 99.9
    dynamic_range = 3.0

    _sterrad_to_arcsec2 = 2.3504430539466191e-11

    def __init__(self, visualizer: Visualizer, weighted_average: bool = False):
        self._gamma = 3.0
        super().__init__(visualizer, weighted_average)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)

    @classmethod
    def _log_output_to_mag_per_arcsec2(cls, val):
        return -2.5 * (val + np.log10(cls._sterrad_to_arcsec2) - 4) # +4 for (10pc->kpc)^2

    @classmethod
    def _mag_per_arcsec2_to_log_output(cls, val):
        return val/-2.5 + 4 - np.log10(cls._sterrad_to_arcsec2)

    @property
    def max_mag(self):
        return self._log_output_to_mag_per_arcsec2(self.vmin)

    @max_mag.setter
    def max_mag(self, value):
        self.vmin = self._mag_per_arcsec2_to_log_output(value)
        self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)

    @property
    def min_mag(self):
        return self._log_output_to_mag_per_arcsec2(self.vmax)

    @min_mag.setter
    def min_mag(self, value):
        self.vmax = self._mag_per_arcsec2_to_log_output(value)
        self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)

    def autorange_vmin_vmax(self):
        vals = self._visualizer.get_sph_image().ravel()

        self.log_scale = True

        if self.log_scale:
            vals = np.log10(vals)

        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self.vmax = np.percentile(vals, self.max_percentile)
        elif len(vals)>2:
            self.vmax = np.max(vals)
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmax = 1.0

        self.vmin = self.vmax - self.dynamic_range

        self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)
        logger.info(f"vmin={self.vmin}, vmax={self.vmax}")

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        """Map from raw image to the logical content that the colormap will use

        For example, drop unneeded channel if density is being displayed; perform ratio if column average is being
        displayed.
        """
        return numpy_image[..., :3]

class RGBHDRColormap(RGBColormap):
    max_percentile = 99.0
    dynamic_range = 2.5 # nb this is the SDR-equivalent dynamic range -- HDR exceeds this.

class BivariateColormap(Colormap):
    map_dimension = wgpu.TextureViewDimension.d2

    def __init__(self, visualizer: Visualizer):
        super().__init__(visualizer, True)
        self.density_vmin = -9.4
        self.density_vmax = -5.6

    def _setup_shader_module(self, active_flags=None):
        assert active_flags is None
        super()._setup_shader_module(["BIVARIATE"])


    def _generate_mapping_rgba_f32(self, num_points):
        cmap = matplotlib.colormaps[self._colormap_name]

        rgba = np.ones((num_points, num_points, 4), dtype=np.float32)
        rgba[:, :, :] = cmap(np.linspace(0.001, 0.999, num_points))[:, np.newaxis, :]

        hsv = matplotlib.colors.rgb_to_hsv(rgba[..., :3])
        hsv[..., 2] = np.linspace(0.001, 0.999, num_points)[np.newaxis, :]
        rgba[..., :3] =matplotlib.colors.hsv_to_rgb(hsv)

        return rgba