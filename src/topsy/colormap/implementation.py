from __future__ import annotations

import logging
import numpy as np
import wgpu
import matplotlib

from .. import config

from ..drawreason import DrawReason
from ..util import load_shader, preprocess_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ColormapBase:
    _default_params = {}

    def __init__(self, device: wgpu.GPUDevice, input_texture: wgpu.GPUTexture, output_format: wgpu.TextureFormat, params: dict):
        self._device = device
        self._input_texture = input_texture
        self._output_format = output_format
        self._params = self._default_params | params

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        """Check if the colormap accepts the given parameters"""
        return False

    def update_parameters(self, parameters: dict):
        """Update the colormap parameters"""
        if not self.accepts_parameters(self._params | parameters):
            raise ValueError(f"Colormap {self.__class__.__name__} does not accept parameter update: {parameters}")
        self._params.update(parameters)

    def get_parameter(self, name: str):
        """Get a parameter value by name"""
        return self._params.get(name, None)

    def get_parameters(self) -> dict:
        """Get all parameters as a dictionary"""
        return self._params.copy()

    def encode_render_pass(self, command_encoder, target_texture_view, bind_group = None):
        """Encode the render pass for the colormap"""
        raise NotImplementedError("Subclasses must implement encode_render_pass")

    def set_scaling(self, output_width, output_height, mass_scaling):
        """Set the scaling parameters for the colormap"""
        raise NotImplementedError("Subclasses must implement set_scaling")

class NoColormap(ColormapBase):
    """A colormap that does nothing, used when the colormap has not yet been selected"""

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        return parameters.get("type", None) == "none"


class Colormap(ColormapBase):
    input_channels = 2
    fragment_shader = "fragment_main"
    percentile_scaling = [1.0, 99.9]
    map_dimension = wgpu.TextureViewDimension.d1

    _default_params = {'colormap_name': 'viridis', 'vmin': 0.0, 'vmax': 1.0, 'log': True, 'weighted_average': False}

    shader_parameter_dtype = np.dtype([("vmin", np.float32, (1,)),
                                       ("vmax", np.float32, (1,)),
                                       ("density_vmin", np.float32, (1,)),
                                       ("density_vmax", np.float32, (1,)),
                                       ("window_aspect_ratio", np.float32, (1,)),
                                       ("gamma", np.float32, (1,))])

    def __init__(self, device, input_texture, output_format, params):
        super().__init__(device, input_texture, output_format, params)
        self._setup_map_texture()
        self._setup_shader_module()
        self._setup_render_pipeline()

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        return parameters.get("type", None) == "density"

    def update_parameters(self, parameters: dict):
        parameters_before = self.get_parameters()
        super().update_parameters(parameters)
        parameters_after = self.get_parameters()
        if parameters_before['log'] != parameters_after['log'] or parameters_before['weighted_average'] != parameters_after['weighted_average']:
            self._setup_shader_module()
            self._setup_render_pipeline()
        if parameters_before['colormap_name'] != parameters_after['colormap_name']:
            self._setup_map_texture()
            self._setup_render_pipeline()

    def _setup_shader_module(self, active_flags = None):
        shader_code = load_shader("colormap.wgsl")

        if active_flags is None:
            active_flags = []

        mode = "WEIGHTED_MEAN" if self._params.get('weighted_average', False) else "DENSITY"
        active_flags.append(mode)

        if self._params['log']:
            active_flags.append("LOG_SCALE")

        shader_code = preprocess_shader(shader_code, active_flags)

        self._shader = self._device.create_shader_module(code=shader_code, label="colormap")

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        """Map from raw image to the logical content that the colormap will use

        For example, drop unneeded channel if density is being displayed; perform ratio if column average is being
        displayed.
        """
        if self._params['weighted_average']:
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

        self.set_scaling(*destination_texture.size[:2], 1.0)

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



    def _setup_map_texture(self, num_points=config.COLORMAP_NUM_SAMPLES):
        rgba = self._generate_mapping_rgba_f32(num_points)

        dim = len(rgba.shape) - 1
        size = rgba.shape[:dim] + (1,)

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
        cmap = matplotlib.colormaps[self._params.get('colormap_name', config.DEFAULT_COLORMAP)]
        rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)
        return rgba

    def _setup_render_pipeline(self):
        self._parameter_buffer = self._device.create_buffer(size = self.shader_parameter_dtype.itemsize,
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

    def set_scaling(self, width, height, scaling):
        self._update_parameter_buffer(width, height, scaling)

    @classmethod
    def _finite_range(cls, values):
        valid = np.isfinite(values)
        valid_values = values[valid]
        if len(valid_values) > 0:
            return np.min(valid_values), np.max(valid_values)
        else:
            return np.nan, np.nan

    def autorange_vmin_vmax(self, vals):
        """Set the vmin and vmax values for the colomap based on the most recent SPH render"""

        # This can and probably should be done on-GPU using a compute shader, but for now
        # we'll do it on the CPU

        vals = self.sph_raw_output_to_content(vals).ravel()
        self._autorange_using_values(vals)

    def _autorange_using_values(self, vals):
        new_params = {}

        log_vals_min, log_vals_max = self._finite_range(np.log10(vals))
        vals_min, vals_max = self._finite_range(vals)
        if log_vals_max == log_vals_min:
            log_vals_max += 1.0
            log_vals_min -= 1.0
        if vals_max == vals_min:
            vals_max += 1.0
            vals_min -= 1.0

        new_params['ui_range_linear'] = (vals_min, vals_max)
        new_params['ui_range_log'] = (log_vals_min, log_vals_max)

        if (vals < 0).any():
            new_params['log'] = False
        else:
            new_params['log'] = True

        if new_params['log']:
            vals = np.log10(vals)
        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self._params['vmin'], self._params['vmax'] = np.percentile(vals, self.percentile_scaling)
        elif len(vals) > 2:
            self._params['vmin'], self._params['vmax'] = np.min(vals), np.max(vals)
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self._params['vmin'], self._params['vmax'] = 0.0, 1.0

        self.update_parameters(new_params)

        logger.info(f"Autoscale: log_scale={self._params['log']}, vmin={self._params['vmin']}, vmax={self._params['vmax']}")

    def _update_parameter_buffer(self, width, height, mass_scale):
        parameters = np.zeros((), dtype=self.shader_parameter_dtype)
        d_vmin = self._params.get('density_vmin', 0.0)
        if d_vmin is None:
            d_vmin = 0.0
        d_vmax = self._params.get('density_vmax', 1.0)
        if d_vmax is None:
            d_vmax = 1.0
        parameters["density_vmin"] = d_vmin - np.log10(mass_scale)
        parameters["density_vmax"] = d_vmax - np.log10(mass_scale)

        if self._params.get('weighted_average', False):
            mass_scale = 1.0

        parameters["vmin"] = self._params['vmin']
        parameters["vmax"] = self._params['vmax']
        if self._params['log']:
            parameters["vmin"] -= np.log10(mass_scale)
            parameters["vmax"] -= np.log10(mass_scale)
        else:
            parameters["vmin"] /= mass_scale
            parameters["vmax"] /= mass_scale

        parameters["window_aspect_ratio"] = float(width)/height
        parameters["gamma"] = self._params.get('gamma', 1.0)

        self._device.queue.write_buffer(self._parameter_buffer, 0, parameters)


class RGBColormap(Colormap):
    input_channels = 3
    fragment_shader = "fragment_main_tri"
    max_percentile = 99.9
    dynamic_range = 3.0

    _sterrad_to_arcsec2 = 2.3504430539466191e-11

    _default_params = {'vmin': 0.0, 'vmax': 1.0, 'log': True, 'gamma': 1.0}

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        parameters = cls._default_params | parameters
        return (parameters.get("type", None) == "rgb" and (not parameters['hdr']) and parameters['log'])

    @classmethod
    def _log_output_to_mag_per_arcsec2(cls, val):
        if val is not None:
            return -2.5 * (val + np.log10(cls._sterrad_to_arcsec2) - 4) # +4 for (10pc->kpc)^2
        else:
            return None

    @classmethod
    def _mag_per_arcsec2_to_log_output(cls, val):
        if val is not None:
            return val/-2.5 + 4 - np.log10(cls._sterrad_to_arcsec2)
        else:
            return None

    def get_parameters(self) -> dict:
        """Get all parameters as a dictionary"""
        params = super().get_parameters()
        params['min_mag'] = self._log_output_to_mag_per_arcsec2(params['vmax'])
        params['max_mag'] = self._log_output_to_mag_per_arcsec2(params['vmin'])
        return params

    def get_parameter(self, name: str):
        if name == "min_mag":
            return self._log_output_to_mag_per_arcsec2(self.get_parameter("vmax"))
        elif name == "max_mag":
            return self._log_output_to_mag_per_arcsec2(self.get_parameter("vmin"))
        else:
            return super().get_parameter(name)

    def update_parameters(self, parameters: dict):
        if "min_mag" in parameters:
            parameters['vmax'] = self._mag_per_arcsec2_to_log_output(parameters['min_mag'])
        if "max_mag" in parameters:
            parameters['vmin'] = self._mag_per_arcsec2_to_log_output(parameters['max_mag'])

        # NB we are skipping a level in the hierarchy here, which shows that we shouldn't
        # really be basing off Colormap, but rather ColormapBase. TODO.
        ColormapBase.update_parameters(self, parameters)

    def autorange_vmin_vmax(self, vals):
        vals = vals.ravel()

        self.log_scale = True
        vals = np.log10(vals)

        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self._params['vmax'] = np.percentile(vals, self.max_percentile)
        elif len(vals)>2:
            self._params['vmax'] = np.max(vals)
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self._params['vmax'] = 1.0

        self._params['vmin'] = self._params['vmax'] - self.dynamic_range

        logger.info(f"vmin={self._params['vmin']}, vmax={self._params['vmax']}")

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        """Map from raw image to the logical content that the colormap will use

        For example, drop unneeded channel if density is being displayed; perform ratio if column average is being
        displayed.
        """
        return numpy_image[..., :3]



class RGBHDRColormap(RGBColormap):
    max_percentile = 99.0
    dynamic_range = 2.5 # nb this is the SDR-equivalent dynamic range -- HDR exceeds this.

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        parameters = cls._default_params | parameters
        return (parameters.get("type", None) == "rgb" and parameters['hdr'] and parameters['log'])


class BivariateColormap(Colormap):
    default_quantity_name = 'rho'
    map_dimension = wgpu.TextureViewDimension.d2

    _default_params = Colormap._default_params | {'density_vmin': 0.0, 'density_vmax': 1.0,
                                                  'ui_range_density': (0.0, 1.0)}

    @classmethod
    def accepts_parameters(cls, parameters: dict) -> bool:
        return parameters.get("type", None) == "bivariate" and (not parameters.get("hdr", False))

    def _setup_shader_module(self, active_flags=None):
        assert active_flags is None
        super()._setup_shader_module(["BIVARIATE"])

    def sph_raw_output_to_content(self, numpy_image: np.ndarray):
        ret_image = numpy_image.copy()
        if self._params['weighted_average']:
            ret_image[..., 1] /= ret_image[..., 0]
        else:
            ret_image[..., 1] = ret_image[..., 0]
        return ret_image

    def autorange_vmin_vmax(self, vals):
        vals = self.sph_raw_output_to_content(vals)
        den_vals = vals[..., 0].ravel()
        den_vals = np.log10(den_vals)
        den_vals = den_vals[np.isfinite(den_vals)]
        density_vmin, density_vmax = np.percentile(den_vals, self.percentile_scaling)
        density_ui_min, density_ui_max = self._finite_range(den_vals)
        self.update_parameters({
            'density_vmin': density_vmin,
            'density_vmax': density_vmax,
            'ui_range_density': (density_ui_min, density_ui_max),
        })
        self._autorange_using_values(vals[..., 1])

    def _generate_mapping_rgba_f32(self, num_points):
        cmap = matplotlib.colormaps[self._params['colormap_name']]

        rgba = np.ones((num_points, num_points, 4), dtype=np.float32)
        rgba[:, :, :] = cmap(np.linspace(0.001, 0.999, num_points))[:, np.newaxis, :]

        hsv = matplotlib.colors.rgb_to_hsv(rgba[..., :3])
        hsv[..., 2] = np.linspace(0.001, 0.999, num_points)[np.newaxis, :]

        reduce_saturation = np.ones(num_points)
        reduce_saturation[3*num_points//4:] = np.linspace(1.0, 0.0, num_points//4)

        hsv[..., 1] *= reduce_saturation[np.newaxis, :]

        rgba[..., :3] =matplotlib.colors.hsv_to_rgb(hsv)

        return rgba