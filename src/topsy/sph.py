from __future__ import annotations

import numpy as np
import wgpu
import pynbody

from logging import getLogger

from .util import load_shader, preprocess_shader
from . import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

logger = getLogger(__name__)

class SPH:
    render_format = wgpu.TextureFormat.rg32float
    _nchannels_input = 2
    _nchannels_output = 2
    _output_dtype = np.float32

    def __init__(self, visualizer: Visualizer, render_resolution,
                 wrapping = False):
        self._visualizer = visualizer

        logger.info(f"Creating SPH renderer with resolution {render_resolution}")
        self._render_texture = visualizer.device.create_texture(
            size=(render_resolution, render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=self.render_format,
            label="sph_render_texture",
        )

        self._device = visualizer.device
        self._wrapping = wrapping
        self._kernel = None

        self._setup_shader_module()
        self._setup_transform_buffer()
        self._setup_kernel_texture()
        self._setup_render_pipeline()

        self.scale = config.DEFAULT_SCALE
        self.min_pixels = 0.0    # minimum size of softening, in pixels, to qualify for rendering
        self.max_pixels = np.inf # maximum size of softening, in pixels, to qualify for rendering
        self.downsample_factor = 1 # number of particles to increment
        self.downsample_offset = 0  # offset to start skipping particles
        self.rotation_matrix = np.eye(3)
        self.position_offset = np.zeros(3)

    def get_output_texture(self) -> wgpu.Texture:
        return self._render_texture


    def _setup_shader_module(self):
        wrap_flag = "WRAPPING" if self._wrapping else "NO_WRAPPING"
        type_flag = "CHANNELED" if self._nchannels_input == 3 else "WEIGHTED"
        code = preprocess_shader(load_shader("sph.wgsl"), [wrap_flag, type_flag])

        self._shader = self._device.create_shader_module(code=code, label="sph")

    def _setup_transform_buffer(self):
        self._transform_buffer = self._device.create_buffer(
            size=(4*4*4) + 4 + 8 + 8 + 12,  # 4x4 float32 matrix + one float32 scale + float32 min,max size + int32 x 2 + padding
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def _setup_render_pipeline(self):
        self._bind_group_layout = \
            self._device.create_bind_group_layout(
                label="sph_bind_group_layout",
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.uniform},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                ]
            )

        self._bind_group = \
            self._device.create_bind_group(
                label="sph_bind_group",
                layout=self._bind_group_layout,
                entries=[
                    {"binding": 0,
                     "resource": {
                         "buffer": self._transform_buffer,
                         "offset": 0,
                         "size": self._transform_buffer.size
                      }
                     },
                    {"binding": 1,
                        "resource": self._kernel_texture.create_view(),
                    },
                    {"binding": 2,
                        "resource": self._kernel_sampler,
                    }
                ]
            )

        self._pipeline_layout = \
            self._device.create_pipeline_layout(
                label="sph_pipeline_layout",
                bind_group_layouts=[self._bind_group_layout]
            )

        vertex_format = wgpu.VertexFormat.float32x3

        if self._nchannels_input == 3 :
            channel_buffers = [{
                "array_stride": 12,
                "step_mode": wgpu.VertexStepMode.instance,
                "attributes": [
                    {
                        "format": vertex_format,
                        "offset": 0,
                        "shader_location": 1,
                    }
                ]
            } ]
        else:
            channel_buffers = [ {
                                "array_stride": 4,
                                "step_mode": wgpu.VertexStepMode.instance,
                                "attributes": [
                                    {
                                        "format": wgpu.VertexFormat.float32,
                                        "offset": 0,
                                        "shader_location": i+1,
                                    }
                                ]
                            } for i in range(self._nchannels_input)]

        self._render_pipeline = \
            self._device.create_render_pipeline(
                layout=self._pipeline_layout,
                label="sph_render_pipeline",
                vertex = {
                    "module": self._shader,
                    "entry_point": "vertex_main",
                    "buffers": [
                        {
                            "array_stride": 16,
                            "step_mode": wgpu.VertexStepMode.instance,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 0,
                                    "shader_location": 0,
                                }
                            ]
                        },


                    ] + channel_buffers
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": self._shader,
                    "entry_point": "fragment_main",
                    "targets": [
                        {
                            "format": self.render_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.one,
                                    "operation": wgpu.BlendOperation.add,
                                },
                              }
                        }
                    ]
                }
            )

    def _update_transform_buffer(self):
        model_displace = np.array([[1.0, 0, 0, self.position_offset[0]],
                                   [0, 1.0, 0, self.position_offset[1]],
                                   [0, 0, 1.0, self.position_offset[2]],
                                   [0, 0, 0.0, 1.0]])

        # self._transform is the transformation around the origin (fine for opengl)
        # but in webgpu, the clip space in the z direction is [0,1]
        # so we need a matrix that brings z=0. to z=0.5 and squishes the z direction
        # so that the clipping is the same in all dimensions
        clipcoord_displace = np.array([[1.0, 0, 0, 0.0],
                             [0, 1.0, 0, 0.0],
                             [0, 0, 0.5, 0.5],
                             [0, 0, 0.0, 1.0]])
        transform = np.zeros((4, 4))

        transform[:3,:3] = self.rotation_matrix
        rotation_and_scaling = transform / self.scale

        rotation_and_scaling[3, 3] = 1.0  # w should be unchanged after transform

        scaled_displaced_transform = (clipcoord_displace @ rotation_and_scaling @ model_displace).T
        transform_params_dtype = [("transform", np.float32, (4, 4)),
                                  ("scale_factor", np.float32, (1,)),
                                  ("min_max_size", np.float32, (2,)),
                                  ("downsample_factor", np.uint32, (1,)),
                                  ("downsample_offset", np.uint32, (1,)),
                                  ("boxsize_by_2_clipspace", np.float32, (1,)),
                                  ("padding", np.int32, (1,))]
        transform_params = np.zeros((), dtype=transform_params_dtype)
        transform_params["transform"] = scaled_displaced_transform
        transform_params["scale_factor"] = 1. / self.scale
        # transform_params["mass_scale"] = self._get_mass_scale()
        # logger.info(f"downsample_factor: {self.downsample_factor}; mass_scale: {transform_params['mass_scale']}")
        transform_params["boxsize_by_2_clipspace"] = 0.5 * \
                                                     self._visualizer.periodicity_scale / self.scale

        resolution = self._render_texture.width
        assert resolution == self._render_texture.height

        # min_max_size to be sent in viewport coordinates
        transform_params["min_max_size"] = (2.*self.min_pixels/resolution, 2.*self.max_pixels/resolution)
        transform_params["downsample_factor"] = self.downsample_factor
        transform_params["downsample_offset"] = self.downsample_offset

        self.last_transform_params = transform_params

        self._device.queue.write_buffer(self._transform_buffer, 0, transform_params)
        self.last_render_mass_scale = self.downsample_factor

    def encode_render_pass(self, command_encoder):
        self._update_transform_buffer()
        view: wgpu.GPUTextureView = self._render_texture.create_view()
        sph_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        sph_render_pass.set_pipeline(self._render_pipeline)

        num_particles_to_render = len(self._visualizer.data_loader)//self.downsample_factor
        start_particles = num_particles_to_render * self.downsample_offset

        sph_render_pass.set_vertex_buffer(0, self._visualizer.data_loader.get_pos_smooth_buffer(),
                                          offset=start_particles*16)
        if self._nchannels_input == 2:
            sph_render_pass.set_vertex_buffer(1, self._visualizer.data_loader.get_mass_buffer(),
                                              offset=start_particles*4)
            sph_render_pass.set_vertex_buffer(2, self._visualizer.data_loader.get_quantity_buffer(),
                                              offset=start_particles*4)
        elif self._nchannels_input == 3:
            sph_render_pass.set_vertex_buffer(1, self._visualizer.data_loader.get_rgb_masses_buffer(),
                                              offset=start_particles*12)
        else:
            raise ValueError("Unexpected number of channels")

        sph_render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        sph_render_pass.draw(6, num_particles_to_render, 0, 0)
        sph_render_pass.end()


    def _get_kernel_at_resolution(self, n_samples):
        if self._kernel is None:
            try:
                self._kernel = pynbody.sph.Kernel2D()
            except AttributeError:
                # pynbody v2:
                self._kernel = pynbody.sph.kernels.Kernel2D()

        # sph kernel is sampled at the centre of the pixels, and the full grid ranges from -2 to 2.
        # thus the left hand most pixel is at -2+2/n_samples, and the right hand most pixel is at 2-2/n_samples.
        pixel_centres = np.linspace(-2+2./n_samples, 2-2./n_samples, n_samples)
        x, y = np.meshgrid(pixel_centres, pixel_centres)
        distance = np.sqrt(x ** 2 + y ** 2)

        # TODO: the below could easily be optimized
        kernel_im = np.array([self._kernel.get_value(d) for d in distance.flatten()]).reshape(n_samples, n_samples)

        # make kernel explicitly mass conserving; naive pixelization makes it not automatically do this.
        # It should be normalized such that the integral over the kernel is 1/h^2. We have h=1 here, and the
        # full width is 4h, so the width of a pixel is dx=4/n_samples. So we need to multiply by dx^2=(n_samples/4)^2.
        # This results in a correction of a few percent, typically; not huge but not negligible either.
        #
        # (Obviously h!=1 generally, so the h^2 normalization occurs within the shader later.)
        kernel_im *= (n_samples / 4) ** 2 / kernel_im.sum()

        return kernel_im

    def _setup_kernel_texture(self, n_samples=64, n_mip_levels = 4):
        if hasattr(SPH, "_kernel_texture"):
            # we only do this once, even if multiple SPH objects (i.e. multi-resolution) is in play
            return

        SPH._kernel_texture = self._device.create_texture(
            label="kernel_texture",
            size=(n_samples, n_samples, 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.r32float,
            mip_level_count=n_mip_levels,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )


        for i in range(0, n_mip_levels):
            self._device.queue.write_texture(
                {
                    "texture": self._kernel_texture,
                    "mip_level": i,
                    "origin": (0, 0, 0),
                },
                self._get_kernel_at_resolution(n_samples//2**i).astype(np.float32).tobytes(),
                {
                    "offset": 0,
                    "bytes_per_row": 4 * n_samples // 2**i,
                },
                (n_samples//2**i, n_samples//2**i, 1)
            )


        SPH._kernel_sampler = self._device.create_sampler(label="kernel_sampler",
                                                           mag_filter=wgpu.FilterMode.linear, )

class RGBSPH(SPH):
    render_format = wgpu.TextureFormat.rgba32float
    _nchannels_input = 3
    _nchannels_output = 4
    _output_dtype = np.float32
