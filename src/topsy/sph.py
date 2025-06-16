from __future__ import annotations

import copy
import numpy as np
import wgpu
import pynbody

from logging import getLogger

from .util import load_shader, preprocess_shader, TimeGpuOperation
from .drawreason import DrawReason

from . import config, performance

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

logger = getLogger(__name__)


class SPH:
    render_format = wgpu.TextureFormat.rg32float
    _nchannels_input = 2
    _nchannels_output = 2
    _output_dtype = np.float32
    _buffer_name = "mass_and_quantity" # as defined in particle_buffers.py
    _vertex_name = "vertex_weighting" # as defined in sph.wgsl
    _fragment_name = "fragment_weighting"

    def __init__(self, visualizer: Visualizer, render_resolution,
                 wrapping = False, share_render_progression=None):
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

        self._render_resolution = render_resolution

        self._device : wgpu.GPUDevice = visualizer.device
        self._wrapping = wrapping
        self._kernel = None

        self._render_timer = TimeGpuOperation(self._device)

        if share_render_progression is not None:
            self._render_progression = share_render_progression
        else:
            self._render_progression = self._visualizer.data_loader.get_render_progression()

        self._setup_shader_module()
        self._setup_transform_buffer()
        self._setup_kernel_texture()
        self._setup_render_pipeline()

        self.scale = config.DEFAULT_SCALE
        self.min_pixels = 0.0    # minimum size of softening, in pixels, to qualify for rendering
        self.max_pixels = np.inf # maximum size of softening, in pixels, to qualify for rendering
        self.rotation_matrix = np.eye(3)
        self.position_offset = np.zeros(3)
        self.has_rendered = False

    def _get_depth_renderer(self) -> SPH:
        """Returns a SPH renderer that will generate the depth in the scene"""
        renderer = DepthSPH(self._visualizer, self._render_resolution, wrapping=self._wrapping,
                   share_render_progression=copy.copy(self._render_progression))
        renderer.rotation_matrix = self.rotation_matrix
        renderer.position_offset = self.position_offset
        renderer.scale = self.scale
        return renderer


    def get_depth_image(self, depth_renderer_reason=DrawReason.CHANGE) -> np.ndarray:
        """Produces and returns the weighted depth image in the scene, used for finding points of interest in the UI

        A renderer reason may be passed to force different quality settings. For most purposes, a rough (real-time)
        render is sufficient, so DrawReason.CHANGE is a good choice. However, real-time rendering on test machines
        may be slow, which leads DrawReason.CHANGE renders to use only a small fraction of particles. Therefore, for
        testing purposes DrawReason.EXPORT may be required.
        """
        depth_renderer = self._get_depth_renderer()
        depth_renderer.render(depth_renderer_reason) # CHANGE should normally be good enough; EXPORT for reliability

        image = depth_renderer.get_image()

        depth_viewport = image[..., 1] / image[..., 0]

        # transform from viewport to simulation units
        return (depth_viewport - 0.5)*self.scale*2.0

    def get_image(self) -> np.ndarray:
        """Reads and returns the last rendered SPH image.

        If the current SPH output is invalid, this triggers an EXPORT-quality render. If you don't want this to happen,
        you should call your own CHANGED render for example.
        """

        if not self.has_rendered:
            logger.info("Export-quality render has been triggered, because no render has been done yet.")
            self.render(DrawReason.EXPORT)

        np_dtype = self._output_dtype
        bytes_per_pixel = self._nchannels_output * np.dtype(np_dtype).itemsize
        im = self._device.queue.read_texture({'texture': self.get_output_texture(), 'origin': (0, 0, 0)},
                                             {'bytes_per_row': bytes_per_pixel * self._render_resolution},
                                             (self._render_resolution, self._render_resolution, 1))
        np_im = np.frombuffer(im, dtype=np_dtype).reshape((self._render_resolution, self._render_resolution,
                                                           self._nchannels_output))

        return np_im * self.last_render_mass_scale

    def get_output_texture(self) -> wgpu.Texture:
        return self._render_texture


    def _setup_shader_module(self):
        code = load_shader("sph.wgsl")
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

        self._render_pipeline = \
            self._device.create_render_pipeline(
                layout=self._pipeline_layout,
                label="sph_render_pipeline",
                vertex = {
                    "module": self._shader,
                    "entry_point": self._vertex_name,
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
                    "entry_point": self._fragment_name,
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
                                  ("boxsize_by_2_clipspace", np.float32, (1,)),
                                  ("padding", np.int32, (1,))]
        transform_params = np.zeros((), dtype=transform_params_dtype)
        transform_params["transform"] = scaled_displaced_transform
        transform_params["scale_factor"] = 1. / self.scale
        if self._visualizer.periodicity_scale is not None:
            transform_params["boxsize_by_2_clipspace"] = 0.5 * \
                                                         self._visualizer.periodicity_scale / self.scale
        else:
            transform_params["boxsize_by_2_clipspace"] = 0.0

        resolution = self._render_texture.width
        assert resolution == self._render_texture.height

        # min_max_size to be sent in viewport coordinates
        transform_params["min_max_size"] = (2.*self.min_pixels/resolution, 2.*self.max_pixels/resolution)

        self.last_transform_params = transform_params

        self._device.queue.write_buffer(self._transform_buffer, 0, transform_params)

    def invalidate(self, draw_reason=DrawReason.CHANGE):
        """Invalidates the current render, so that an attempt to get the current image will fail."""
        if draw_reason != DrawReason.REFINE and draw_reason != DrawReason.PRESENTATION_CHANGE:
            self.has_rendered = False

    def render(self, draw_reason=DrawReason.CHANGE):
        performance.signposter.emit_event("Start SPH render")

        if draw_reason == DrawReason.PRESENTATION_CHANGE:
            return

        if draw_reason != DrawReason.REFINE:
            self._render_progression.select_sphere(-self.position_offset, self.scale*1.2)
            self._update_transform_buffer()

        clear = self._render_progression.start_frame(draw_reason)

        while block := self._render_progression.get_block(self._render_timer.total_time_in_frame()):
            encoded_render_pass = self.encode_render_pass(clear=clear)
            self._visualizer.particle_buffers.update_particle_ranges(*block)
            with self._render_timer:
                # we only time this part, because otherwise the timing is very unstable in interactive
                # use where most of the time we are not updating particle ranges
                self._device.queue.submit([encoded_render_pass])
            self._render_progression.end_block(self._render_timer.total_time_in_frame())
            clear = False

        self._render_timer.end_frame()

        self.last_render_mass_scale = self._render_progression.end_frame_get_scalefactor()
        self.last_render_fps = 1.0 / self._render_timer.running_mean_duration
        self.has_rendered = True

    def needs_refine(self):
        return self._render_progression.needs_refine()

    def encode_render_pass(self, clear=True) -> wgpu.GPUCommandBuffer:
        command_encoder: wgpu.GPUCommandEncoder = self._device.create_command_encoder(label='sph_render')
        view: wgpu.GPUTextureView = self._render_texture.create_view()
        sph_render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        sph_render_pass.set_pipeline(self._render_pipeline)

        vb_assignment = ['pos_smooth', self._buffer_name]

        self._visualizer.particle_buffers.specify_vertex_buffer_assignment(vb_assignment)
        sph_render_pass.set_bind_group(0, self._bind_group, [],
                                       0, 99)

        self._visualizer.particle_buffers.issue_draw_indirect(sph_render_pass)
        sph_render_pass.end()

        return command_encoder.finish()

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

        # The below could easily be optimized but doesn't seem worth it
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
            # As things stand, the kernel texture is actually shared between instances of SPH for efficiency.
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

class BivariateSPH(SPH):
    """Renders a density, mass-weighted-mean pair"""


class RGBSPH(SPH):
    render_format = wgpu.TextureFormat.rgba32float
    _buffer_name = 'rgb'
    _nchannels_input = 3
    _nchannels_output = 4
    _output_dtype = np.float32
    _vertex_name = "vertex_rgb"
    _fragment_name = "fragment_rgb"



class DepthSPH(SPH):
    """Renders a map of the depth of the particles in the scene."""

    _vertex_name = "vertex_depth"

