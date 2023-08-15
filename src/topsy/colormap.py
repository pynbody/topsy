from __future__ import annotations

import numpy as np
import wgpu
import matplotlib

from . import config

from .util import load_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer_wgpu import Visualizer

class Colormap:

    def __init__(self, visualizer: Visualizer):
        self._visualizer = visualizer
        self._device = visualizer.device
        self._colormap_name = visualizer.colormap_name
        self._input_texture = visualizer.render_texture
        self._output_format = visualizer.canvas_format

        self.vmin, self.vmax = 0,1

        self._setup_texture()
        self._setup_shader_module()
        self._setup_render_pipeline()
    def _setup_shader_module(self):
        self._shader = self._device.create_shader_module(code=load_shader("colormap.wgsl"), label="colormap")

    def _setup_texture(self, num_points=config.COLORMAP_NUM_SAMPLES):
        cmap = matplotlib.colormaps[self._colormap_name]
        rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32)

        self._texture = self._device.create_texture(
            label="colormap_texture",
            size=(num_points, 1, 1),
            dimension=wgpu.TextureDimension.d1,
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
            (num_points, 1, 1)
        )

    def _setup_render_pipeline(self):
        self._vmin_vmax_buffer = self._device.create_buffer(size =4 * 2,
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
                                    "view_dimension": wgpu.TextureViewDimension.d1},
                    },
                    {
                        "binding": 3,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 4,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "buffer": {"type": wgpu.BufferBindingType.uniform}
                    }
                ]
            )

        self._input_interpolation = self._device.create_sampler(label="colormap_sampler",
                                                                mag_filter=wgpu.FilterMode.linear, )

        self._bind_group = \
            self._device.create_bind_group(
                label="colormap_bind_group",
                layout=self._bind_group_layout,
                entries=[
                    {"binding": 0,
                     "resource": self._input_texture.create_view(),
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
                        "resource": {"buffer": self._vmin_vmax_buffer,
                                     "offset": 0,
                                     "size": self._vmin_vmax_buffer.size}
                     }
                ]
            )

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
                    "entry_point": "fragment_main",
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

    def encode_render_pass(self, command_encoder):
        display_texture = self._visualizer.context.get_current_texture()
        colormap_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": display_texture,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        colormap_render_pass.set_pipeline(self._pipeline)
        colormap_render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        colormap_render_pass.draw(4, 1, 0, 0)
        colormap_render_pass.end()


    def set_vmin_vmax(self):
        """Set the vmin and vmax values for the colomap based on the most recent SPH render"""

        # This can and probably should be done on-GPU using a compute shader, but for now
        # we'll do it on the CPU
        vals = np.log10(self._visualizer.get_rendered_image()).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self.vmin, self.vmax = np.percentile(vals, [1.0, 99.9])
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmin, self.vmax = 0.0, 1.0

        self._update_vmin_vmax_buffer()
        # self.logMapper['vmin'] = self.vmin
        # self.logMapper['vmax'] = self.vmax
        # self.update_matplotlib_colorbar_texture()


    def _update_vmin_vmax_buffer(self):
        vmin_vmax_dtype = [("vmin", np.float32, (1,)),
                           ("vmax", np.float32, (1,))]
        vmin_vmax = np.zeros((), dtype=vmin_vmax_dtype)
        vmin_vmax["vmin"] = self.vmin
        vmin_vmax["vmax"] = self.vmax
        self._device.queue.write_buffer(self._vmin_vmax_buffer, 0, vmin_vmax)