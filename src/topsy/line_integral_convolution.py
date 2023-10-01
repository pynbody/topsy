from __future__ import annotations

import numpy as np
import wgpu
import pynbody

from .util import load_shader, preprocess_shader
from . import config

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import VisualizerBase


class LineIntegralConvolution:
    def __init__(self, visualizer: VisualizerBase,
                 render_texture: wgpu.GPUTexture, source_texture: wgpu.GPUTexture):
        self._visualizer = visualizer
        self._device = visualizer.device
        self._render_texture = render_texture
        self._source_texture = source_texture
        self._scale = 0.05
        self._displacement_map_resolution = 100

        self._params_buffer = self._device.create_buffer(
            label="lic_params_buffer",
            size=4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        self._sampler = self._device.create_sampler(label="lic_sampler",
                                                    mag_filter=wgpu.FilterMode.linear,
                                                    min_filter=wgpu.FilterMode.linear)

        self._setup_shader_module()
        self._setup_displacement_texture()
        self._setup_render_pipeline()

    def _setup_shader_module(self):
        code = load_shader("line_integral_convolution.wgsl")
        self._shader = self._device.create_shader_module(code=code, label='line_integral_convolution')

    def _setup_displacement_texture(self):
        # temporary
        x,y = np.meshgrid(np.linspace(-1,1,self._displacement_map_resolution), np.linspace(-1,1,self._displacement_map_resolution))
        vx = 0.1*y
        vy = -0.1*x
        v = np.stack([vx,vy], axis=-1).astype(np.float32)

        self._displacement_texture = self._device.create_texture(
            size=(self._displacement_map_resolution,self._displacement_map_resolution,1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rg32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.RENDER_ATTACHMENT,
            label="displacement_texture"
        )

        self._device.queue.write_texture(
            {
                "texture": self._displacement_texture,
                "mip_level": 0,
                "origin": (0,0,0),
            },
            v.tobytes(),
            {
                "offset": 0,
                "bytes_per_row": 8*self._displacement_map_resolution,
            },
            (self._displacement_map_resolution,self._displacement_map_resolution,1)
        )

    def _setup_render_pipeline(self):
        # bind group layout for:
        # 0: uniform buffer
        # 1: texture
        # 2: sampler
        self._bind_group_layout = self._device.create_bind_group_layout(
            label="lic_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                    },
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {
                        "type": wgpu.SamplerBindingType.filtering,
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": wgpu.TextureViewDimension.d2
                    },
                },
            ],
        )

        self._bind_group = self._device.create_bind_group(
            label="lic_bind_group",
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._params_buffer,
                        "offset": 0,
                        "size": 4 ,
                    },
                },
                {
                    "binding": 1,
                    "resource": self._source_texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": self._sampler,
                },
                {
                    "binding": 3,
                    "resource": self._displacement_texture.create_view(),
                },
            ],
        )

        self._pipeline_layout = self._device.create_pipeline_layout(
            label="lic_render_pipeline_layout",
            bind_group_layouts=[self._bind_group_layout]
        )

        self._render_pipeline = self._device.create_render_pipeline(
            label="lic_render_pipeline",
            layout=self._pipeline_layout,
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
                        "format": self._render_texture.format,
                        "blend": {
                            "color": (wgpu.BlendFactor.one, wgpu.BlendFactor.zero, wgpu.BlendOperation.add),
                            "alpha": (wgpu.BlendFactor.one, wgpu.BlendFactor.zero, wgpu.BlendOperation.add),
                        },
                    }
                ]
            }
        )


    def _update_params_buffer(self):
        self._scale+=0.05
        if self._scale>1:
            self._scale = 0.0
        self._device.queue.write_buffer(self._params_buffer, 0,
                                        np.array([self._scale], dtype=np.float32).tobytes())
    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder):
        self._update_params_buffer()
        tv = self._render_texture.create_view()
        render_pass = command_encoder.begin_render_pass(
            label="lic_render_pass",
            color_attachments=[
                {
                    "view": tv,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 999999)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end()


