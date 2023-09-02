from __future__ import annotations

import numpy as np
from abc import ABCMeta, abstractmethod
import wgpu
from .util import load_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer

class Overlay(metaclass=ABCMeta):

    _blending = (
                    wgpu.BlendFactor.src_alpha,
                    wgpu.BlendFactor.one_minus_src_alpha,
                    wgpu.BlendOperation.add,
                 )
    def __init__(self, visualizer: Visualizer, target_texture: wgpu.GPUTexture | None = None):
        """Setup the overlay.

        :param visualizer: The visualizer instance
        :param target_texture: The texture to render the overlay to. If None, the visualizer's canvas is used.
        """
        self._visualizer = visualizer
        self._device = self._visualizer.device
        self._contents = None
        self._target_texture = target_texture
        if target_texture is None:
            self._target_canvas_format = visualizer.canvas_format
        else:
            self._target_canvas_format = target_texture.format

        self._setup_shader_module()
        self._setup_texture()
        self._setup_sampler()
        self._setup_params_buffer()
        self._setup_render_pipeline()

    def _setup_shader_module(self):
        if not hasattr(self._visualizer, "_overlay_shader"):
            # this looks a bit weird, but it's because we want to share the shader module
            # between all overlays using the same device
            self._visualizer._overlay_shader = self._device.create_shader_module(code=load_shader("overlay.wgsl"),
                                                                                 label="overlay")

    @property
    def _shader(self):
        return self._visualizer._overlay_shader

    def _setup_sampler(self):
        self._sampler = self._device.create_sampler(label="overlay_sampler",
                                                    mag_filter=wgpu.FilterMode.linear,
                                                    min_filter=wgpu.FilterMode.linear)

    def _setup_texture(self):
        im = self.get_contents()

        assert len(im.shape)==3 and im.shape[2]==4, "Overlay must be RGBA image"
        assert im.dtype==np.float32, "Overlay must be float32"

        self._texture = self._device.create_texture(
            label="overlay_texture",
            size=(im.shape[1], im.shape[0], 1),
            dimension=wgpu.TextureDimension.d2,
            format=wgpu.TextureFormat.rgba32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING)

        self._device.queue.write_texture(
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            im.tobytes(),
            {
                "bytes_per_row": 4 * 4 * im.shape[1],
                "offset": 0,
            },
            (im.shape[1], im.shape[0], 1)
        )



    def _setup_params_buffer(self):
        self._overlay_params_buffer = self._device.create_buffer(
            label="overlay_params_buffer",
            size=4*8,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

    def get_texturespace_coordinates(self, width, height) -> tuple[float,float,float,float]:
        return (0.0,0.0,1.0,1.0)

    def _update_params_buffer(self, width, height):
        x, y, w, h = self.get_clipspace_coordinates(width, height)
        x_t, y_t, w_t, h_t = self.get_texturespace_coordinates(width, height)
        self._device.queue.write_buffer(self._overlay_params_buffer, 0,
                                        np.array([x,y,w,h,x_t,y_t,w_t,h_t], dtype=np.float32).tobytes())
    def _setup_render_pipeline(self):


        # bind group layout for:
        # 0: uniform buffer
        # 1: texture
        # 2: sampler
        self._bind_group_layout = self._device.create_bind_group_layout(
            label="overlay_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
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
            ],
        )

        self._bind_group = self._device.create_bind_group(
            label="overlay_bind_group",
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._overlay_params_buffer,
                        "offset": 0,
                        "size": 4*8,
                    },
                },
                {
                    "binding": 1,
                    "resource": self._texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": self._sampler,
                },
            ],
        )

        self._pipeline_layout = self._device.create_pipeline_layout(
            label="overlay_render_pipeline_layout",
            bind_group_layouts=[self._bind_group_layout]
        )

        self._render_pipeline = self._device.create_render_pipeline(
            label="overlay_render_pipeline",
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
                        "format": self._target_canvas_format,
                        "blend": {
                            "color": self._blending,
                            "alpha": self._blending,
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ]
            }
        )

    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder):
        targ_tex: wgpu.GPUTextureView
        if self._target_texture is None:
            targ_tex = self._visualizer.context.get_current_texture()
        else:
            targ_tex = self._target_texture.create_view()
        self._update_params_buffer(targ_tex.size[0], targ_tex.size[1])

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": targ_tex,
                "resolve_target": None,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }],
        )


        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99999)
        render_pass.draw(4, 1, 0, 0)
        render_pass.end()

    @abstractmethod
    def get_clipspace_coordinates(self, width, height) -> tuple[float,float,float,float]:
        """Must return geometry of overlay as x0, y0, width, height in clip space coordinates."""
        pass

    def get_contents(self) -> np.ndarray:
        """Return a cached 2D image with RGBA channels for display.

        If needed, invokes a render_contents() call to the subclass."""
        if self._contents is None:
            self._contents = self.render_contents()
        return self._contents

    def invalidate_contents(self):
        """Mark the contents as invalid, needing a re-render"""
        self._contents = None
    @abstractmethod
    def render_contents(self) -> np.ndarray:
        """Must return a 2D image with RGBA channels for display."""
        pass

    def update(self):
        self._contents = None
        self._setup_texture()
        self._setup_render_pipeline()