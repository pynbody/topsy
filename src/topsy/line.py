from __future__ import annotations

import numpy as np
import wgpu

from .util import load_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer
class Line:
    def __init__(self, visualizer: Visualizer, path, color, width):
        self._visualizer = visualizer
        self._path = np.asarray(path, dtype=np.float32)
        assert self._path.ndim == 2, "Path must be an array of points, each with 4 (xyzw) coordinates"
        assert self._path.shape[1] == 4, "Path must be an array of points, each with 4 (xyzw) coordinates"
        self._color = color
        self._width = width
        self._device = visualizer.device
        self._target_canvas_format = visualizer.canvas_format

        self._setup_shader_module()
        self._setup_buffers()
        self._setup_render_pipeline()

    def _setup_shader_module(self):
        self._shader_module = self._visualizer.device.create_shader_module(
            code=load_shader("line.wgsl"),
            label="line_shader_module"
        )

    def _setup_buffers(self):
        self._vertex_buffer = self._visualizer.device.create_buffer_with_data(
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="line_vertex_buffer",
            data=self._path
        )

        _param_dtype = np.dtype([
            ("transform", np.float32, (4, 4)),
            ("color", np.float32, 4),
            ("width", np.float32),
            ("padding", np.float32, 3)
        ])

        self._param_buffer = self._device.create_buffer(
            label="line_param_buffer",
            size=_param_dtype.itemsize,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )

        self._params = np.zeros(1, dtype=_param_dtype)
        self._params["transform"] = np.eye(4)
        self._params["color"] = self._color
        self._params["width"] = self._width

    def _setup_render_pipeline(self):
        self._bind_group_layout = self._device.create_bind_group_layout(
            label="line_bind_group_layout",
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {
                        "type": wgpu.BufferBindingType.uniform,
                    }
                }
            ]
        )

        self._bind_group = self._device.create_bind_group(
            label="line_bind_group",
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._param_buffer,
                        "offset": 0,
                        "size": self._param_buffer.size
                    }
                }
            ]
        )

        self._pipeline_layout = self._device.create_pipeline_layout(
            label="line_pipeline_layout",
            bind_group_layouts=[self._bind_group_layout]
        )

        self._render_pipeline = self._device.create_render_pipeline(
            label="line_render_pipeline",
            layout=self._pipeline_layout,
            vertex={
                "module": self._shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {   # start of line segment
                        "array_stride": 4*4,
                        "step_mode": wgpu.VertexStepMode.instance,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x4,
                                "offset": 0,
                                "shader_location": 0
                            }
                        ]
                    },
                    {   # end of line segment
                        "array_stride": 4 * 4,
                        "step_mode": wgpu.VertexStepMode.instance,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x4,
                                "offset": 4*4,
                                "shader_location": 1
                            }
                        ]
                    }
                ]
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_strip,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": self._shader_module,
                "entry_point": "fragment_main",
                "targets": [
                    {
                        "format": self._target_canvas_format,
                        "blend": {
                            "color": (
                                        wgpu.BlendFactor.src_alpha,
                                        wgpu.BlendFactor.one_minus_src_alpha,
                                        wgpu.BlendOperation.add,
                                     ),
                            "alpha": (
                                        wgpu.BlendFactor.src_alpha,
                                        wgpu.BlendFactor.one_minus_src_alpha,
                                        wgpu.BlendOperation.add,
                                     ),
                        },
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ]
            }
        )


    def encode_render_pass(self, command_encoder: wgpu.GPURenderPassEncoder):
        self._device.queue.write_buffer(self._param_buffer, 0, self._params)


        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": self._visualizer.context.get_current_texture(),
                "resolve_target": None,
                "clear_value": (0.0, 0.0, 0.0, 1.0),
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }],
        )
        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self._vertex_buffer)
        render_pass.set_vertex_buffer(1, self._vertex_buffer)
        render_pass.draw(4, len(self._path)-1, 0, 0)
        render_pass.end()
