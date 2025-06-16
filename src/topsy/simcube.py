from __future__ import annotations

from .line import Line
import numpy as np

class SimCube(Line):
    def __init__(self, visualizer, color, width):
        size = visualizer.data_loader.get_periodicity_scale() or 1.0
        line_starts_ends = [[0,0,0], [0,0,1],
                            [0,0,0], [0,1,0],
                            [0,0,0], [1,0,0],
                            [1,1,1], [1,1,0],
                            [1,1,1], [1,0,1],
                            [1,1,1], [0,1,1],
                            [0,1,0], [0,1,1],
                            [0,1,0], [1,1,0],
                            [1,0,1], [1,0,0],
                            [1,0,1], [0,0,1],
                            [1,0,0], [1,1,0],
                            [0,1,1], [0,0,1]
                            ]

        line_starts_ends = np.array(line_starts_ends, dtype=np.float32)
        line_starts_ends -= 0.5

        line_starts_ends *= size

        line_starts_ends = np.concatenate([line_starts_ends, np.ones((line_starts_ends.shape[0], 1))], axis=1)

        self._line_starts = np.ascontiguousarray(line_starts_ends[::2,:], dtype=np.float32)
        self._line_ends = np.ascontiguousarray(line_starts_ends[1::2,:], dtype=np.float32)


        super().__init__(visualizer, None, color, width)

    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder,
                                 target_texture_view: wgpu.GPUTextureView):
        self._params["transform"] = self._visualizer._sph.last_transform_params["transform"] @ self._visualizer.sph_clipspace_to_screen_clipspace_matrix()
        super().encode_render_pass(command_encoder, target_texture_view)


