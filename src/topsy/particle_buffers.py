import logging
import numpy as np
import wgpu

from . import loader, split_buffers

logger = logging.getLogger(__name__)

class ParticleBuffers:
    def __init__(self, loader: loader.AbstractDataLoader, device: wgpu.GPUDevice):
        self.buffers = {}
        self._split_buffers = split_buffers.SplitBuffers(len(loader))
        self._device = device
        self._loader = loader

        self.quantity_name = None
        self._named_quantity_buffers = None
        self._quantity_buffer_is_for_name = None
        self._current_vertex_buffers = []

        self._last_bufnum = -1

    def specify_vertex_buffer_assignment(self, buffer_names):
        buffers = []
        for name in buffer_names:
            match name:
                case "pos_smooth":
                    buffers.append(self.get_pos_smooth_buffers())
                case "mass":
                    buffers.append(self.get_mass_buffers())
                case "quantity":
                    buffers.append(self.get_quantity_buffers())
                case "rgb_masses":
                    buffers.append(self.get_rgb_masses_buffers())
                case _:
                    raise ValueError(f"Unknown buffer name: {name}")
        self._current_vertex_buffers = buffers
        self._last_bufnum = -1

    def set_vertex_buffers(self, bufnum: int, render_pass: wgpu.GPURenderPassEncoder):
        if bufnum == self._last_bufnum:
            return
        for i, buffers in enumerate(self._current_vertex_buffers):
            render_pass.set_vertex_buffer(i, buffers[bufnum])
        self._last_bufnum = bufnum

    def iter_particle_ranges(self, particle_mins: list[int], particle_lens: list[int], render_pass: wgpu.GPURenderPassEncoder):
        """Iterate over logical particle ranges yielding local starts/lengths for each buffer, setting vertex buffers as needed."""
        per_buf_start_lens = self._split_buffers.global_to_split_monotonic(particle_mins, particle_lens)
        for i, (this_buf_starts, this_buf_lens) in enumerate(per_buf_start_lens):
            self.set_vertex_buffers(i, render_pass)
            for start, length in zip(this_buf_starts, this_buf_lens):
                yield start, length

    def get_pos_smooth_buffers(self):
        if not hasattr(self, "_pos_smooth_buffers"):
            logger.info("Creating position+smoothing buffer")
            data = self._loader.get_pos_smooth().astype(np.float32)
            self._pos_smooth_buffers = self._split_buffers.create_buffers(self._device, 4 * 4,
                                                                          wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self._split_buffers.write_buffers(self._device, self._pos_smooth_buffers, data)
        return self._pos_smooth_buffers

    def get_mass_buffers(self):
        if not hasattr(self, "_mass_buffers"):
            logger.info("Creating mass buffer")
            data = self._loader.get_mass().astype(np.float32)
            self._mass_buffers = self._split_buffers.create_buffers(self._device, 4,
                                                                     wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self._split_buffers.write_buffers(self._device, self._mass_buffers, data)
        return self._mass_buffers

    def get_quantity_buffers(self):
        if self.quantity_name is None:
            return self.get_mass_buffers()
        elif self._quantity_buffer_is_for_name != self.quantity_name:
            self._create_quantity_buffers_if_needed()
            logger.info(f"Transferring {self.quantity_name} into buffer")
            data = self._loader.get_named_quantity(self.quantity_name).view(np.float32)
            self._split_buffers.write_buffers(self._device, self._named_quantity_buffers, data)
            self._quantity_buffer_is_for_name = self.quantity_name
        return self._named_quantity_buffers

    def get_rgb_masses_buffers(self):
        if not hasattr(self, "_rgb_masses_buffers"):
            logger.info("Creating RGB masses buffer")
            data = self._loader.get_rgb_masses().view(np.float32)
            self._rgb_masses_buffers = self._split_buffers.create_buffers(self._device, 4 * 3,
                                                wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            self._split_buffers.write_buffers(self._device, self._rgb_masses_buffers, data)
        return self._rgb_masses_buffers

    def _create_quantity_buffers_if_needed(self):
        if self._named_quantity_buffers is not None:
            return
        logger.info("Creating quantity buffer")
        self._named_quantity_buffers = self._split_buffers.create_buffers(self._device, 4,
                                             wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)