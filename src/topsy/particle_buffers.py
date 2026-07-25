import logging
import numpy as np
import wgpu

from . import loader, split_buffers

logger = logging.getLogger(__name__)

_UNSET = object()

class ParticleBuffers:
    def __init__(self, loader: loader.AbstractDataLoader, device: wgpu.GPUDevice, max_draw_calls_per_buffer: int):
        self.buffers = {}
        self._split_buffers = split_buffers.SplitBuffers(len(loader))
        self._device = device
        self._loader = loader

        self.quantity_name = None
        self.vector_name = 'vel'
        self._mass_and_quantity_buffers = None
        self._quantity_buffer_is_for_name = _UNSET # can't use None here because None is valid (means 'density render')
        self._vec_buffer_is_for_name = _UNSET # tracks which vector_name _vec_buffers currently holds
        self._current_vertex_buffers = []

        self._create_indirect_draw_buffers(max_draw_calls_per_buffer)

        self._last_bufnum = -1

    def _create_indirect_draw_buffers(self, max_draw_calls_per_buffer: int):
        self._indirect_buffers = [] # for indirect draw calls, one needed per physical buffer
        self._indirect_count_buffers = []
        self._indirect_buffers_npy = []
        self._indirect_count_buffers_npy = []
        self._max_draw_calls_per_buffer = max_draw_calls_per_buffer

        for i in range(self._split_buffers.num_buffers):
            self._indirect_buffers.append(
                self._device.create_buffer(size=max_draw_calls_per_buffer * 4 * np.dtype(np.uint32).itemsize,
                                           usage = wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST)
            )
            #self._indirect_count_buffers.append(
            #    self._device.create_buffer(size=np.dtype(np.uint32).itemsize,
            #                               usage=wgpu.BufferUsage.INDIRECT | wgpu.BufferUsage.COPY_DST)
            #)
            self._indirect_buffers_npy.append(np.zeros((max_draw_calls_per_buffer, 4), dtype=np.uint32))
            #self._indirect_count_buffers_npy.append(np.zeros((1,), dtype=np.uint32))
            self._indirect_buffers_npy[-1][:, 0] = 6 # vertex count


    def specify_vertex_buffer_assignment(self, buffer_names):
        buffers = []
        for name in buffer_names:
            match name:
                case "pos_smooth":
                    buffers.append(self.get_pos_smooth_buffers())
                case "mass_and_quantity":
                    buffers.append(self.get_mass_and_quantity_buffers())
                case "rgb":
                    buffers.append(self.get_rgb_buffers())
                case "vec":
                    buffers.append(self.get_vec_buffers())
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

    def issue_draw_indirect(self, sph_render_pass: wgpu.GPURenderPassEncoder):

        for bufnum in range(self._split_buffers.num_buffers):
            self.set_vertex_buffers(bufnum, sph_render_pass)
            sph_render_pass._multi_draw_indirect(self._indirect_buffers[bufnum], 0, self._max_draw_calls_per_buffer)

    def update_particle_ranges(self, particle_mins: list[int], particle_lens: list[int]):
        per_buf_start_lens = self._split_buffers.global_to_split_monotonic(particle_mins, particle_lens)
        for bufnum, (particle_min, particle_len) in enumerate(per_buf_start_lens):
            self._indirect_buffers_npy[bufnum][len(particle_min):,1] = 0
            self._indirect_buffers_npy[bufnum][:len(particle_min), 1] = particle_len # instance count
            self._indirect_buffers_npy[bufnum][:len(particle_min), 3] = particle_min # first instance
            self._device.queue.write_buffer(self._indirect_buffers[bufnum], 0, self._indirect_buffers_npy[bufnum])

    def get_pos_smooth_buffers(self):
        if not hasattr(self, "_pos_smooth_buffers"):
            logger.info("Creating position+smoothing buffer")
            data = self._loader.get_pos_smooth().astype(np.float32)
            self._pos_smooth_buffers = self._split_buffers.create_buffers(self._device, 4 * 4,
                                                                          wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            self._split_buffers.write_buffers(self._device, self._pos_smooth_buffers, data)
        return self._pos_smooth_buffers

    def get_mass_and_quantity_buffers(self):
        self._setup_mass_and_quantity_buffers_if_needed()
        return self._mass_and_quantity_buffers

    def _setup_mass_and_quantity_buffers_if_needed(self):
        if self._quantity_buffer_is_for_name != self.quantity_name:
            if self.quantity_name is not None:
                quantity_data =  self._loader.get_named_quantity(self.quantity_name)
                match quantity_data.shape:
                    case (N,):
                        buffer_dim = 3
                    case (N, 3):
                        buffer_dim = 4
                    case _:
                        raise ValueError(f"Quantity {self.quantity_name} has wrong shape {quantity_data.shape}, "
                                         f"expected (N,) or (N,3)")
                if N != len(self._loader):
                    raise ValueError(f"Quantity {self.quantity_name} has wrong length {N}, expected {len(self._loader)}")
            else:
                buffer_dim = 3
                quantity_data = None
            self._create_mass_and_quantity_buffers_if_needed(buffer_dim)
            data = np.zeros((len(self._loader), buffer_dim), dtype=np.float32)
            data[:, 0] = self._loader.get_mass()
            if quantity_data is not None:
                if buffer_dim == 4:
                    data[:, 1:] = quantity_data
                else:
                    data[:, 1] = quantity_data
            self._split_buffers.write_buffers(self._device, self._mass_and_quantity_buffers, data)
            self._quantity_buffer_is_for_name = self.quantity_name

    def get_mass_and_quantity_buffers_dimension(self):
        self._setup_mass_and_quantity_buffers_if_needed()
        return self._mass_and_quantity_buffers_dim


    def get_rgb_buffers(self):
        if not hasattr(self, "_rgb_masses_buffers"):
            logger.info("Creating rgb buffer")
            data = self._loader.get_rgb_masses().view(np.float32)
            self._rgb_masses_buffers = self._split_buffers.create_buffers(self._device, 4 * 3,
                                                wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            self._split_buffers.write_buffers(self._device, self._rgb_masses_buffers, data)
        return self._rgb_masses_buffers

    def get_vec_buffers(self):
        """Buffers for rendering a projected vector field, independent of the main visualised quantity.

        Each element is a vec4 (mass, vx, vy, vz), i.e. the mass is bundled with the vector quantity.
        This duplicates the mass (already present in the mass_and_quantity buffer), but in return the
        vector SPH shader can read a single, contiguous vec4 vertex input exactly as it already does
        for the LOS vector renderer -- no extra vertex buffer binding or shader input is needed. The
        modest GPU-RAM overhead (one extra float per particle) is well traded off against that
        simplicity and the better memory access pattern.
        """
        if not hasattr(self, "_vec_buffers"):
            logger.info("Creating vector buffer")
            self._vec_buffers = self._split_buffers.create_buffers(self._device, 4 * 4,
                                                wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        if self._vec_buffer_is_for_name != self.vector_name:
            # (re)fill the buffer whenever the selected vector quantity changes
            logger.info(f"Filling vector buffer for '{self.vector_name}'")
            data = np.zeros((len(self._loader), 4), dtype=np.float32)
            data[:, 0] = self._loader.get_mass()
            data[:, 1:] = self._loader.get_named_quantity(self.vector_name)
            self._split_buffers.write_buffers(self._device, self._vec_buffers, data)
            self._vec_buffer_is_for_name = self.vector_name
        return self._vec_buffers

    def _create_mass_and_quantity_buffers_if_needed(self, buffer_dim: int):
        if self._mass_and_quantity_buffers is not None and self._mass_and_quantity_buffers_dim == buffer_dim:
            return
        logger.info("Creating quantity buffer")
        self._mass_and_quantity_buffers = self._split_buffers.create_buffers(self._device, 4 * buffer_dim,
                                                                             wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self._mass_and_quantity_buffers_dim = buffer_dim
