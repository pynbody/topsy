import logging
import numpy as np
import wgpu

from . import config, performance

logger = logging.getLogger(__name__)

class SplitBuffers:
    """Manages splitting buffers into smaller buffers for GPU hardware.

    This is needed because some/most GPUs have a limit on the size of buffers which is less than the actual total
    RAM available.

    We adopt terminology of a 'global' address (in terms of particles) which can then be mapped onto a 'split'
    address which is a tuple of (buffer number, buffer offset)."""

    def __init__(self, num_particles: int, max_particles_per_buffer: int | None = None):
        if max_particles_per_buffer is None:
            max_particles_per_buffer = config.MAX_PARTICLES_PER_BUFFER

        self._num_particles = num_particles
        self._max_particles_per_buffer = max_particles_per_buffer
        self._calculate_splits()

    def _calculate_splits(self):
        if self._num_particles > self._max_particles_per_buffer:
            num_buffers = int(np.ceil(self._num_particles / self._max_particles_per_buffer))
        else:
            num_buffers = 1

        self._num_buffers = num_buffers
        self._buffer_particle_sizes = np.empty(num_buffers, dtype=np.intp)
        self._buffer_particle_sizes.fill(self._max_particles_per_buffer)
        self._buffer_particle_sizes[-1] = self._num_particles - (len(self._buffer_particle_sizes)-1) * self._max_particles_per_buffer

        self._buffer_particle_starts = np.cumsum(self._buffer_particle_sizes) - self._buffer_particle_sizes
        logger.info(f"Splitting {self._num_particles} particles into {self._num_buffers} buffer(s)")


    def _global_to_split_address(self, address: int) -> (int, int):
        """Given a logical buffer particle offset, returns the physical buffer number and address"""
        bufnum = np.searchsorted(self._buffer_particle_starts, address, side='right')-1
        return bufnum, address - self._buffer_particle_starts[bufnum]

    @property
    def num_buffers(self) -> int:
        """Number of buffers per global buffer"""
        return self._num_buffers

    def global_to_split(self, start: int, length: int) -> tuple[list[int], list[int], list[int]]:
        """Map global start and length to split buffer numbers, starts and lengths."""
        bufs = []
        starts = []
        lengths = []

        global_start = start
        global_length_remaining = length
        bufnum, local_start = self._global_to_split_address(global_start)

        while global_length_remaining>0 and bufnum<self._num_buffers:
            maxlen = self._buffer_particle_sizes[bufnum] - local_start
            buf_len = min(global_length_remaining, maxlen)
            bufs.append(bufnum)
            starts.append(local_start)
            lengths.append(buf_len)

            global_start += buf_len
            global_length_remaining -= buf_len
            bufnum += 1
            local_start = 0

        if global_length_remaining > 0:
            raise ValueError(f"Requested length {length} starting at {start} exceeds available buffers")

        return bufs, starts, lengths

    def global_to_split_monotonic(self, start: list[int], length: list[int]) -> list[tuple[list[int], list[int]]]:
        """Map global start and length to starts and lengths for each buffer. Addressing must be monotonically increasing."""
        performance.signposter.emit_event("global_to_split_monotonic")
        starts = []
        lengths = []

        cur_buf = 0
        cur_buf_start = 0
        cur_buf_end = self._buffer_particle_sizes[cur_buf]

        all_buf_results = [(starts, lengths)]

        for global_start, global_length in zip(start, length):
            while global_length > 0:
                while global_start >= cur_buf_end:
                    # move to next buffer
                    cur_buf += 1
                    if cur_buf>=self._num_buffers:
                        raise ValueError(f"Requested length {global_length} starting at {global_start} exceeds available buffers")
                    cur_buf_start = self._buffer_particle_starts[cur_buf]
                    cur_buf_end = cur_buf_start + self._buffer_particle_sizes[cur_buf]
                    starts = []
                    lengths = []
                    all_buf_results.append((starts, lengths))

                this_buf_start = global_start - cur_buf_start
                this_buf_length = min(global_length, cur_buf_end - global_start)
                starts.append(this_buf_start)
                lengths.append(this_buf_length)
                global_length -= this_buf_length
                global_start += this_buf_length

        if cur_buf < self._num_buffers-1:
            for bufnum in range(cur_buf+1, self._num_buffers):
                all_buf_results.append(([], []))

        performance.signposter.emit_event("end global_to_split_monotonic")

        return all_buf_results


    def create_buffers(self, wgpu_device: wgpu.GPUDevice, item_size: int, usage: wgpu.BufferUsage) -> list[wgpu.GPUBuffer]:
        """Create a set of split buffers

        Parameters
        ----------
        wgpu_device : wgpu.GPUDevice
            The GPU device to create the buffer on.
        item_size : int
            The size of each item in the buffer.
        usage : int
            The usage flags for the buffer.

        """
        buffers = []
        for this_size in self._buffer_particle_sizes:
            size = this_size * item_size
            buffer = wgpu_device.create_buffer(
                size=size,
                usage=usage,
            )
            buffers.append(buffer)
        return buffers

    def write_buffers(self, wgpu_device: wgpu.GPUDevice, buffers: list[wgpu.GPUBuffer], data: np.ndarray) -> None:
        """Write data to the split buffers.

        Parameters
        ----------
        wgpu_device : wgpu.GPUDevice
            The GPU device to write the buffer on.
        buffers : list[wgpu.GPUBuffer]
            The buffers to write to.
        data : np.ndarray
            The data to write.
        """
        if len(buffers) != self._num_buffers:
            raise ValueError(f"Number of buffers {len(buffers)} does not match number of split buffers {self._num_buffers}")
        if len(data) != self._num_particles:
            raise ValueError(f"Data size {len(data)} does not match number of particles {self._num_particles}")

        for bufnum, buf in enumerate(buffers):
            start = self._buffer_particle_starts[bufnum]
            length = self._buffer_particle_sizes[bufnum]
            wgpu_device.queue.write_buffer(buf, 0, data[start:start+length])