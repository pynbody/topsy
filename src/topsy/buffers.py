import numpy as np

from . import config

class Buffers:
    def __init__(self, num_particles: int, max_particles_per_buffer: int | None = None):
        if max_particles_per_buffer is None:
            max_particles_per_buffer = config.MAX_PARTICLES_PER_BUFFER

        self._num_particles = num_particles
        self._max_particles_per_buffer = max_particles_per_buffer

    def _calculate_splits(self):
        if self._num_particles > self._max_particles_per_buffer:
            num_buffers = int(np.ceil(self._num_particles / self._max_particles_per_buffer))
        else:
            num_buffers = 1

        self._buffer_particle_sizes = np.empty(num_buffers, dtype=np.uintp)
        self._buffer_particle_sizes.fill(self._max_particles_per_buffer)
        self._buffer_particle_sizes[-1] = self._num_particles % self._max_particles_per_buffer

        self._buffer_particle_starts = np.cumsum(self._buffer_particle_sizes) - self._buffer_particle_sizes

