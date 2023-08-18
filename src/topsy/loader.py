import numpy as np
import wgpu
import logging
# ABC support:
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
class AbstractDataLoader(ABC):
    def __init__(self, device: wgpu.GPUDevice):
        self._device = device
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_positions(self):
        pass

    @abstractmethod
    def get_smooth(self):
        pass

    @abstractmethod
    def get_mass(self):
        pass

    def get_pos_smooth(self):
        pos_smooth = np.empty((len(self), 4), dtype=np.float32)
        pos_smooth[:, :3] = self.get_positions()
        pos_smooth[:, 3] = self.get_smooth()
        return pos_smooth

    def get_pos_smooth_buffer(self):
        if not hasattr(self, "_pos_smooth_buffer"):
            logger.info("Creating position+smoothing buffer")
            data = self.get_pos_smooth()
            self._pos_smooth_buffer = self._device.create_buffer_with_data(
                data=data,
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM)
        return self._pos_smooth_buffer

    def get_mass_buffer(self):
        if not hasattr(self, "_mass_buffer"):
            logger.info("Creating mass buffer")
            data = self.get_mass()
            self._mass_buffer = self._device.create_buffer_with_data(
                data=data,
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM)
        return self._mass_buffer

class TestDataLoader(AbstractDataLoader):
    def __init__(self, device: wgpu.GPUDevice, n_particles: int = int(5e6)):
        self._n_particles = n_particles
        super().__init__(device)
    def __len__(self):
        return self._n_particles

    def get_positions(self):
        # simple gaussian mixture model
        data = np.random.normal(size=(self._n_particles, 3), scale=0.2).astype(np.float32)

        data[:self._n_particles // 2] = \
            np.random.normal(size=(self._n_particles // 2, 3), scale=0.4).astype(np.float32) * [1.0, 0.05, 1.0]

        data[:self._n_particles // 4] = \
            np.random.normal(size=(self._n_particles // 4, 3), scale=0.1).astype(np.float32) \
            + [0.6, 0.0, 0.0]

        return data

    def get_smooth(self):
        return np.random.uniform(0.01, 0.05, size=(self._n_particles)).astype(np.float32)

    def get_mass(self):
        return np.random.uniform(0.01, 1.0, size=(self._n_particles)).astype(np.float32)*1e-8