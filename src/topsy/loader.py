import numpy as np
import wgpu
import logging
import pynbody
import pickle 

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


class PynbodyDataLoader(AbstractDataLoader):
    def __init__(self, device: wgpu.GPUDevice, filename: str, center: str, particle: str):
        super().__init__(device)

        logger.info(f"Data filename = {filename}, center = {center}, particle = {particle}")
        self.f = pynbody.load(filename)
        self.f.physical_units()

        self.f = self.f[pynbody.family.get_family(particle)]

        self._perform_centering(center)
        self._perform_smoothing()

        # randomize order to avoid artifacts when downsampling number of particles on display
        self.random_order = np.random.permutation(len(self.f))

    def _perform_centering(self, center):
        logger.info("Performing centering...")
        if center.startswith("halo-"):
            halo_number = int(center[5:])
            h = self.f.ancestor.halos()
            pynbody.analysis.halo.center(h[halo_number])

        elif center == 'zoom':
            f_dm = self.f.ancestor.dm
            pynbody.analysis.halo.center(f_dm[f_dm['mass'] < 1.01 * f_dm['mass'].min()])
        elif center == 'all':
            pynbody.analysis.halo.center(self.f)
        elif center == 'none':
            pass
        else:
            raise ValueError("Unknown centering type")

    def _perform_smoothing(self):
        try:
            logger.info("Looking for cached smoothing/density data...")
            smooth = pickle.load(open('topsy-smooth.pkl', 'rb'))
            if len(smooth) == len(self.f):
                self.f['smooth'] = smooth
            else:
                raise ValueError("Incorrect number of particles in cached smoothing data")
            logger.info("...success!")

            rho = pickle.load(open('topsy-rho.pkl', 'rb'))
            if len(rho) == len(self.f):
                self.f['rho'] = rho
            else:
                raise ValueError("Incorrect number of particles in cached density data")
        except:
            logger.info("Generating smoothing/density data - this can take a while but will be cached for future runs")
            pickle.dump(self.f['smooth'], open('topsy-smooth.pkl', 'wb'))
            pickle.dump(self.f['rho'], open('topsy-rho.pkl', 'wb'))

    def get_positions(self):
        return self.f['pos'].astype(np.float32)[self.random_order]

    def get_smooth(self):
        return self.f['smooth'].astype(np.float32)[self.random_order]

    def get_mass(self):
        return self.f['mass'].astype(np.float32)[self.random_order]

    def __len__(self):
        return len(self.f)

class TestDataLoader(AbstractDataLoader):
    def __init__(self, device: wgpu.GPUDevice, n_particles: int = int(5e6)):
        self._n_particles = n_particles
        self._gmm_weights = [0.5, 0.25, 0.25] # should sum to 1
        self._gmm_means = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.6,0.0,0.0]])
        self._gmm_std = np.array([[0.2, 0.2, 0.2], [0.4, 0.02, 0.4], [0.1, 0.1, 0.1]])

        self._gmm_weights = [1.0]
        self._gmm_means = np.array([[0.0, 0.0, 0.0]])
        self._gmm_std = np.array([[0.2, 0.2, 0.2]])

        self._gmm_pos = self._generate_samples()
        self._gmm_den = self._evaluate_density(self._gmm_pos)
        super().__init__(device)
    def __len__(self):
        return self._n_particles

    def _evaluate_density(self, pos):
        # simple gaussian mixture model, density of particles per unit volume
        den = np.zeros(len(pos))
        for i in range(len(self._gmm_weights)):
            den += self._gmm_weights[i] * np.exp(-np.sum((pos - self._gmm_means[i])**2, axis=1)) \
                                                 / ((2 * np.pi)**1.5 * np.prod(self._gmm_std[i]))
        return den*self._n_particles

    def _generate_samples(self):
        # simple gaussian mixture model
        pos = np.empty((self._n_particles, 3))
        offset = 0
        for i in range(len(self._gmm_weights)):
            cpt_len = int(self._n_particles*self._gmm_weights[i])
            pos[offset:offset+cpt_len] = \
                np.random.normal(size=(cpt_len, 3), scale=1.0).astype(np.float32) * self._gmm_std[np.newaxis,i,:] + self._gmm_means[i]
            offset += cpt_len
        assert offset == self._n_particles
        return pos

    def get_positions(self):

        return self._gmm_pos

    def get_smooth(self):
        return 10./self._gmm_den**0.333333
        #return np.random.uniform(0.01, 0.05, size=(self._n_particles)).astype(np.float32)

    def get_mass(self):
        return np.random.uniform(0.01, 1.0, size=(self._n_particles)).astype(np.float32)*1e-8