import numpy as np
import wgpu
import logging
import pynbody
import pickle 

from . import config
# ABC support:
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)
class AbstractDataLoader(ABC):
    def __init__(self, device: wgpu.GPUDevice):
        self._device = device
        self.quantity_name = None
        self._quantity_buffer_is_for_name = None
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

    @abstractmethod
    def get_named_quantity(self, name):
        pass

    @abstractmethod
    def get_quantity_label(self):
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

    def get_quantity_buffer(self):
        if self.quantity_name is None:
            return self.get_mass_buffer()
        elif self._quantity_buffer_is_for_name != self.quantity_name:
            logger.info(f"Creating {self.quantity_name} buffer")
            data = self.get_named_quantity(self.quantity_name)
            self._named_quantity_buffer = self._device.create_buffer_with_data(
                data=data,
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.UNIFORM)
            self._quantity_buffer_is_for_name = self.quantity_name
        return self._named_quantity_buffer

    def get_periodicity_scale(self):
        return np.inf


class PynbodyDataInMemory(AbstractDataLoader):
    """Base class for data loaders that use pynbody."""

    def __init__(self, device: wgpu.GPUDevice, snapshot: pynbody.snapshot.SimSnap):
        super().__init__(device)

        self.snapshot = snapshot

        # randomize order to avoid artifacts when downsampling number of particles on display
        self._random_order = np.random.permutation(len(self.snapshot))

    def get_positions(self):
        return self.snapshot['pos'].astype(np.float32)[self._random_order]

    def get_smooth(self):
        return self.snapshot['smooth'].astype(np.float32)[self._random_order]

    def get_mass(self):
        return self.snapshot['mass'].astype(np.float32)[self._random_order]

    def get_named_quantity(self, name):
        qty =self.snapshot[name]
        if len(qty.shape)==2:
            qty = qty[:,0]
        return qty.astype(np.float32)[self._random_order]

    def get_quantity_names(self):
        return self.snapshot.loadable_keys()

    def get_quantity_label(self):
        if self.quantity_name is None:
            return r"density / $M_{\odot} / \mathrm{kpc}^2$"
        else:
            lunit = self.snapshot[self.quantity_name].units.latex()
            if lunit != "":
                lunit = "$/" + lunit + "$"
            return self.quantity_name + lunit

    def __len__(self):
        return len(self.snapshot)

    def get_periodicity_scale(self):
        return float(self.snapshot.properties['boxsize'].in_units("kpc"))
    def get_filename(self):
        return self.snapshot.filename

class PynbodyDataLoader(PynbodyDataInMemory):
    """Literal data loader for pynbody (starts from just a filename)"""
    def __init__(self, device: wgpu.GPUDevice, filename: str, center: str, particle: str):

        logger.info(f"Data filename = {filename}, center = {center}, particle = {particle}")
        snapshot = pynbody.load(filename)
        snapshot.physical_units()
        self.filename = filename

        snapshot = snapshot[pynbody.family.get_family(particle)]

        super().__init__(device, snapshot)

        self._perform_centering(center)
        snapshot.wrap()
        self._perform_smoothing()

    def _perform_centering(self, center):
        logger.info("Performing centering...")
        if center.startswith("halo-"):
            halo_number = int(center[5:])
            h = self.snapshot.ancestor.halos()
            pynbody.analysis.halo.center(h[halo_number], vel=False)

        elif center == 'zoom':
            f_dm = self.snapshot.ancestor.dm
            pynbody.analysis.halo.center(f_dm[f_dm['mass'] < 1.01 * f_dm['mass'].min()])
        elif center == 'all':
            pynbody.analysis.halo.center(self.snapshot)
        elif center == 'none':
            pass
        else:
            raise ValueError("Unknown centering type")

    def _perform_smoothing(self):
        try:
            logger.info("Looking for cached smoothing/density data...")
            smooth = pickle.load(open(self.filename+'-topsy-smooth.pkl', 'rb'))
            if len(smooth) == len(self.snapshot):
                self.snapshot['smooth'] = smooth
            else:
                raise ValueError("Incorrect number of particles in cached smoothing data")
            logger.info("...success!")

            rho = pickle.load(open(self.filename+'-topsy-rho.pkl', 'rb'))
            if len(rho) == len(self.snapshot):
                self.snapshot['rho'] = rho
            else:
                raise ValueError("Incorrect number of particles in cached density data")
        except:
            logger.info("Generating smoothing/density data - this can take a while but will be cached for future runs")
            pickle.dump(self.snapshot['smooth'], open(self.filename+'-topsy-smooth.pkl', 'wb'))
            pickle.dump(self.snapshot['rho'], open(self.filename+'-topsy-rho.pkl', 'wb'))



class TestDataLoader(AbstractDataLoader):
    def __init__(self, device: wgpu.GPUDevice, n_particles: int = config.TEST_DATA_NUM_PARTICLES_DEFAULT):
        self._n_particles = n_particles
        self._gmm_weights = [0.5, 0.4, 0.1] # should sum to 1
        self._gmm_means = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[6.0,10.0,0.0]])
        self._gmm_std = np.array([[20.0, 20.0, 20.0], [4.0, 0.2, 4.0], [2.0,2.0,3.0]])

        self._gmm_pos = self._generate_samples()
        self._gmm_den = self._evaluate_density(self._gmm_pos)
        super().__init__(device)
    def __len__(self):
        return self._n_particles

    def _evaluate_density(self, pos):
        # simple gaussian mixture model, density of particles per unit volume
        den = np.zeros(len(pos))
        for i in range(len(self._gmm_weights)):
            den += self._gmm_weights[i] * np.exp(-np.sum((pos - self._gmm_means[i])**2/self._gmm_std[i]**2, axis=1)) \
                                                 / ((2 * np.pi)**1.5 * np.prod(self._gmm_std[i]))
        return den*self._n_particles

    def _generate_samples(self):
        # simple gaussian mixture model
        pos = np.empty((self._n_particles, 3), dtype=np.float32)
        offset = 0
        for i in range(len(self._gmm_weights)):
            cpt_len = int(self._n_particles*self._gmm_weights[i])
            pos[offset:offset+cpt_len] = \
                np.random.normal(size=(cpt_len, 3), scale=1.0).astype(np.float32) * self._gmm_std[np.newaxis,i,:] + self._gmm_means[i]
            offset += cpt_len
        assert offset == self._n_particles
        return np.random.permutation(pos)

    def get_positions(self):

        return self._gmm_pos

    def get_smooth(self):
        sm = 2.0/self._gmm_den**0.333333
        return sm

    def get_mass(self):
        return np.random.uniform(0.01, 1.0, size=(self._n_particles)).astype(np.float32)*1e-8

    def get_named_quantity(self, name):
        if name=="test-quantity":
            return np.sin(self._gmm_pos[:,0])*np.cos(self._gmm_pos[:,1])*np.cos(self._gmm_pos[:,2])
        else:
            raise KeyError("Unknown quantity name")

    def get_quantity_names(self):
        return ["test-quantity"]

    def get_quantity_label(self):
        if self.quantity_name is None:
            return r"test density / $M_{\odot} / \mathrm{kpc}^2$"
        elif self.quantity_name == "test-quantity":
            return "test quantity"
        else:
            return "unknown"

    def get_filename(self):
        return "test data"
