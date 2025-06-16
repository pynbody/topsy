import numpy as np
import wgpu
import logging
import pynbody
import pickle

from typing import Optional

from . import config, cell_layout

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

    @abstractmethod
    def get_named_quantity(self, name):
        pass

    @abstractmethod
    def get_quantity_label(self, quantity_name):
        pass

    @abstractmethod
    def get_rgb_masses(self):
        pass

    def get_pos_smooth(self):
        pos_smooth = np.empty((len(self), 4), dtype=np.float32)
        pos_smooth[:, :3] = self.get_positions()
        pos_smooth[:, 3] = self.get_smooth()
        return pos_smooth

    def get_periodicity_scale(self):
        return np.inf

    def get_render_progression(self):
        from . import progressive_render
        if hasattr(self, '_cell_layout'):
            return progressive_render.RenderProgressionWithCells(self._cell_layout, len(self))
        else:
            return progressive_render.RenderProgression(len(self))


class PynbodyDataInMemory(AbstractDataLoader):
    """Base class for data loaders that use pynbody."""

    _name_smooth_array = 'smooth'

    def __init__(self, device: wgpu.GPUDevice, snapshot: pynbody.snapshot.SimSnap):
        super().__init__(device)

        self.snapshot = snapshot

        boxmin = self.snapshot['pos'].min()-1e-6
        boxmax = self.snapshot['pos'].max()+1e-6
        self._cell_layout, ordering = cell_layout.CellLayout.from_positions(self.snapshot['pos'], boxmin, boxmax,
                                                                            config.DEFAULT_CELLS_NSIDE)
        self._particle_order = ordering[self._cell_layout.randomize_within_cells()]

    def get_positions(self):
        return self.snapshot['pos'].astype(np.float32)[self._particle_order]

    def get_smooth(self):
        return self.snapshot[self._name_smooth_array].astype(np.float32)[self._particle_order]

    def get_mass(self):
        return self.snapshot['mass'].astype(np.float32)[self._particle_order]

    def _effective_mass_for_band(self, band):
        return (10 ** (-0.4 * self.snapshot[band + "_mag"]))[self._particle_order]

    def get_rgb_masses(self):
        rgb = np.empty((len(self.snapshot), 3), dtype=np.float32)
        rgb[:, 0] = self._effective_mass_for_band('I') * 0.5
        rgb[:, 1] = self._effective_mass_for_band('V')
        rgb[:, 2] = self._effective_mass_for_band('U')
        rgb[np.isnan(rgb)] = 0.0
        return rgb

    def get_named_quantity(self, name):
        qty = self.snapshot[name]
        if len(qty.shape) == 2:
            qty = qty[:, 0]
        return qty.astype(np.float32)[self._particle_order]

    def get_quantity_names(self):
        return self.snapshot.loadable_keys()

    def get_quantity_label(self, quantity_name):
        if quantity_name is None:
            return r"density / $M_{\odot} / \mathrm{kpc}^2$"
        else:
            lunit = self.snapshot[quantity_name].units.latex()
            if lunit != "":
                lunit = "$/" + lunit + "$"
            return quantity_name + lunit

    def __len__(self):
        return len(self.snapshot)

    def get_periodicity_scale(self):
        if 'boxsize' in self.snapshot.properties:
            return float(self.snapshot.properties['boxsize'].in_units("kpc"))
        else:
            return None

    def get_filename(self):
        return self.snapshot.filename


class PynbodyDataLoader(PynbodyDataInMemory):
    """Literal data loader for pynbody (starts from just a filename)"""

    _name_smooth_array = 'topsy_smooth'

    def __init__(self, device: wgpu.GPUDevice, filename: str, center: str, particle: str,
                 take_region: Optional[pynbody.filt.Filter] = None):

        logger.info(f"Data filename = {filename}, center = {center}, particle = {particle}")
        if take_region is None:
            snapshot = pynbody.load(filename)
        else:
            snapshot = pynbody.load(filename, take_region=take_region)

        snapshot.physical_units()
        self.filename = filename

        fam = pynbody.family.get_family(particle)
        snapshot = snapshot[fam]

        self._family_name = fam.name
        logger.info("Loading position data...")
        _ = snapshot['pos'] # just trigger the load
        self.snapshot = snapshot

        self._perform_centering(center)

        super().__init__(device, snapshot)

        self._perform_smoothing()

    @property
    def _smooth_cache_filename(self):
        return f"{self.filename}-topsy-smooth-{self._family_name}.pkl"

    @property
    def _rho_cache_filename(self):
        return f"{self.filename}-topsy-rho-{self._family_name}.pkl"

    def _perform_centering(self, center):
        logger.info("Performing centering...")
        if center.startswith("halo-"):
            halo_number = int(center[5:])
            h = self.snapshot.ancestor.halos()
            pynbody.analysis.halo.center(h[halo_number], vel=False)
            self.snapshot.wrap()
        elif center == 'zoom':
            f_dm = self.snapshot.ancestor.dm
            pynbody.analysis.halo.center(f_dm[f_dm['mass'] < 1.01 * f_dm['mass'].min()])
            self.snapshot.wrap()
        elif center == 'all':
            pynbody.analysis.halo.center(self.snapshot)
        elif center == 'none':
            self.snapshot['pos']-=(self.snapshot['pos'].max(axis=0) + self.snapshot['pos'].min(axis=0))/2.0
        else:
            raise ValueError("Unknown centering type")

    def _perform_smoothing(self):
        try:
            logger.info("Looking for cached smoothing data...")
            smooth = pickle.load(open(self._smooth_cache_filename, 'rb'))
            if len(smooth) == len(self.snapshot):
                self.snapshot[self._name_smooth_array] = smooth
            else:
                raise ValueError("Incorrect number of particles in cached smoothing data")
            logger.info("...success!")
        except:
            logger.info("Generating smoothing data - this can take a while but will be cached for future runs")
            self.snapshot[self._name_smooth_array] = pynbody.sph.smooth(self.snapshot)
            try:
                pickle.dump(self.snapshot[self._name_smooth_array], open(self._smooth_cache_filename, 'wb'))
                logger.info("Smoothing data saved successfully")
            except IOError:
                logger.warning("Unable to save smoothing data to disk")


class TestDataLoader(AbstractDataLoader):
    def __init__(self, device: wgpu.GPUDevice, n_particles: int = config.TEST_DATA_NUM_PARTICLES_DEFAULT,
                 n_cells = 10, seed: int = 1337, with_cells = False, periodic = False):
        self._n_particles = n_particles
        self._gmm_weights = [0.5, 0.4, 0.1]  # should sum to 1
        self._gmm_means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, 10.0, 0.0]])
        self._gmm_std = np.array([[20.0, 20.0, 20.0], [4.0, 0.2, 4.0], [2.0, 2.0, 3.0]])

        self._gmm_pos = self._generate_samples(seed)
        self._gmm_den = self._evaluate_density(self._gmm_pos)

        self._periodic = periodic

        if with_cells:
            self._cell_layout, ordering = cell_layout.CellLayout.from_positions(self._gmm_pos, self._gmm_pos.min()-1e-3,
                                                                                self._gmm_pos.max()+1, n_cells)
            self._gmm_pos = self._gmm_pos[ordering]
            self._gmm_den = self._gmm_den[ordering]

        super().__init__(device)

    def __len__(self):
        return self._n_particles

    def _evaluate_density(self, pos):
        # simple gaussian mixture model, density of particles per unit volume
        den = np.zeros(len(pos))
        for i in range(len(self._gmm_weights)):
            den += self._gmm_weights[i] * np.exp(
                -np.sum((pos - self._gmm_means[i]) ** 2 / self._gmm_std[i] ** 2, axis=1)) \
                   / ((2 * np.pi) ** 1.5 * np.prod(self._gmm_std[i]))
        return den * self._n_particles

    def _generate_samples(self, seed):
        # simple gaussian mixture model
        np.random.seed(seed)
        pos = np.empty((self._n_particles, 3), dtype=np.float32)
        if self._n_particles == 1:
            pos[0] = self._gmm_means[0]
        else:
            offset = 0
            for i in range(len(self._gmm_weights)):
                cpt_len = int(self._n_particles * self._gmm_weights[i])
                pos[offset:offset + cpt_len] = \
                    np.random.normal(size=(cpt_len, 3), scale=1.0).astype(np.float32) * self._gmm_std[np.newaxis, i,
                                                                                        :] + self._gmm_means[i]
                offset += cpt_len
            assert offset == self._n_particles
        return np.random.permutation(pos)

    def get_positions(self):
        return self._gmm_pos

    def get_smooth(self):
        sm = 2.0 / self._gmm_den ** 0.333333
        return sm

    def get_mass(self):
        return np.repeat(np.float32(1e-8), self._n_particles)

    def get_named_quantity(self, name):
        if name == "test-quantity":
            return np.sin(self._gmm_pos[:, 0]) * np.cos(self._gmm_pos[:, 1]) * np.cos(self._gmm_pos[:, 2]) * 1e-4
        else:
            raise KeyError("Unknown quantity name")

    def get_quantity_names(self):
        return ["test-quantity"]

    def get_quantity_label(self, quantity_name):
        if quantity_name is None:
            return r"test density / $M_{\odot} / \mathrm{kpc}^2$"
        elif quantity_name == "test-quantity":
            return "test quantity"
        else:
            return "unknown"

    def get_filename(self):
        return "test data"

    def get_periodicity_scale(self):
        return 100.0 if self._periodic else None

    def get_rgb_masses(self):
        rgb = np.empty((len(self._gmm_pos), 3), dtype=np.float32)
        rgb[:, 0] = abs(np.sin(self._gmm_pos[:, 0] / 10.0))
        rgb[:, 1] = abs(np.cos(self._gmm_pos[:, 1] / 10.0))
        rgb[:, 2] = abs(np.cos(self._gmm_pos[:, 2] / 10.0))
        return rgb
