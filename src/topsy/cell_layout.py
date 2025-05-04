"""Classes to keep track of the cellular layout of a simulation"""

import numpy as np
import pynbody

from pynbody.filt import geometry_selection

class CellLayout:
    """Class to keep track of segmentation of a simulation into cells"""
    def __init__(self, centres: np.ndarray, offsets: np.ndarray, lengths: np.ndarray):
        self._centres = np.ascontiguousarray(centres)
        self._offsets = offsets
        self._lengths = lengths
        self._num_particles = lengths.sum()
        self._cell_size = np.linalg.norm(self._centres[1]-self._centres[0])

    def randomize_within_cells(self):
        """Get a reordering of the particles which randomizes the order within cells, but leaves the cell structure"""
        total_len = self._lengths.sum()
        reordering = np.empty(total_len, dtype=np.uintp)
        for offset, length in zip(self._offsets, self._lengths):
            # randomize the order of the particles within this cell
            reordering[offset:offset+length] = np.random.permutation(length) + offset
        return reordering

    def cells_in_sphere(self, centre: tuple[float, float, float], radius: float) -> np.ndarray:
        """Get the indices of the cells that are within a sphere of given centre and radius"""
        expand_radius = self._cell_size*np.sqrt(3.0)
        offsets = self._centres - centre
        selection = np.linalg.norm(offsets, axis=1) < (radius + expand_radius)
        return np.where(selection)[0]

    def cell_index_from_offset(self, offset: int) -> int:
        """Get the cell index from the offset of a particle"""

        cell_index = np.searchsorted(self._offsets, offset, side='right') - 1
        if cell_index < 0 or cell_index >= len(self._lengths):
            raise ValueError("Offset is out of bounds")
        return cell_index

    def cell_slice(self, cell_index: int) -> slice:
        """Get the indices of the particles in a given cell"""
        start = self._offsets[cell_index]
        end = start + self._lengths[cell_index]
        return slice(start, end)

    def get_num_cells(self):
        """Get the total number of cells"""
        return len(self._lengths)

    def get_num_particles(self):
        """Get the total number of particles"""
        return self._num_particles

    def get_cell_length(self, cell_index: int | np.ndarray[int]) -> int | np.ndarray[int]:
        """Get the length of a given cell"""
        return self._lengths[cell_index]

    def get_cell_offset(self, cell_index: int) -> int:
        """Get the offset of a given cell"""
        return self._offsets[cell_index]

    @classmethod
    def from_positions(cls, particle_positions: np.ndarray, box_min: float, box_max: float, nside: int):
        """Create a CellLayout object from the positions of the particles with arbitrary ordering

        Parameters
        ----------

        particle_positions: array of the positions of the particles (Nx3)
        box_min: minimum coordinate of the box (for all 3 dimensions)
        box_max: maximum coordinate of the box (for all 3 dimensions)
        nside: number of cells in each of the 3 dimensions, e.g. nside=10 implies 10^3 cells in total

        Returns
        -------

        cell_layout, particle_ordering:
            cell_layout: CellLayout object
            particle_ordering: array of the ordering of the particles to put them into the cells
        """

        if particle_positions.min()<box_min or particle_positions.max()>=box_max:
            raise ValueError("Particle positions are outside the box")

        # get the cell size
        cell_size = (box_max - box_min) / nside

        # get the centre of the first cell
        cell_cen0 = box_min + cell_size / 2

        # get the cell centres
        centres = np.mgrid[cell_cen0:box_max:cell_size,
                           cell_cen0:box_max:cell_size,
                           cell_cen0:box_max:cell_size].reshape(3, -1).T

        # figure out the cell x,y,z indices of each particle
        pos_indices = np.floor((particle_positions - box_min) / cell_size).astype(np.intp)

        if pos_indices.min() < 0 or pos_indices.max() >= nside:
            raise ValueError("Particle positions are too close to edge of box; expand box size")

        # get the cell index of each particle
        cell_indices = pos_indices[:, 2] + nside * (pos_indices[:, 1] + nside * pos_indices[:, 0])
        # sort the particles by cell index
        ordering = np.argsort(cell_indices)

        # figure out the segmentation
        lengths = np.bincount(cell_indices, minlength=nside ** 3)
        assert len(lengths) == len(centres), "Logic error within from_positions"

        offsets = np.cumsum(lengths) - lengths

        return cls(centres, offsets, lengths), ordering