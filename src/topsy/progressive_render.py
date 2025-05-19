import math
import numpy as np

from . import config
from . import drawreason
from .cell_layout import CellLayout

class RenderProgression:
    """Recommends the block of particles to SPH render, taking into account historical timing"""
    def __init__(self, total_particles, initial_particles = None):
        if initial_particles is None:
            initial_particles = int(config.INITIAL_PARTICLES_TO_RENDER)
        self._recommended_num_particles_to_render = min(initial_particles, total_particles)
        self._start_index = 0
        self._max_num_particles = total_particles
        self._current_draw_reason = None
        self._last_num_to_render = 1

    def get_max_particle_regions_per_block(self):
        """Get the maximum number of particle regions that will be returned by get_block"""
        return 1

    def start_frame(self, draw_reason: drawreason.DrawReason):
        """Called at the start of a frame to reset the start index if needed

        Returns
        -------

        update_particle_ranges: bool
            Whether the particle ranges need to be updated before render command is issued

        """
        self._current_draw_reason = draw_reason
        self._first_block_in_frame = True
        self._total_num_rendered_in_frame = 0
        if draw_reason not in (drawreason.DrawReason.PRESENTATION_CHANGE, drawreason.DrawReason.REFINE):
            self._start_index = 0
            return True
        else:
            return False

    def end_frame_get_scalefactor(self):
        """Ends a frame and returns the scale factor for the colormap"""
        self._perform_particle_number_update()
        self._current_draw_reason = None
        return self._max_num_particles / self._start_index

    def get_block(self, time_elapsed_in_frame: float) -> tuple[list[int], list[int]] | None:
        """Returns a list of starting indicies and lengths from the particle buffer to render"""
        if self._current_draw_reason is None:
            raise RuntimeError("get_block called without a current frame")
        draw_reason = self._current_draw_reason
        if draw_reason == drawreason.DrawReason.PRESENTATION_CHANGE:
            return None
        elif draw_reason == drawreason.DrawReason.EXPORT:
            if self._start_index < self._max_num_particles:
                try_rendering = self._max_num_particles - self._start_index
                max_to_render = int(config.MAX_PARTICLES_PER_EXPORT_RENDERCALL / self.get_fraction_volume_selected())
                if try_rendering > max_to_render:
                    try_rendering = max_to_render
                self._last_num_to_render = try_rendering
                return ([self._start_index], [try_rendering])
            else:
                return None
        else:
            if self._start_index >= self._max_num_particles:
                return None

            if self._first_block_in_frame:
                time_available = 1./config.TARGET_FPS
                self._first_block_in_frame = False
            else:
                time_available = 1./config.TARGET_FPS - time_elapsed_in_frame

            if time_available<=0.4/config.TARGET_FPS:
                return None
                # while we could set a time_available based on the time taken to render the last block here, we
                # no longer attempt to do multiple blocks per frame in interactive context,
                # as it leads to unpredictable frame rates and stuttering. Instead, we will just render one block per
                # frame. We'll correct any mismatch in particle number on the next frame (possibly a REFINE frame).

            num_to_render = int(self._recommended_num_particles_to_render * time_available * config.TARGET_FPS)
            if num_to_render + self._start_index > self._max_num_particles:
                num_to_render = self._max_num_particles - self._start_index
            self._last_num_to_render = num_to_render
            return ([self._start_index], [num_to_render])

    def _perform_particle_number_update(self):
        num_achievable = int(self._total_num_rendered_in_frame / (self._time_in_frame * config.TARGET_FPS))
        num_achievable = min(num_achievable, self._max_num_particles)
        if num_achievable < 1:
            # very strange edge case, but must never recommend rendering less than one particle!
            num_achievable = 1

        if self._current_draw_reason != drawreason.DrawReason.REFINE:
            log2_change = abs(math.log2(num_achievable) - math.log2(self._recommended_num_particles_to_render))
            if log2_change > 1.5:
                # emergency situation, update completely
                self._recommended_num_particles_to_render = num_achievable
            elif log2_change > 0.3:
                # modest mismatch, make a more cautious update
                self._recommended_num_particles_to_render = int(
                    num_achievable ** 0.3 * self._recommended_num_particles_to_render ** 0.7)


    def end_block(self, time_elapsed_in_frame: float):
        """Report the time taken to render a number of particles, so that the next recommendation can be made"""
        self._start_index += self._last_num_to_render
        self._total_num_rendered_in_frame += self._last_num_to_render
        self._time_in_frame = time_elapsed_in_frame

        """ if self._current_draw_reason == drawreason.DrawReason.REFINE:
            # the next frame needs to update the GPU render particle ranges, come what may
            self._update_particle_ranges = True
        elif (abs(math.log2(num_achievable) - math.log2(self._recommended_num_particles_to_render)) > 0.6):
            if was_first_block_in_frame:
                # only update based on the first block in the frame (which is the one which is most critical to
                # get to roughly the right length)
                self._recommended_num_particles_to_render = num_achievable
                self._update_particle_ranges = True
        else:
            self._update_particle_ranges = False"""

    def needs_refine(self):
        """Check if the render progression is not yet complete"""
        needs_refine = self._start_index < self._max_num_particles
        return needs_refine

    def select_sphere(self, cen, radius):
        # default render progression has no way to select a sphere of particles
        pass

    def select_all(self):
        pass

    def get_fraction_volume_selected(self):
        return 1.0


class RenderProgressionWithCells(RenderProgression):
    def __init__(self, cell_layout: CellLayout, total_particles: int, initial_particles=None):
        super().__init__(total_particles, initial_particles)
        self._cell_layout = cell_layout
        random_state = np.random.RandomState(1337)
        self._cell_phase_shifts = random_state.permutation(self._cell_layout.get_num_cells())
        self._selected_cells_hash = 0
        self.select_all()

    def get_max_particle_regions_per_block(self):
        return self._cell_layout.get_num_cells()

    def _map_logical_range_to_actual_ranges(self, start, length):
        """Map from logical range to actual ranges in the cell layout.

        Performance critical - called during rendering. Optimized into numpy array operations but could
        probably be usefully optimized further based on profiling. (Although should double check
        in the profile that it really is *this* routine that is taking the time.)
        """
        num_particles = self._cell_layout.get_num_particles()
        fractional_start = start / num_particles
        fractional_end = (start + length) / num_particles

        num_cells = self._cell_layout.get_num_cells()
        offset_per_cell = self._cell_layout._offsets

        # the 'phase shift' ensures that if a very low number of particles are selected such that the mean
        # number of particles per cell is less than one, some particles still get selected when we are
        # starting at zero. Otherwise quantization effects would make it impossible to select any particles
        # until a much later block. Also the phase shift must be evenly distributed across cell so that
        # we don't get differential spatial effects
        cell_phase_shifts = self._cell_phase_shifts/num_cells
        total_particles_in_cells = self._cell_layout._lengths

        ideal_start_per_cell = fractional_start*total_particles_in_cells.astype(np.float64)
        ideal_end_per_cell = fractional_end*total_particles_in_cells.astype(np.float64)

        # the above are floating point numbers, but we can actually only take an integer, so floor it
        start_per_cell = (ideal_start_per_cell+cell_phase_shifts).astype(np.intp)
        end_per_cell = (ideal_end_per_cell+cell_phase_shifts).astype(np.intp)
        len_per_cell = end_per_cell-start_per_cell

        start_global = (start_per_cell + offset_per_cell)[self._selected_cells]
        len_global = len_per_cell[self._selected_cells]

        mask = len_global>0

        return start_global[mask], len_global[mask]



    def get_block(self, time_elapsed_in_frame: float) -> tuple[list[int], list[int]] | None:
        result = super().get_block(time_elapsed_in_frame)
        if result is None:
            return None
        starts, lens = result
        assert len(starts) == len(lens) == 1
        if lens[0] == self._max_num_particles:
            return starts, lens
        else:
            return self._map_logical_range_to_actual_ranges(starts[0], lens[0])

    def select_all(self):
        """Select all cells for inclusion in next render pass"""
        self._selected_cells = np.arange(self._cell_layout.get_num_cells())
        self._check_cells_for_update()

    def select_sphere(self, cen, r):
        """Select a sphere of particles for inclusion in next render pass"""
        self._selected_cells = self._cell_layout.cells_in_sphere(cen, r)
        self._check_cells_for_update()

    def _check_cells_for_update(self):
        sc_hash = hash(self._selected_cells.tobytes())
        if sc_hash != self._selected_cells_hash:
            self._selected_cells_hash = sc_hash
            self._update_particle_ranges = True

    def get_fraction_volume_selected(self):
        """Get the number of cells selected for inclusion in next render pass"""
        return max(1,len(self._selected_cells))/self._cell_layout.get_num_cells()




