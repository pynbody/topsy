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
        self._recommendation_based_on_num_particles = 0
        self._start_index = 0
        self._max_num_particles = total_particles
        self._current_draw_reason = None
        self._last_num_to_render = 1


    def start_frame(self, draw_reason: drawreason.DrawReason):
        """Called at the start of a frame to reset the start index if needed

        Returns True if the start index was reset, False otherwise"""
        self._current_draw_reason = draw_reason
        self._first_block_in_frame = True
        if draw_reason not in (drawreason.DrawReason.PRESENTATION_CHANGE, drawreason.DrawReason.REFINE):
            self._start_index = 0
            return True
        else:
            return False

    def end_frame_get_scalefactor(self):
        """Ends a frame and returns the scale factor for the colormap"""
        self._current_draw_reason = None
        return self._max_num_particles / self._start_index

    def get_block(self, time_elapsed_in_frame: float) -> tuple[list[int], list[int]] | None:
        """Recommends the starting index and number of particles to render, or None if no rendering should be undertaken"""
        if self._current_draw_reason is None:
            raise RuntimeError("get_block called without a current frame")
        draw_reason = self._current_draw_reason
        self._last_block_start_time = time_elapsed_in_frame
        if draw_reason == drawreason.DrawReason.PRESENTATION_CHANGE:
            return None
        elif draw_reason == drawreason.DrawReason.EXPORT:
            if self._start_index == 0:
                self._last_num_to_render = self._max_num_particles
                return ([0], [self._max_num_particles])
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
            if time_available<=0.05/config.TARGET_FPS:
                return None
            else:
                num_to_render = int(self._recommended_num_particles_to_render * time_available * config.TARGET_FPS)
                if num_to_render + self._start_index > self._max_num_particles:
                    num_to_render = self._max_num_particles - self._start_index
                self._last_num_to_render = num_to_render
                return ([self._start_index], [num_to_render])

    def end_block(self, time_elapsed_in_frame: float, actual_num_rendered: int = None):
        """Report the time taken to render a number of particles, so that the next recommendation can be made"""
        num_rendered = actual_num_rendered or self._last_num_to_render
        time_taken = time_elapsed_in_frame - self._last_block_start_time
        self._start_index += num_rendered
        num_achievable = int(num_rendered / (time_taken * config.TARGET_FPS))
        if num_achievable<1:
            # very strange edge case, but must never recommend rendering less than one particle!
            num_achievable = 1

        if abs(math.log2(num_achievable) - math.log2(self._recommended_num_particles_to_render)) > 1.0:
            # substantial (factor 2) difference between what could be achieved and what was achieved
            self._recommended_num_particles_to_render = num_achievable
            self._recommendation_based_on_num_particles = num_rendered

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
        self.select_all()


    def _map_logical_range_to_actual_ranges(self, start, length):
        """Map from logical range to actual ranges in the cell layout"""
        num_particles = self._cell_layout.get_num_particles()
        fractional_start = start / num_particles
        fractional_length = length / num_particles

        starts = []
        lens = []

        num_cells = self._cell_layout.get_num_cells()

        for i in self._selected_cells:
            # the 'phase shift' ensures that if a very low number of particles are selected such that the mean
            # number of particles per cell is less than one, some particles still get selected when we are
            # starting at zero. Otherwise quantization effects would make it impossible to select any particles
            # until a much later block. Also the phase shift must be evenly distributed across cell so that
            # we don't get differential spatial effects

            cell_phase_shift = self._cell_phase_shifts[i]/num_cells
            total_particles_in_cell = self._cell_layout.get_cell_length(i)

            ideal_start_this_cell = fractional_start * total_particles_in_cell
            ideal_len_this_cell = fractional_length * total_particles_in_cell
            # the above are floating point numbers, but we can actually only take an integer, so round

            start_this_cell = int(ideal_start_this_cell + cell_phase_shift)
            end = int(ideal_start_this_cell + ideal_len_this_cell+cell_phase_shift)
            len_this_cell = end - start_this_cell

            #if i<10:
            #    print(f"{i} {cell_phase_shift} {start_this_cell}, {ideal_len_this_cell:.2f}, {len_this_cell} of {total_particles_in_cell}")

            if len_this_cell>0:
                starts.append(start_this_cell + self._cell_layout.get_cell_offset(i))
                lens.append(len_this_cell)


        return starts, lens

    def get_block(self, time_elapsed_in_frame: float) -> tuple[list[int], list[int]] | None:
        result = super().get_block(time_elapsed_in_frame)
        if result is None:
            return None
        starts, lens = result
        assert len(starts) == len(lens) == 1
        return self._map_logical_range_to_actual_ranges(starts[0], lens[0])

    def select_all(self):
        """Select all cells for inclusion in next render pass"""
        self._selected_cells = np.arange(self._cell_layout.get_num_cells())

    def select_sphere(self, cen, r):
        """Select a sphere of particles for inclusion in next render pass"""
        self._selected_cells = self._cell_layout.cells_in_sphere(cen, r)

    def get_fraction_volume_selected(self):
        """Get the number of cells selected for inclusion in next render pass"""
        return len(self._selected_cells)/self._cell_layout.get_num_cells()




