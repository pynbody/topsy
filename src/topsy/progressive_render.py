import math

from . import config
from . import drawreason

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
        """Called at the start of a frame to reset the start index if needed"""
        self._current_draw_reason = draw_reason
        self._first_block_in_frame = True
        if draw_reason not in (drawreason.DrawReason.PRESENTATION_CHANGE, drawreason.DrawReason.REFINE):
            self._start_index = 0

    def end_frame_get_scalefactor(self):
        """Ends a frame and returns the scale factor for the colormap"""
        self._current_draw_reason = None
        return self._max_num_particles / self._start_index

    def get_block(self, time_elapsed_in_frame: float) -> tuple[int, int] | None:
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
                return 0, self._max_num_particles
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
                return self._start_index, num_to_render

    def end_block(self, time_elapsed_in_frame: float):
        """Report the time taken to render a number of particles, so that the next recommendation can be made"""
        num_rendered = self._last_num_to_render
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