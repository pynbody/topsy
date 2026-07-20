from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.backends.backend_agg
import matplotlib.figure as figure

from .overlay import Overlay

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer


class VectorOverlay(Overlay):
    """Overlay that renders a projected vector field as a matplotlib quiver plot.

    The vector field is read out (in later stages) from the GPU, then rendered by
    matplotlib into a transparent RGBA texture which is overplotted on top of the
    visualization but underneath the other overlays (colorbar, scalebar, ...).

    For now the field is initialized to a recognisable test pattern (a whirlpool)
    so that the rendering path can be verified.
    """

    def __init__(self, visualizer: Visualizer, *, dpi_logical=72, **kwargs):
        """Setup the vector overlay.

        :param visualizer: the visualizer instance
        :param dpi_logical: the logical matplotlib dpi; the physical dpi (and hence
            the texture resolution) is derived from this and the canvas pixel ratio
        """
        self.dpi_logical = dpi_logical
        self._last_width = None
        self._last_height = None
        self._vector_field = self._make_test_field()
        super().__init__(visualizer, **kwargs)
        self.opacity = 0.5 # the vector field is overplotted at half opacity by default

    @staticmethod
    def _make_test_field(num_arrows=24) -> np.ndarray:
        """Generate a recognisable test vector field: a circular whirlpool.

        Returns a (num_arrows, num_arrows, 2) array. Following the convention used
        elsewhere for images read back from the GPU, row 0 is the *top* of the
        image, component 0 is the x (rightward) component and component 1 is the y
        (upward) component.
        """
        i = np.arange(num_arrows)[:, np.newaxis]
        j = np.arange(num_arrows)[np.newaxis, :]

        # screen coordinates (row 0 at top => screen y decreases with row index)
        screen_x = j.astype(np.float32)
        screen_y = float(num_arrows - 1) - i.astype(np.float32)

        dx = screen_x - (num_arrows - 1) / 2.0
        dy = screen_y - (num_arrows - 1) / 2.0

        # counter-clockwise circulation
        u = -dy * np.ones_like(dx)
        v = dx * np.ones_like(dy)

        return np.stack(np.broadcast_arrays(u, v), axis=-1).astype(np.float32)

    def update_vector_field(self, vector_field: np.ndarray):
        """Update the displayed vector field and regenerate the quiver texture.

        :param vector_field: a (ny, nx, 2) array. Row 0 is the top of the image;
            component 0 is the x (rightward) component, component 1 is the y
            (upward) component.
        """
        vector_field = np.asarray(vector_field)
        if vector_field.ndim != 3 or vector_field.shape[2] != 2:
            raise ValueError("vector_field must have shape (ny, nx, 2), "
                             f"but got shape {vector_field.shape}")
        self._vector_field = vector_field.astype(np.float32)
        self.update()

    def render_contents(self) -> np.ndarray:
        field = self._vector_field
        ny, nx = field.shape[:2]

        # arrow positions in screen coordinates: row 0 at the top of the image
        col = np.arange(nx)
        row = np.arange(ny)
        x = np.tile(col, (ny, 1))
        y = (ny - 1) - np.repeat(row[:, np.newaxis], nx, axis=1)

        u = field[..., 0]
        v = field[..., 1]

        # make longest arrow have length 1.0
        # this will then be scaled to the x distance between quiver points
        magnitude = np.hypot(u, v).max()
        if magnitude <= 0.0:
            magnitude = 1.0
        u = u / magnitude
        v = v / magnitude


        # Render at the native resolution of the on-screen square. The square SPH
        # image fills the longer canvas dimension (overspilling the shorter one),
        # so the texture is sized to the larger physical canvas dimension.
        dpi_physical = self.dpi_logical * self._visualizer.canvas.pixel_ratio
        size_physical = max(self._visualizer.canvas.width_physical,
                            self._visualizer.canvas.height_physical)
        size_inches = size_physical / dpi_physical
        fig = figure.Figure(figsize=(size_inches, size_inches), dpi=dpi_physical,
                             facecolor="none")
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        # axes filling the entire figure, with no furniture whatsoever
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_facecolor("none")

        ax.quiver(x, y, u, v, color="white", pivot="mid",
                  scale=nx/2, scale_units="width", units='dots',
                  width=2.0 * self._visualizer.canvas.pixel_ratio,
                  headwidth=5.0,
                  headlength=5.0)

        # edge-to-edge: data spans exactly the arrow grid, no margins
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)

        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf.astype(np.float32) / 255.0

    def get_clipspace_coordinates(self, pixel_width, pixel_height):
        if self._last_width != pixel_width or self._last_height != pixel_height:
            # the canvas has been resized, so the texture is the wrong resolution
            self.update()
        self._last_width = pixel_width
        self._last_height = pixel_height

        # Match the square SPH image, which is displayed edge-to-edge with the
        # same aspect-ratio correction applied by the colormap shader: the shorter
        # axis spans the clip range [-1, 1] and the longer axis overspills.
        aspect_ratio = pixel_width / pixel_height
        if aspect_ratio > 1.0:
            width = 2.0
            height = 2.0 * aspect_ratio
        else:
            width = 2.0 / aspect_ratio
            height = 2.0
        return -width / 2.0, -height / 2.0, width, height
