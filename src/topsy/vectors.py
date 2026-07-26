from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.backends.backend_agg
import matplotlib.figure as figure
import wgpu
from pynbody.plot.util import PynbodyQuiverKey

from .overlay import Overlay
from .util import quantize_to_1_2_5

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer


def _quiver_style(pixel_ratio):
    """Return the keyword arguments defining the appearance of the arrows.

    Shared between the vector field itself and its key, so that the key arrow
    looks exactly like the arrows it is describing.

    Everything is anchored to physical pixels ("dots" at the figure's physical
    dpi): both the arrow thickness (``units="dots"``) and, crucially, the arrow
    length (``scale_units="dots"``, so that the quiver ``scale`` is in vector
    units per physical pixel). This lets the field and its key share a single
    scale factor and produce identically-sized arrows.

    :param pixel_ratio: the canvas pixel ratio, used to scale the arrow thickness
    """
    return dict(color="white", pivot="mid", scale_units="dots", units="dots",
                width=2.0 * pixel_ratio, headwidth=5.0, headlength=5.0)


class VectorOverlay(Overlay):
    """Overlay that renders a projected vector field as a matplotlib quiver plot.

    The vector field is read out from the GPU, then rendered by matplotlib into
    a transparent RGBA texture which is overplotted on top of the visualization
    but underneath the other overlays (colorbar, scalebar, ...).
    """

    # The longest arrow the auto-scaling aims for, expressed as a multiple of the
    # spacing between neighbouring arrows. The reference velocity is snapped to a
    # nice number near the field maximum, so the actual longest arrow can be up to
    # ~1.6x this; kept below ~1.5 cells to avoid arrows overrunning their neighbours.
    _target_max_arrow_length_in_cells = 1.5

    def __init__(self, visualizer: Visualizer, *, dpi_logical=72, **kwargs):
        """Setup the vector overlay.

        :param visualizer: the visualizer instance
        :param dpi_logical: the logical matplotlib dpi; the physical dpi (and hence
            the texture resolution) is derived from this and the canvas pixel ratio
        """
        self.dpi_logical = dpi_logical
        self._last_width = None
        self._last_height = None
        self._vector_field = None

        # The scaling relating vector magnitudes to on-screen arrow lengths. These
        # are (re)computed each time the field is rendered, and read by the vector
        # key so that it can draw an arrow of matching scale.
        self.reference_value = 1.0 # a nice-number vector magnitude to label the key with
        self.vector_units_per_dot = 1.0 # quiver scale: vector magnitude per physical pixel

        super().__init__(visualizer, **kwargs)
        self.opacity = 0.5 # the vector field is overplotted at half opacity by default

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

    def _update_scaling(self, max_magnitude, target_max_length_dots):
        """Pick the vector-magnitude-to-pixels scaling for the current field.

        A nice round magnitude (of the form {1,2,5}x10^N) near the field's maximum
        is chosen as the reference; the scaling is set so that this reference maps
        exactly to ``target_max_length_dots`` physical pixels. The longest actual
        arrow (at ``max_magnitude``) is then close to, but not exactly, the target
        length -- the same trade-off the scalebar makes to get a round label.

        Results are stored on ``self.reference_value`` and
        ``self.vector_units_per_dot`` (the latter is the quiver ``scale``).

        :param max_magnitude: the largest vector magnitude present in the field
        :param target_max_length_dots: the arrow length, in physical pixels, that
            the reference magnitude should map to
        """
        if not np.isfinite(max_magnitude) or max_magnitude <= 0.0:
            # a degenerate (empty or zero) field: keep unit scaling so the
            # zero-length arrows render harmlessly
            self.reference_value = 1.0
            self.vector_units_per_dot = 1.0
            return

        self.reference_value = quantize_to_1_2_5(max_magnitude, mode="nearest")
        self.vector_units_per_dot = self.reference_value / target_max_length_dots

    @property
    def reference_length_dots(self) -> float:
        """On-screen length, in physical pixels, of an arrow of ``reference_value``.

        This is the length the vector key should give its arrow so that it matches
        the field arrows drawn at the same scale.
        """
        return self.reference_value / self.vector_units_per_dot

    def render_contents(self) -> np.ndarray:
        if self._vector_field is None:
            # no field to render: return a single transparent pixel
            return np.zeros((1, 1, 4), dtype=np.float32)

        field = self._vector_field
        ny, nx = field.shape[:2]

        # arrow positions in screen coordinates: row 0 at the top of the image
        col = np.arange(nx)
        row = np.arange(ny)
        x = np.tile(col, (ny, 1))
        y = (ny - 1) - np.repeat(row[:, np.newaxis], nx, axis=1)

        u = field[..., 0]
        v = field[..., 1]

        # Render at the native resolution of the on-screen square. The square SPH
        # image fills the longer canvas dimension (overspilling the shorter one),
        # so the texture is sized to the larger physical canvas dimension.
        dpi_physical = self.dpi_logical * self._visualizer.canvas.pixel_ratio
        size_physical = max(self._visualizer.canvas.width_physical,
                            self._visualizer.canvas.height_physical)
        size_inches = size_physical / dpi_physical

        # Choose the vector-magnitude-to-pixels scaling. One arrow-grid cell spans
        # size_physical/nx physical pixels; we want the longest arrow to be about
        # _target_max_arrow_length_in_cells of those, but referenced to a nice
        # round magnitude so the (later) key reads e.g. "100 km/s" rather than an
        # arbitrary number. See _update_scaling.
        cell_size_dots = size_physical / nx
        self._update_scaling(np.hypot(u, v).max(),
                             self._target_max_arrow_length_in_cells * cell_size_dots)

        fig = figure.Figure(figsize=(size_inches, size_inches), dpi=dpi_physical,
                             facecolor="none")
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        # axes filling the entire figure, with no furniture whatsoever
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_facecolor("none")

        ax.quiver(x, y, u, v, scale=self.vector_units_per_dot,
                  **_quiver_style(self._visualizer.canvas.pixel_ratio))

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


class VectorKeyOverlay(Overlay):
    """Overlay that renders a scale reference (key) for the vector field.

    A single arrow of known length is drawn, with a label giving the physical
    quantity it corresponds to. It sits at the bottom of the viewport, just to
    the left of the colorbar region, so that it lines up with the scalebar at
    the bottom left.
    """

    # The figure is deliberately oversized and the result is cropped to the key
    # itself, so that the on-screen size is exactly the size of the key however
    # long the label happens to be.
    _figsize_logical = (4.0, 1.5) # inches at dpi_logical
    _padding_logical = 8 # gap (in logical pixels) between the key and the colorbar
    _y0_clipspace = -0.9 # bottom of the key, matching the scalebar

    def __init__(self, visualizer: Visualizer, *, dpi_logical=72,
                 label='100 km/s', key_length_dots=40.0, **kwargs):
        """Setup the vector key overlay.

        :param visualizer: the visualizer instance
        :param dpi_logical: the logical matplotlib dpi; the physical dpi (and hence
            the texture resolution) is derived from this and the canvas pixel ratio
        :param label: the text describing the length of the key arrow
        :param key_length_dots: the on-screen length of the key arrow, in physical
            pixels; normally driven by the vector field's scaling
        """
        self.dpi_logical = dpi_logical
        self._label = label
        self._key_length_dots = key_length_dots
        self._last_pixel_ratio = None
        super().__init__(visualizer, **kwargs)

    @property
    def label(self):
        """The text describing the length of the key arrow."""
        return self._label

    @label.setter
    def label(self, value):
        if value != self._label:
            self._label = value
            self.update()

    @property
    def key_length_dots(self):
        """The on-screen length of the key arrow, in physical pixels (dots).

        Set this to the vector field's ``reference_length_dots`` so that the key
        arrow matches the field arrows. Setting it only triggers a re-render if the
        value actually changes.
        """
        return self._key_length_dots

    @key_length_dots.setter
    def key_length_dots(self, value):
        if value != self._key_length_dots:
            self._key_length_dots = value
            self.update()

    def _colorbar_width_clipspace(self, pixel_width, pixel_height):
        """Return the width of the region occupied by the colorbar, in clip space."""
        colorbar = getattr(self._visualizer, "_colorbar", None)
        if colorbar is None or not self._visualizer.show_colorbar:
            return 0.0
        im = colorbar.get_contents()
        return 2.0 * pixel_height * im.shape[1] / im.shape[0] / pixel_width

    def get_clipspace_coordinates(self, pixel_width, pixel_height):
        if self._last_pixel_ratio != self._visualizer.canvas.pixel_ratio:
            # the texture was rendered for a different device pixel ratio
            self.update()
        self._last_pixel_ratio = self._visualizer.canvas.pixel_ratio

        # the contents is cropped to the key, so its size is the on-screen size
        im = self.get_contents()
        width = 2.0 * im.shape[1] / pixel_width
        height = 2.0 * im.shape[0] / pixel_height

        padding = 2.0 * self._padding_logical * self._visualizer.canvas.pixel_ratio / pixel_width
        x = 1.0 - self._colorbar_width_clipspace(pixel_width, pixel_height) - padding - width

        return x, self._y0_clipspace, width, height

    def render_contents(self) -> np.ndarray:
        pixel_ratio = self._visualizer.canvas.pixel_ratio
        dpi_physical = self.dpi_logical * pixel_ratio

        fig = figure.Figure(figsize=self._figsize_logical, dpi=dpi_physical,
                            facecolor="none")
        matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_facecolor("none")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        # The key arrow length is set by the quiver's scale which (with
        # scale_units='dots', see _quiver_style) is in data units per physical
        # pixel. A unit-magnitude reference arrow of scale 1/key_length_dots is
        # therefore key_length_dots pixels long. A dummy arrow is placed outside
        # the axes limits so that it is clipped away; only the key itself is seen.
        key_value = 1.0
        quiver = ax.quiver([2.0], [2.0], [key_value], [0.0],
                           scale=key_value / self._key_length_dots,
                           **_quiver_style(pixel_ratio))

        key = PynbodyQuiverKey(quiver, 0.5, 0.5, key_value, self._label,
                               coordinates='axes', labelpos='N',
                               color="white", labelcolor="white",
                               boxfacecolor=(0.0, 0.0, 0.0, 0.5),
                               boxedgecolor="white")
        ax.add_artist(key)

        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0
        return self._crop_to_contents(buf)

    @staticmethod
    def _crop_to_contents(image: np.ndarray) -> np.ndarray:
        """Crop an RGBA image down to the bounding box of its non-transparent pixels."""
        rows, cols = np.nonzero(image[..., 3] > 0.0)
        if len(rows) == 0:
            return image
        return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
