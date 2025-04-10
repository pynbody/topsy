import matplotlib.pyplot as plt
import matplotlib, matplotlib.backends.backend_agg
import matplotlib.figure as figure
import matplotlib.colors as colors
import numpy as np

from .overlay import Overlay


class ColorbarOverlay(Overlay):
    def __init__(self, visualizer, vmin, vmax, colormap, label, *, dpi_logical=72, **kwargs):
        self.dpi_logical = dpi_logical
        self.kwargs = kwargs
        self._aspect_ratio = 0.2
        self.vmin = vmin
        self.vmax = vmax
        self.colormap = colormap
        self.label = label
        self._last_width = None
        self._last_height = None

        super().__init__(visualizer)

    def get_clipspace_coordinates(self, pixel_width, pixel_height):
        im = self.get_contents()
        height = 2.0
        width = 2.0*pixel_height*im.shape[1]/im.shape[0]/pixel_width
        x,y = 1.0-width,-1.0
        if self._last_width!=pixel_width or self._last_height!=pixel_height:
            # contents is the wrong size
            self.update()
        self._last_width = pixel_width
        self._last_height = pixel_height
        return x, y, width, height
    def render_contents(self):
        dpi_physical = self.dpi_logical*self._visualizer.canvas.pixel_ratio

        fig = figure.Figure(figsize=(self._visualizer.canvas.height_physical * self._aspect_ratio/dpi_physical,
                                  self._visualizer.canvas.height_physical/dpi_physical),
                         dpi=dpi_physical,
                         facecolor=(1.0, 1.0, 1.0, 0.5))

        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)

        cmap = matplotlib.colormaps[self.colormap]
        cNorm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        cb1 = matplotlib.colorbar.ColorbarBase(fig.add_axes([0.05, 0.05, 0.3, 0.9]),
                                               cmap=cmap, norm=cNorm, orientation='vertical')
        cb1.set_label(self.label)

        fig.canvas.draw()
        width,height = fig.canvas.get_width_height(physical=True)

        result: np.ndarray = np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape((height,width,4)).transpose((1,0,2))
        result = result.swapaxes(0,1).astype(np.float32)/256

        return result
