import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import numpy as np

from .overlay import Overlay


class ColorbarOverlay(Overlay):
    def __init__(self, visualizer, vmin, vmax, colormap, label, *, dpi=200, **kwargs):
        self.dpi = dpi
        self.kwargs = kwargs
        self._aspect_ratio = 0.15
        self.vmin = vmin
        self.vmax = vmax
        self.colormap = colormap
        self.label = label

        super().__init__(visualizer)

    def get_clipspace_coordinates(self, pixel_width, pixel_height):
        im = self.get_contents()
        height = 2.0
        width = 2.0*pixel_height*im.shape[1]/im.shape[0]/pixel_width
        x,y = 1.0-width,-1.0
        return x, y, width, height
    def render_contents(self):
        fig = plt.figure(figsize=(10 * self._aspect_ratio, 10), dpi=200,
                         facecolor=(1.0, 1.0, 1.0, 0.5))

        cmap = matplotlib.colormaps[self.colormap]
        cNorm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        cb1 = matplotlib.colorbar.ColorbarBase(fig.add_axes([0.05, 0.05, 0.3, 0.9]),
                                               cmap=cmap, norm=cNorm, orientation='vertical')
        cb1.set_label(self.label)

        fig.canvas.draw()
        width,height = fig.canvas.get_width_height(physical=True)

        result: np.ndarray = np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape((height,width,4)).transpose((1,0,2))
        result = result.swapaxes(0,1).astype(np.float32)/256

        plt.close(fig)
        return result
