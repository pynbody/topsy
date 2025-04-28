from io import BytesIO

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from .overlay import Overlay


class TextOverlay(Overlay):
    def __init__(self, visualizer, text, clipspace_origin, logical_pixels_height, *, dpi=200, **kwargs):
        self.text = text
        self.dpi = dpi
        self.clipspace_origin = clipspace_origin
        self.pixelspace_height = logical_pixels_height
        self.kwargs = kwargs

        super().__init__(visualizer)

    def get_clipspace_coordinates(self, width, height):
        im = self.get_contents()
        x,y = self.clipspace_origin
        height = self.pixelspace_height*self._visualizer.canvas.pixel_ratio/height
        width = self.pixelspace_height*self._visualizer.canvas.pixel_ratio*im.shape[1]/im.shape[0]/width
        return x, y, width, height

    def render_contents(self):
        return self.text_to_rgba(self.text, dpi=self.dpi, **self.kwargs)

    @staticmethod
    def text_to_rgba(s, *, dpi, **kwargs):
        """Render text to RGBA image.

        Based on
        https://matplotlib.org/stable/gallery/text_labels_and_annotations/mathtext_asarray.html"""
        fig = Figure(facecolor="none")
        fig.text(0, 0, s, **kwargs)
        with BytesIO() as buf:
            fig.savefig(buf, dpi=dpi, format="png", bbox_inches="tight",
                        pad_inches=0)
            buf.seek(0)
            rgba = plt.imread(buf)
        return rgba