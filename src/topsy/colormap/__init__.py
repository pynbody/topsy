from .implementation import Colormap, RGBColormap, RGBHDRColormap, BivariateColormap

class ColorMapHolder:
    """
    A class to hold and update the color map for a visualizer.

    This is necessary because the color map may change during the lifetime of a visualizer, and
    the logic for updating the color map is encapsulated here rather than in the visualizer itself.
    """

    def __init__(self, parameters):
        pass
