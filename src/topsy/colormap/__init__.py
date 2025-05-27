import wgpu

from .implementation import ColormapBase, Colormap, RGBColormap, RGBHDRColormap, BivariateColormap, WeightedColormap

class ColorMapHolder:
    """
    A class to hold and update the color map for a visualizer.

    This is necessary because the color map may change during the lifetime of a visualizer, and
    the logic for updating the color map is encapsulated here rather than in the visualizer itself.
    """

    def __init__(self, parameters):
        self._parameters = parameters

    @classmethod
    def _class_from_parameters(cls, parameters) -> type[ColormapBase]:
        hdr = parameters.get("hdr", False)
        if parameters["type"] == "rgb":
            if hdr:
                return RGBHDRColormap
            else:
                return RGBColormap
        elif parameters["type"] == "bivariate":
            if hdr:
                raise ValueError("HDR is not supported for bivariate colormaps")
            return BivariateColormap
        elif parameters["type"] == "weighted":
            if hdr:
                raise ValueError("HDR is not supported for weighted colormaps")
            return WeightedColormap
        elif parameters["type"] == "density":
            if hdr:
                raise ValueError("HDR is not supported for density colormaps")
            return Colormap
        else:
            raise ValueError(f"Unknown colormap type: {parameters['type']}")

    @classmethod
    def _instance_from_parameters(cls, parameters, input_texture: wgpu.GPUTexture, output_format: wgpu.TextureFormat) -> ColormapBase:
        colormap_class = cls._class_from_parameters(parameters)
        return colormap_class(input_texture, output_format)
