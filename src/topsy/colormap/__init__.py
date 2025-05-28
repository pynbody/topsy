import numpy as np
import wgpu

from .implementation import ColormapBase, Colormap, RGBColormap, RGBHDRColormap, BivariateColormap, WeightedColormap
from .. import config

from typing import Iterator, Optional

class ColormapHolder:
    """
    A class to hold and update the color map for a visualizer.

    This is necessary because the color map may change during the lifetime of a visualizer, and
    the logic for updating the color map is encapsulated here rather than in the visualizer itself.
    """

    def __init__(self, device: wgpu.GPUDevice, input_texture: wgpu.GPUTexture, output_format: wgpu.TextureFormat):
        self._device = device
        self._input_texture = input_texture
        self._output_format = output_format
        self._colormap: Optional[ColormapBase] = self.instance_from_parameters(
            {
                'colormap_name': config.DEFAULT_COLORMAP,
                'vmin': 0,
                'vmax': 1,
                'log_scale': False,
                'type': 'density'
            }, device, input_texture, output_format,
            )

    @classmethod
    def _iter_classes(cls, base_class=ColormapBase) -> Iterator[ColormapBase] :
        """
        Iterate over all subclasses of ColormapBase that match the given parameters.
        """
        for subclass in base_class.__subclasses__():
            yield subclass
            yield from cls._iter_classes(subclass)


    @classmethod
    def _class_from_parameters(cls, parameters) -> Optional[type[ColormapBase]]:
        for cl in cls._iter_classes():
            if cl.accepts_parameters(parameters):
                return cl

        return None

    @classmethod
    def instance_from_parameters(cls, parameters, device: wgpu.GPUDevice, input_texture: wgpu.GPUTexture,
                                  output_format: wgpu.TextureFormat) -> ColormapBase:
        colormap_class = cls._class_from_parameters(parameters)
        if colormap_class is None:
            raise ValueError(f"No colormap class found for parameters: {parameters}")
        return colormap_class(device, input_texture, output_format, parameters)

    def update_parameters(self, parameters: dict):
        """
        Update the colormap parameters and recreate the colormap if necessary.

        Returns True if the colormap was recreated, False if it was updated in place.
        """
        parameters = self.get_parameters() | parameters  # merge with existing parameters
        if self._colormap is None and self._class_from_parameters(parameters) is None:
            return # we are in an initialization phase and it's fine to have no colormap yet
        if self._colormap is None or not self._colormap.accepts_parameter_update(parameters):
            self._colormap = self.instance_from_parameters(parameters, self._device, self._input_texture,
                                                           self._output_format)
            return True
        else:
            self._colormap.update_parameters(parameters)
            return False

    def get_parameter(self, name: str):
        """
        Get a parameter from the colormap.
        """

        if self._colormap is None:
            raise ValueError("Colormap not set")
        return self._colormap.get_parameter(name)

    def get_parameters(self) -> dict:
        """
        Get all parameters from the colormap.
        """
        if self._colormap is None:
            raise ValueError("Colormap not set")
        return self._colormap.get_parameters()

    def get_parameter_ui_range(self, name: str) -> tuple[float, float]:
        """
        Get the UI range for a parameter from the colormap.
        """
        if self._colormap is None:
            raise ValueError("Colormap not set")
        return self._colormap.get_parameter_ui_range(name)

    def autorange(self, sph_render_output: np.ndarray):
        """Update the colormap ranges based on the provided SPH render output."""
        if self._colormap is None:
            raise ValueError("Colormap not set")
        self._colormap.autorange_vmin_vmax(sph_render_output)

    def encode_render_pass(self, command_encoder, target_texture_view):
        """
        Encode the render pass for the colormap.
        This will set up the necessary buffers and shaders for rendering the colormap.
        """
        if self._colormap is None:
            raise ValueError("Colormap not set")

        self._colormap.encode_render_pass(command_encoder, target_texture_view)

    def set_scaling(self, width, height, mass_scaling):
        """
        Set the scaling for the colormap.
        """
        if self._colormap is None:
            raise ValueError("Colormap not set")
        self._colormap.set_scaling(width, height, mass_scaling)