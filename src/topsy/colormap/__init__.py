import numpy as np
import wgpu

from .implementation import ColormapBase, NoColormap, Colormap, RGBColormap, RGBHDRColormap, BivariateColormap
from .ui import ColorMapController, BivariateColorMapController, RGBMapController, GenericController
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
        self._impl: ColormapBase = self.instance_from_parameters(
            {
                'colormap_name': config.DEFAULT_COLORMAP,
                'vmin': None,
                'vmax': None,
                'log': False,
                'type': 'none',
            }, device, input_texture, output_format,
            )

    def _check_valid(self):
        if self._impl is None or isinstance(self._impl, NoColormap):
            raise ValueError("ColormapHolder is not fully initialized")

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
        all_parameters = self.get_parameters() | parameters  # merge with existing parameters
        if self._impl is None and self._class_from_parameters(all_parameters) is None:
            return # we are in an initialization phase and it's fine to have no colormap yet
        if self._impl is None or not self._impl.accepts_parameters(all_parameters):
            self._impl = self.instance_from_parameters(all_parameters, self._device, self._input_texture,
                                                       self._output_format)
            return True
        else:
            # Update the existing colormap parameters, without passing back in already known parameters
            self._impl.update_parameters(parameters)
            return False

    def get_parameter(self, name: str):
        """
        Get a parameter from the colormap.
        """
        return self._impl.get_parameter(name)

    def get_parameters(self) -> dict:
        """
        Get all parameters from the colormap.
        """
        return self._impl.get_parameters()

    def autorange(self, sph_render_output: np.ndarray):
        """Update the colormap ranges based on the provided SPH render output."""
        self._check_valid()
        self._impl.autorange_vmin_vmax(sph_render_output)

    def encode_render_pass(self, command_encoder, target_texture_view):
        """
        Encode the render pass for the colormap.
        This will set up the necessary buffers and shaders for rendering the colormap.
        """
        self._check_valid()

        self._impl.encode_render_pass(command_encoder, target_texture_view)

    def set_scaling(self, width, height, mass_scaling):
        """
        Set the scaling for the colormap.
        """
        self._check_valid()
        self._impl.set_scaling(width, height, mass_scaling)

    def sph_raw_output_to_image(self, sph_raw_output: np.ndarray) -> np.ndarray:
        """
        Convert SPH raw output to an image using the colormap.
        """
        self._check_valid()
        return self._impl.sph_raw_output_to_image(sph_raw_output)

    def sph_raw_output_to_content(self, sph_raw_output: np.ndarray) -> np.ndarray:
        """
        Convert SPH raw output to the logical content represented by the colormap.

        This is typically used for debugging or analysis purposes.
        """
        self._check_valid()
        return self._impl.sph_raw_output_to_content(sph_raw_output)

    def make_ui_controller(self, visualizer, refresh_ui_callback: Optional[callable] = None) -> GenericController:
        """
        Make a UI controller for the currently instantiated colormap.

        This is used to interact with the colormap in a user interface. The controller is an abstract
        description of the UI elements and their behavior, allowing for different implementations
        (specifically Qt or Jupyter) to render the UI accordingly.
        """
        self._check_valid()
        if isinstance(self._impl, BivariateColormap):
            return BivariateColorMapController(visualizer, refresh_ui_callback)
        elif isinstance(self._impl, RGBColormap):
            return RGBMapController(visualizer, refresh_ui_callback)
        else:
            return ColorMapController(visualizer, refresh_ui_callback)

    def __getitem__(self, key: str):
        """
        Allow dictionary-like access to colormap parameters.
        """
        return self.get_parameter(key)

    def __setitem__(self, key: str, value):
        """
        Allow dictionary-like setting of colormap parameters.
        """
        self.update_parameters({key: value})