from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import matplotlib as mpl
import abc
import logging

from .. import config, drawreason, sph

if TYPE_CHECKING:
    from .. import visualizer

logger = logging.getLogger(__name__)
    
@dataclass
class ControlSpec:
    name: str
    type: str  # 'combo' | 'combo-edit' | 'checkbox' | 'range_slider' | 'button'
    label: Optional[str] = None
    options: Optional[List[str]] = None
    value: Any = None
    range: Optional[Tuple[float, float]] = None
    callback: Callable[[Any], None] = lambda _: None

    def get_first_named_element(self, name):
        if self.name == name:
            return name
        else:
            return None

@dataclass
class LayoutSpec:
    type: str  # 'vbox' | 'hbox'
    children: List[Union['LayoutSpec', ControlSpec]]

    def get_first_named_element(self, name):
        for c in self.children:
            if result := c.get_first_named_element(name):
                return result
        return None

class GenericController(abc.ABC):
    def __init__(self, visualizer: visualizer.Visualizer, 
                 refresh_ui_callback: Callable[[LayoutSpec, bool], None]):

        self.visualizer = visualizer
        self.colormap = visualizer.colormap
        self._refresh_ui_callback = refresh_ui_callback
        self._layout_on_last_refresh = self.get_layout()

    @abc.abstractmethod
    def get_layout(self) -> LayoutSpec:
        pass

    def refresh_ui(self) -> None:
        if self._refresh_ui_callback is not None:
            current_layout = self.get_layout()
            different_widgets = self._layout_has_different_widgets(current_layout, self._layout_on_last_refresh)
            self._refresh_ui_callback(current_layout, different_widgets)
            self._layout_on_last_refresh = current_layout

    @classmethod
    def _layout_has_different_widgets(cls, layout1: LayoutSpec, layout2: LayoutSpec) -> bool:
        """Check if two layouts have different widgets."""
        if layout1.type != layout2.type or len(layout1.children) != len(layout2.children):
            return True
        for child1, child2 in zip(layout1.children, layout2.children):
            if type(child1) != type(child2):
                return True
            elif isinstance(child1, ControlSpec) and isinstance(child2, ControlSpec):
                if child1.name != child2.name or child1.type != child2.type or child1.value != child2.value:
                    return True
            elif isinstance(child1, LayoutSpec) and isinstance(child2, LayoutSpec):
                if cls._layout_has_different_widgets(child1, child2):
                    return True
            else:
                raise TypeError(f"Unexpected child type: {type(child1)} or {type(child2)}")
        return False

class ColorMapController(GenericController):
    """Controller description for standard color maps"""

    default_quantity_name = config.PROJECTED_DENSITY_NAME

    def __init__(self, visualizer, refresh_ui_callback: Optional[Callable] = None):
        super().__init__(visualizer, refresh_ui_callback)

    def get_colormap_list(self) -> List[str]:
        return list(mpl.colormaps.keys())

    def get_quantity_list(self) -> List[str]:
        names = sorted(self.visualizer.data_loader.get_quantity_names(), key=str.lower)
        return [self.default_quantity_name] + names

    def apply_auto(self) -> None:
        self.visualizer.colormap_autorange()
        self.refresh_ui()

    def apply_colormap(self, name: str) -> None:
        self.visualizer.colormap.update_parameters({'colormap_name': name})
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def apply_log_scale(self, state: bool) -> None:
        params = self.colormap.get_parameters()
        ui_range = params['ui_range_linear'] if not state else params['ui_range_log']
        self.colormap.update_parameters({
            'log': state,
            'vmin': ui_range[0],
            'vmax': ui_range[1]
        })
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)
        self.refresh_ui()

    def apply_quantity(self, name: str) -> None:
        new = None if name == self.default_quantity_name else name
        self.visualizer.quantity_name = new
        # other elements of the UI may need to be updated
        self.refresh_ui()

    def apply_slider(self, vmin: float, vmax: float) -> None:
        self.colormap.update_parameters({'vmin': vmin, 'vmax': vmax})
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def get_layout(self, suppress_range=False) -> LayoutSpec:
        params = self.visualizer.colormap.get_parameters()
        cmap = params["colormap_name"]
        qty = self.visualizer.quantity_name or self.default_quantity_name
        ui_range = params['ui_range_linear'] if not params['log'] else params['ui_range_log']

        first_row = [
                    ControlSpec("colormap", "combo", options=self.get_colormap_list(),
                                value=cmap, callback=self.apply_colormap),
                    ControlSpec("quantity", "combo-edit", options=self.get_quantity_list(),
                                value=qty, callback=self.apply_quantity),]
        if not suppress_range:
            first_row.append(ControlSpec("log", "checkbox", label="Log scale",
                                value=params['log'], callback=self.apply_log_scale),)
        children = [
            LayoutSpec("hbox", first_row)
        ]
        
        if not suppress_range:
            children.append(LayoutSpec("hbox", [
                                ControlSpec("range", "range_slider", 
                                            value=(params['vmin'], params['vmax']),
                                            range=ui_range, callback=lambda vv: self.apply_slider(*vv)),
                                ControlSpec("auto", "button", label="Auto",
                                            callback=lambda _: self.apply_auto()),
                            ]))

        return LayoutSpec(
            type="vbox",
            children=children
        )

class BivariateColorMapController(ColorMapController):
    def apply_denslider(self, vmin: float, vmax: float) -> None:
        self.colormap.update_parameters({
            'density_vmin': vmin,
            'density_vmax': vmax
        })
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def get_layout(self) -> LayoutSpec:
        layout = super().get_layout()
        params = self.colormap.get_parameters()

        den_ui_range = params['ui_range_density']
        den_range = params['density_vmin'], params['density_vmax']

        children = layout.children

        children.append(LayoutSpec("hbox", [
            ControlSpec("range_den", "range_slider",
                        value=den_range,
                        range=den_ui_range, callback=lambda vv: self.apply_denslider(*vv),
                        label="density")
            ]))

        return LayoutSpec("vbox", children=children)

class RGBMapController(GenericController):
    """Controller description for RGB (stellar rendering) outputs"""

    def __init__(self, visualizer, refresh_ui_callback: Optional[Callable] = None):
        super().__init__(visualizer, refresh_ui_callback)

    def get_state(self) -> dict:
        cmap_params = self.visualizer.colormap.get_parameters()
        return {
            "mag_range": (cmap_params['min_mag'], cmap_params['max_mag']),
            "gamma": cmap_params['gamma'],
        }

    def apply_mag_range(self, mag_pair: Tuple[float, float]) -> None:
        lo, hi = mag_pair
        self.visualizer.colormap.update_parameters({
            'min_mag': lo,
            'max_mag': hi,
        })
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def apply_gamma(self, g: float) -> None:
        self.visualizer.colormap.update_parameters({'gamma': g})
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def get_layout(self) -> LayoutSpec:
        st = self.get_state()
        return LayoutSpec(
            type="vbox",
            children=[
                ControlSpec(
                    name="mag_range",
                    type="range_slider",
                    label='mag/"^2',
                    range=(15.0,40.0),
                    value=st["mag_range"],
                    callback=lambda v: self.apply_mag_range(v),
                ),
                ControlSpec(
                    name="gamma",
                    type="slider",
                    label="gamma",
                    range=(0.25, 8.0),          # fixed slider‐domain
                    value=st["gamma"],          # initial value = current state
                    callback=lambda v: self.apply_gamma(v),
                ),
            ],
        )

class SurfaceMapController(ColorMapController):
    def set_den_cut(self, val):
        self.visualizer._sph.set_log_density_cut(val)
        self.visualizer.invalidate(drawreason.DrawReason.CHANGE)

    def set_smoothing_scale(self, val):
        self.visualizer.colormap.update_parameters(
            {'smoothing_scale': val}
        )
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    @classmethod
    def hex2rgbfloat(cls, hex_color: str) -> Tuple[float, float, float]:
        """Convert a hex color string to a tuple of floats in the range [0, 1]."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    @classmethod
    def rgbfloat2hex(cls, rgb: Tuple[float, float, float]) -> str:
        """Convert a tuple of floats in the range [0, 1] to a hex color string."""
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    def set_diffuse_lighting(self, color: str):
        """Set the diffuse lighting color for the surface rendering."""
        self.visualizer.colormap.update_parameters(
            {'light_color': self.hex2rgbfloat(color)}
        )
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def set_ambient_lighting(self, color: str):
        """Set the ambient lighting color for the surface rendering."""
        self.visualizer.colormap.update_parameters(
            {'ambient_color': self.hex2rgbfloat(color)}
        )
        self.visualizer.invalidate(drawreason.DrawReason.PRESENTATION_CHANGE)

    def get_layout(self) -> LayoutSpec:
        suppress_range = self.visualizer.quantity_name is None 

        standard_cmap_children = super().get_layout(suppress_range=suppress_range).children

        sph_ = self.visualizer._sph
        assert isinstance(sph_, sph.DepthSPHWithOcclusion)

        params = self.visualizer.colormap.get_parameters()

        cut_range = sph_.get_log_density_cut_range()
        cut_val = sph_.get_log_density_cut()
        
        lighting_spec = LayoutSpec(
            type="hbox",
            children=[
                ControlSpec(
                    name="diffuse_lighting",
                    type="color_picker",
                    label="Diffuse light",
                    value=self.rgbfloat2hex(params['light_color']),
                    callback = self.set_diffuse_lighting
                ),
                ControlSpec(
                    name="ambient_lighting",
                    type="color_picker",
                    label="Ambient light",
                    value=self.rgbfloat2hex(params['ambient_color']),
                    callback = self.set_ambient_lighting
                )
            ]
        )
        return LayoutSpec(
            type="vbox",
            children=[
                ControlSpec(
                    name="log_den_threshold",
                    type="slider",
                    label="Density threshold",
                    range=cut_range,
                    value=cut_val,
                    callback = self.set_den_cut
                ),
                ControlSpec(
                    name="smoothing_scale",
                    type="slider",
                    label="Surface smoothing",
                    range=(0.0, 0.05),
                    value=params['smoothing_scale'],
                    callback=self.set_smoothing_scale
                ),
                lighting_spec,
            ] + standard_cmap_children
        )
    
class UnifiedColorMapController(GenericController):
    """Class that implements a dropdown to choose different render modes and then presents the controls for that mode"""
    def __init__(self, visualizer: visualizer.Visualizer,
                 refresh_ui_callback: Optional[Callable] = None):
        super().__init__(visualizer, refresh_ui_callback)
        self._controller: GenericController = self._get_controller_for_mode(visualizer.render_mode)
        
        

    def _get_controller_for_mode(self, mode: str) -> GenericController:
        if mode in ['univariate', 'density']:
            return ColorMapController(self.visualizer, self._refresh_ui_callback_wrapper)
        elif mode == 'bivariate':
            return BivariateColorMapController(self.visualizer, self._refresh_ui_callback_wrapper)
        elif mode in ['rgb', 'rgb-hdr']:
            return RGBMapController(self.visualizer, self._refresh_ui_callback_wrapper)
        elif mode == 'surface':
            return SurfaceMapController(self.visualizer, self._refresh_ui_callback_wrapper)
        else:
            raise ValueError(f"Unknown render mode: {mode}")
        
    def _update_mode(self, mode: str) -> None:
        """Update the controller to the new mode and refresh the UI."""
        try:
            self.visualizer.render_mode = mode
            self._controller = self._get_controller_for_mode(self.visualizer.render_mode)
        except ValueError as e:
            logger.error(f"Failed to set render mode: {e}")
        
        self.refresh_ui()

    def _get_mode_dropdown_element(self) -> ControlSpec:
        """Get the dropdown element for selecting the render mode."""
        mode_list = ['univariate', 'bivariate', 'rgb', 'rgb-hdr', 'surface']
        return ControlSpec(
            name="render_mode",
            type="combo",
            options=mode_list,
            value=self.visualizer.render_mode,
            callback=self._update_mode
        )
    
    def _refresh_ui_callback_wrapper(self, root_spec: LayoutSpec, new_widgets: bool) -> None:
        """Wrapper for when child controls need refreshing"""
        # possible optimization: could wrap the layout here rather than getting it again
        self._refresh_ui_callback(self._add_mode_dropdown_to_controls(root_spec), new_widgets)

    def get_layout(self) -> LayoutSpec:
        if hasattr(self, '_controller'):
            map_controls = self._controller.get_layout()
        else:
            map_controls = LayoutSpec(
                type="vbox",
                children=[ControlSpec(name="placeholder", type="label", value="No controls available for this mode")]
            )
        return self._add_mode_dropdown_to_controls(map_controls)

    def _add_mode_dropdown_to_controls(self, map_controls):
        mode_dropdown = self._get_mode_dropdown_element()
        return LayoutSpec(
            type="vbox",
            children=[
                mode_dropdown,
                map_controls
            ]
        )
