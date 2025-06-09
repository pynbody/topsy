from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
import matplotlib as mpl
import abc

from .. import config, drawreason

@dataclass
class ControlSpec:
    name: str
    type: str  # 'combo' | 'combo-edit' | 'checkbox' | 'range_slider' | 'button'
    label: Optional[str] = None
    options: Optional[List[str]] = None
    value: Any = None
    range: Optional[Tuple[float, float]] = None
    callback: Callable[[Any], None] = lambda _: None

@dataclass
class LayoutSpec:
    type: str  # 'vbox' | 'hbox'
    children: List[Union['LayoutSpec', ControlSpec]]

class GenericController(abc.ABC):
    def __init__(self, visualizer, refresh_ui_callback: Optional[Callable] = None):
        self.visualizer = visualizer
        self.colormap = visualizer.colormap
        self._refresh_ui_callback = refresh_ui_callback

    @abc.abstractmethod
    def get_layout(self) -> LayoutSpec:
        pass

    def refresh_ui(self) -> None:
        if self._refresh_ui_callback is not None:
            self._refresh_ui_callback()

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

    def get_layout(self) -> LayoutSpec:
        params = self.visualizer.colormap.get_parameters()
        cmap = params["colormap_name"]
        qty = self.visualizer.quantity_name or self.default_quantity_name
        ui_range = params['ui_range_linear'] if not params['log'] else params['ui_range_log']

        return LayoutSpec(
            type="vbox",
            children=[
                LayoutSpec("hbox", [
                    ControlSpec("colormap", "combo", options=self.get_colormap_list(),
                                value=cmap, callback=self.apply_colormap),
                    ControlSpec("quantity", "combo-edit", options=self.get_quantity_list(),
                                value=qty, callback=self.apply_quantity),
                    ControlSpec("log", "checkbox", label="Log scale",
                                value=params['log'], callback=self.apply_log_scale),
                ]),
                LayoutSpec("hbox", [
                    ControlSpec("range", "range_slider", value=(params['vmin'], params['vmax']),
                                range=ui_range, callback=lambda vv: self.apply_slider(*vv)),
                    ControlSpec("auto", "button", label="Auto",
                                callback=lambda _: self.apply_auto()),
                ]),
            ],
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
    