from IPython.display import display
import ipywidgets as widgets
from typing import Callable, Any

from rendercanvas.jupyter import RenderCanvas, loop
from . import VisualizerCanvasBase
from ..config import JUPYTER_UI_LAG
from ..colormap.ui import ControlSpec

class VisualizerCanvas(VisualizerCanvasBase, RenderCanvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_events = True

    def request_draw(self, function=None):

        # As a side effect, wgpu gui layer stores our function call, to enable it to be
        # repainted later. But we want to distinguish such repaints and handle them
        # differently, so we need to replace the function with our own

        #def function_wrapper():
        #    function()
        #    self._subwidget.draw_frame = lambda: self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)
        # TODO: above needs to be hacked for jupyter

        super().request_draw(function)



    @classmethod
    def call_later(cls, delay, fn, *args):
        loop.call_later(delay, fn, *args)

    def ipython_display_with_widgets(self):
        """Display the canvas in a Jupyter notebook with widgets."""
        color_controls = self.build_color_controls()

        # stack canvas, dropdown and range slider
        display(widgets.VBox([self, color_controls]))

    


    def build_color_controls(self) -> widgets.Widget:
        """
        Return a nested ipywidget (HBox/VBox) tree driven by the generic ColorMapController.get_layout() spec.
        """
        self._controller = self._visualizer.colormap.make_ui_controller(self._visualizer, self._refresh_ui)
        if self._controller:
            self._controls = self.convert_layout_to_widget(self._controller.get_layout())
        else:
            self._controls = widgets.HTML("<b>No colormap controls available</b>")
        return self._controls

    def _callback(self, callback: Callable[[Any], None], value: Any):
        if not self._allow_events:
            return
        callback(value)
       
    
    def _refresh_ui(self):
        """Walk the layout and update all values, including slider ranges."""
        if not hasattr(self, "_controller"):
            return
        root_spec = self._controller.get_layout()
        self._allow_events = False
        try:
            self.update_widget(root_spec, self._controls)
        finally:
             # re-enable events after a delay, to allow the UI to settle (eugh! surely must be a better way?)
            self.call_later(JUPYTER_UI_LAG, lambda: setattr(self, "_allow_events", True))

    def convert_layout_to_widget(self, spec) -> widgets.Widget:
        children = []
        for child in spec.children:
            if isinstance(child, ControlSpec):
                children.append(self.make_widget(child))
            else:
                children.append(self.convert_layout_to_widget(child))
        if spec.type == "hbox":
            return widgets.HBox(children)
        else:
            return widgets.VBox(children)

    def make_widget(self, spec):
        if spec.type == "combo" or spec.type == 'combo-edit':  # can't get a good implementation of combo editing in ipython currently
            w = widgets.Dropdown(
                options=spec.options or [],
                value=spec.value,
                description=spec.label or "",
                layout=widgets.Layout(width="200px")
            )
            w.observe(lambda change, cb=spec.callback: self._callback(cb, change["new"]), names="value")

        elif spec.type == "checkbox":
            w = widgets.Checkbox(
                value=bool(spec.value),
                description=spec.label or ""
            )
            w.observe(lambda change, cb=spec.callback: self._callback(cb, change["new"]), names="value")

        elif spec.type == "range_slider":
            lo, hi = spec.range or (0.0, 1.0)
            w = widgets.FloatRangeSlider(
                value=tuple(spec.value),
                min=lo, max=hi,
                step=None,
                description=spec.label or "",
                layout=widgets.Layout(width="400px")
            )
            w.observe(lambda change, cb=spec.callback: self._callback(cb,change["new"]), names="value")

        elif spec.type == "slider":
            lo, hi = spec.range or (0.0, 1.0)
            w = widgets.FloatSlider(
                value=spec.value,
                min=lo, max=hi,
                step=None,
                description=spec.label or "",
                layout=widgets.Layout(width="400px")
            )
            w.observe(lambda change, cb=spec.callback: self._callback(cb, change["new"]), names="value")

        elif spec.type == "button":
            w = widgets.Button(description=spec.label or "")
            w.on_click(lambda btn, cb=spec.callback: self._callback(cb, None))

        else:
            w = widgets.HTML(f"<b>Unknown control {spec.name}</b>")

        return w


    def update_widget(self, spec, widget):
        if isinstance(spec, ControlSpec):
            if spec.type in {"combo", "combo-edit"}:
                widget.value = spec.value
            elif spec.type == "checkbox":
                widget.value = bool(spec.value)
            elif spec.type == "range_slider":
                lo, hi = spec.range or (0.0, 1.0)
                self.safe_update_slider_range(widget, lo, hi)
                wlo, whi = spec.value

                # seemingly need to set this after the range update has gone through, otherwise get
                # nonsense results in some cases
                self.call_later(JUPYTER_UI_LAG/2, lambda: setattr(widget, "value", (wlo, whi)))

            elif spec.type == "slider":
                lo, hi = spec.range or (0.0, 1.0)
                self.safe_update_slider_range(widget, lo, hi)

                # seemingly need to set this after the range update has gone through, otherwise get
                # nonsense results in some cases
                self.call_later(JUPYTER_UI_LAG/2, lambda: setattr(widget, "value", spec.value))
        else:
            for child_spec, child_widget in zip(spec.children, widget.children):
                self.update_widget(child_spec, child_widget)

    @classmethod
    def safe_update_slider_range(cls, slider, min_, max_):
        # sliders in ipywidgets seem to offer no option to update the range atomically. If one naively sets
        # min and max, the intermediate state can be invalid and raise an exception. Therefore, one needs
        # to first set a bounding range, then narrow back down to min, max.
        if slider.min == min_ and slider.max == max_:
            return
        bounding_min = min(min_, slider.min)
        bounding_max = max(max_, slider.max)
        slider.min = bounding_min
        slider.max = bounding_max
        slider.min = min_
        slider.max = max_

