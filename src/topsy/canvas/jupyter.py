from rendercanvas.jupyter import RenderCanvas, loop
from . import VisualizerCanvasBase

class VisualizerCanvas(VisualizerCanvasBase, RenderCanvas):
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
        from IPython.display import display
        import ipywidgets as widgets

        # an un‑wired dropdown
        dropdown = widgets.Dropdown(
            options=['Option A', 'Option B', 'Option C'],
            value='Option A',
            description='Choice:',
        )

        # a two‑handle range slider
        range_slider = widgets.FloatRangeSlider(
            value=(0.2, 0.8),
            min=0.0,
            max=1.0,
            step=0.01,
            description='Range:',
        )

        # stack canvas, dropdown and range slider
        display(widgets.VBox([self, dropdown, range_slider]))
    

