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




