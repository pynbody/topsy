import PySide6 # noqa: F401 (need to import to select the qt backend)

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason

class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):
    def request_draw(self, function=None):
        # As a side effect, wgpu gui layer stores our function call, to enable it to be
        # repainted later. But we want to distinguish such repaints and handle them
        # differently, so we need to replace the function with our own
        def function_wrapper():
            function()
            self._subwidget.draw_frame = lambda: self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)

        super().request_draw(function_wrapper)

    @classmethod
    def call_later(cls, delay, fn, *args):
        call_later(delay, fn, *args)
