import PySide6 # noqa: F401 (need to import to select the qt backend)
import PySide6.QtWidgets

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason

class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):
    def __init__(self, *, size=None, title=None, max_fps=30, **kwargs):
        super().__init__(**kwargs)

        self._push_button = PySide6.QtWidgets.QPushButton("Push me")
        self._dropdown = PySide6.QtWidgets.QComboBox()
        self._dropdown.addItem("test")
        self._dropdown.addItem("test2")
        # replace wgpu's layout with our own
        our_layout = PySide6.QtWidgets.QVBoxLayout()
        our_layout.addWidget(self._push_button)
        our_layout.addWidget(self._dropdown)
        our_layout.setContentsMargins(30, 30, 30, 30)

        layout = self.layout()
        layout.addLayout(our_layout)

        self.show()

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
