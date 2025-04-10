from __future__ import annotations

from rendercanvas.offscreen import OffscreenRenderCanvas, loop

from . import VisualizerCanvasBase


class VisualizerCanvas(VisualizerCanvasBase, OffscreenRenderCanvas):

    @classmethod
    def call_later(cls, delay, fn, *args):
        loop.call_later(delay, fn, *args)

