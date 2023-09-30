from __future__ import annotations

import numpy as np
from wgpu.gui.offscreen import WgpuManualOffscreenCanvas, call_later

from . import VisualizerCanvasBase

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..visualizer import Visualizer


class VisualizerCanvas(VisualizerCanvasBase, WgpuManualOffscreenCanvas):

    @classmethod
    def call_later(cls, delay, fn, *args):
        call_later(delay, fn, *args)

