from __future__ import annotations

import numpy as np
import wgpu

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer_wgpu import Visualizer

class MultiresolutionSPH:
    """A drop-in replacement for the SPH class, which renders to multiple resolutions and then combines them."""

    def __init__(self, visualizer: Visualizer):
        pass
