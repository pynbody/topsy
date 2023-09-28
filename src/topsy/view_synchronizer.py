from __future__ import annotations

import weakref

from .drawreason import DrawReason

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer


class ViewSynchronizer:
    """A class to manage reflecting rotation/scale/offset changes from one view to another.

    Best used by creating two visualizers, visualizer1 and visualizer2, and then calling
    visualizer1.synchronize_with(visualizer2)

    The two visualizers will then be kept in sync
    """
    def __init__(self, synchronize=['rotation_matrix', 'scale', 'position_offset']):
        self._views : list[weakref.ReferenceType[Visualizer]] = []
        self._requires_update : list[weakref.ReferenceType[Visualizer]] = []
        self._synchronize = synchronize

    def perpetuate_update(self, source):
        """Called when a view has been updated and the update needs to be perpetuated to other views.

        Note that there is built-in protection against infinite loops. If view A calls this method
        which causes view B to update, if view B calls this method nothing will be issued back to view A."""
        sources_needing_update = [view_weakref() for view_weakref in self._requires_update]
        if source in sources_needing_update:
            # OK the update has happened! Great, but don't broadcast it again
            del self._requires_update[sources_needing_update.index(source)]
            return

        for view_weakref in self._views:
            view = view_weakref()
            if (view is not source and view is not None) and (view_weakref not in self._requires_update):
                self._requires_update.append(view_weakref)
                for var in self._synchronize:
                    setattr(view, var, getattr(source, var))

    def update_completed(self, view: Visualizer):
        """Called when a view knows it will not be attempting to perpetuate an update it received

        See note about infinite loops in perpetuate_update. This method is used when a view will
        not be acting on the update it received, so it must be removed from the exclusion list
        that perpetuate_update maintains."""
        sources_needing_update = [view_weakref() for view_weakref in self._requires_update]
        if view in sources_needing_update:
            del self._requires_update[sources_needing_update.index(view)]

    def add_view(self, view: Visualizer):
        self._views.append(weakref.ref(view))
        view._view_synchronizer = self

    def remove_view(self, view: Visualizer):
        self._views.remove(weakref.ref(view))
        del view._view_synchronizer

class SynchronizationMixin:
    """Mixin class for Visualizer to allow it to synchronize with other views"""
    def draw(self, reason, render_texture_view=None):
        super().draw(reason, render_texture_view)
        if hasattr(self, "_view_synchronizer") and reason != DrawReason.REFINE:
            self._view_synchronizer.perpetuate_update(self)

    def synchronize_with(self, other: Visualizer):
        """Start synchronizing this visualizer with another"""
        if hasattr(self, "_view_synchronizer") and hasattr(other, "_view_synchronizer"):
            raise RuntimeError("Both these visualizers are already synchronizing with others")

        if hasattr(self, "_view_synchronizer"):
            self._view_synchronizer.add_view(other)
        elif hasattr(other, "_view_synchronizer"):
            other._view_synchronizer.add_view(self)
        else:
            vs = ViewSynchronizer()
            vs.add_view(self)
            vs.add_view(other)

    def stop_synchronizing(self):
        """Stop synchronizing this visualizer with any other"""
        if hasattr(self, "_view_synchronizer"):
            self._view_synchronizer.remove_view(self)

    def is_synchronizing(self):
        return hasattr(self, "_view_synchronizer")
