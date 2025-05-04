"""Tools for measuring performance"""

try:
    from os_signpost import Signposter
    signposter = Signposter("com.pynbody.topsy", Signposter.Category.PointsOfInterest)
except ImportError:
    # Most of the time we won't have this module, so make a dummy
    class DummySignposter:
        def __init__(self):
            pass

        def begin_interval(self, *args, **kwargs):
            pass

        def emit_event(self, *args, **kwargs):
            pass

        def use_interval(*args, **kwds):
            pass

    signposter = DummySignposter()