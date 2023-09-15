import enum

class DrawReason(enum.Enum):
    """Enum to specify the reason for a draw (which may affect detailed behaviour)"""
    INITIAL_UPDATE = 1      # render from scratch
    CHANGE = 2              # a change has occurred, possibly from the UI
    REFINE = 3              # render the SPH at full resolution, but otherwise no change has occurred
    PRESENTATION_CHANGE = 4 # i.e. don't rerender SPH
