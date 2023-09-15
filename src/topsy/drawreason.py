import enum

class DrawReason(enum.Enum):
    """Enum to specify the reason for a draw (which may affect detailed behaviour)"""
    INITIAL_UPDATE = 1
    CHANGE = 2
    REFINE = 3