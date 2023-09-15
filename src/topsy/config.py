DEFAULT_RESOLUTION = 1024
DEFAULT_COLORMAP = 'twilight_shifted'

DEFAULT_SCALE = 200.0 # viewport width in kpc

TARGET_FPS = 30 # will use downsampling to achieve this
FULL_RESOLUTION_RENDER_AFTER = 0.3 # inactivity seconds to wait before rendering without downs
STATUS_LINE_UPDATE_INTERVAL = 0.2 # seconds
STATUS_LINE_UPDATE_INTERVAL_RAPID = 0.05 # when time-critical information is being displayed

COLORBAR_ASPECT_RATIO = 0.15
COLORMAP_NUM_SAMPLES = 1000

TEST_DATA_NUM_PARTICLES_DEFAULT = int(1e6)