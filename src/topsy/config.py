DEFAULT_RESOLUTION = 1024
DEFAULT_COLORMAP = 'twilight_shifted'

DEFAULT_SCALE = 200.0 # viewport width in kpc

TARGET_FPS = 30 # will use downsampling to achieve this
INITIAL_PARTICLES_TO_RENDER = 1e5 # number of particles to render at first
STATUS_LINE_UPDATE_INTERVAL = 0.2 # seconds
STATUS_LINE_UPDATE_INTERVAL_RAPID = 0.05 # when time-critical information is being displayed

GLIDE_TIME = 0.3 # seconds after double click to reach destination

COLORBAR_ASPECT_RATIO = 0.15
COLORMAP_NUM_SAMPLES = 1000

TEST_DATA_NUM_PARTICLES_DEFAULT = int(1e6)

MAX_PARTICLES_PER_BUFFER = 2**27
# arbitrary number, but small enough that GPU memory fragmentation not a huge issue hopefully, while
# large enough to not cause too much overhead


JUPYTER_UI_LAG = 0.05
# time over which to spread jupyter UI updates, notably for sliders where updating the range and value
# simultaneously seems to lead to problems

# special name for  projected density in UI
PROJECTED_DENSITY_NAME = "Projected density"