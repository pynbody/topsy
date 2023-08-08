"""pynbody_vis - An astrophysics simulation visualization package based on OpenGL, using pynbody for reading data"""

__version__ = "0.1"

import sys
import argparse

from . import config, visualizer
def parse_args():
    """Create arguments and kwargs to pass to the visualizer, from sys.argv"""

    # create the argument parser, add arguments for filename, resolution, and colormap, and parse the arguments
    argparser = argparse.ArgumentParser(description="Visualize an astrophysics simulation")

    argparser.add_argument("filename", help="Specify path to a simulation file to be visualized")
    argparser.add_argument("--resolution", "-r", help="Specify the resolution of the visualization",
                           default=config.DEFAULT_RESOLUTION, type=int)
    argparser.add_argument("--colormap", "-c", help="Specify the matplotlib colormap to be used",
                           default=config.DEFAULT_COLORMAP, type=str)

    args = argparser.parse_args()
    sys.argv = sys.argv[:1] # to prevent confusing moderngl-window

    # the following is unbelievably ugly, but needed to workaround
    # inflexibility in moderngl-window's instantiation of the visualizer
    #
    # in the longer term, we should use a different windowing framework
    visualizer.Visualizer.args = vars(args)
    print(visualizer.Visualizer.args)

def main():
    parse_args()
    visualizer.Visualizer.run()
