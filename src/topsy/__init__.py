"""topsy - An astrophysics simulation visualization package based on OpenGL, using pynbody for reading data"""

__version__ = "0.2.0"

import logging
import sys
import argparse
import matplotlib

from . import config, visualizer, loader

matplotlib.use("Agg")


def parse_args():
    """Create arguments and kwargs to pass to the visualizer, from sys.argv"""

    # create the argument parser, add arguments for filename, resolution, and colormap, and parse the arguments
    argparser = argparse.ArgumentParser(description="Visualize an astrophysics simulation")

    argparser.add_argument("filename", help="Specify path to a simulation file to be visualized")
    argparser.add_argument("--resolution", "-r", help="Specify the resolution of the visualization",
                           default=config.DEFAULT_RESOLUTION, type=int)
    argparser.add_argument("--colormap", "-m", help="Specify the matplotlib colormap to be used",
                           default=config.DEFAULT_COLORMAP, type=str)
    argparser.add_argument("--particle", "-p", help="Specify the particle type to visualise",
                            default="dm", type=str)
    argparser.add_argument("--center", "-c", help="Specify the centering method: 'halo-<N>', 'all', 'zoom' or 'none'",
                           default="halo-1", type=str)

    return argparser.parse_args()

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)

def main():
    setup_logging()
    args = parse_args()

    vis = visualizer.Visualizer(data_loader_class=loader.PynbodyDataLoader,
                                     data_loader_args=(args.filename, args.center, args.particle))
    vis.run()
