"""topsy - An astrophysics simulation visualization package based on webgpu, using pynbody for reading data"""

__version__ = "0.2.1"

import logging
import sys
import argparse
import matplotlib
import pynbody


import PySide6
import wgpu.gui.qt
import wgpu, wgpu.gui.auto

import wgpu.gui.jupyter

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
    argparser.add_argument("--quantity", "-q", help="Specify a quantity to render instead of density",
                           default=None, type=str)
    argparser.add_argument("--tile", "-t", help="Wrap and tile the simulation box using its periodicity",
                           default=False, action="store_true")

    return argparser.parse_args()

def setup_logging():
    global logger
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

    if "test://" in args.filename:
        loader_class = loader.TestDataLoader
        try:
            n_part = int(float(args.filename[7:])) # going through float allows scientific notation
        except ValueError:
            n_part = config.TEST_DATA_NUM_PARTICLES_DEFAULT
        logger.info(f"Using test data with {n_part} particles")
        loader_args = (n_part,)
    else:
        loader_class = loader.PynbodyDataLoader
        loader_args = (args.filename, args.center, args.particle)


    vis = visualizer.Visualizer(data_loader_class=loader_class,
                                data_loader_args=loader_args,
                                colormap_name=args.colormap,
                                periodic_tiling=args.tile,
                                render_resolution=args.resolution)

    vis.quantity_name = args.quantity
    vis.show()

def topsy(snapshot: pynbody.snapshot.SimSnap, quantity: str | None = None, **kwargs):
    vis = visualizer.Visualizer(data_loader_class=loader.PynbodyDataInMemory,
                                data_loader_args=(snapshot,),
                                **kwargs)
    vis.quantity_name = quantity
    return vis

def _test(nparticle=config.TEST_DATA_NUM_PARTICLES_DEFAULT):
    vis = visualizer.Visualizer(data_loader_class=loader.TestDataLoader,
                                data_loader_args=(nparticle,))
    return vis