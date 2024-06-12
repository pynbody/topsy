"""topsy - An astrophysics simulation visualization package based on webgpu, using pynbody for reading data"""

from __future__ import annotations

__version__ = "0.3.4"

import argparse
import logging
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pynbody


from . import config


def parse_args(args=None):
    """Create arguments and kwargs to pass to the visualizer, from sys.argv"""

    # create the argument parser, add arguments for filename, resolution, and colormap, and parse the arguments
    argparser = argparse.ArgumentParser(description="Visualize an astrophysics simulation. Multiple windows can be opened by separating groups of arguments with *.")

    argparser.add_argument("filename", help="Specify path to a simulation file to be visualized")
    argparser.add_argument("--resolution", "-r", help="Specify the resolution of the visualization",
                           default=config.DEFAULT_RESOLUTION, type=int)
    argparser.add_argument("--colormap", "-m", help="Specify the matplotlib colormap to be used",
                           default=config.DEFAULT_COLORMAP, type=str)
    argparser.add_argument("--particle", "-p", help="Specify the particle type to visualise",
                            default="dm", type=str)
    argparser.add_argument("--center", "-c", help="Specify the centering method: 'halo-<N>', 'all', 'zoom' or 'none'",
                           default="none", type=str)
    argparser.add_argument("--quantity", "-q", help="Specify a quantity to render instead of density",
                           default=None, type=str)
    argparser.add_argument("--tile", "-t", help="Wrap and tile the simulation box using its periodicity",
                           default=False, action="store_true")

    argparser.add_argument("--load-sphere", nargs=4, help="Load a sphere of particles with the given "
                                                          "radius and centre in simulation units, "
                                                          "e.g. --load-sphere 0.2 0.3 0.4 0.5 to load a sphere of "
                                                          "particles centre (0.3, 0.4, 0.5), radius 0.2. " 
                                                          "Supported only for swift simulations",
                            default=None, type=float)

    if args is None:
        args = sys.argv[1:]
    arg_batches = []
    # split args into batches separated by '+'
    while len(args) > 0:
        try:
            split_index = args.index("+")
        except ValueError:
            split_index = len(args)
        arg_batches.append(args[:split_index])
        args = args[split_index+1:]

    return [argparser.parse_args(batch) for batch in arg_batches]

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
    all_args = parse_args()

    from . import visualizer, loader

    for args in all_args:
        if "test://" in args.filename:
            loader_class = loader.TestDataLoader
            try:
                n_part = int(float(args.filename[7:])) # going through float allows scientific notation
            except ValueError:
                n_part = config.TEST_DATA_NUM_PARTICLES_DEFAULT
            logger.info(f"Using test data with {n_part} particles")
            loader_args = (n_part,)
        else:
            import pynbody
            loader_class = loader.PynbodyDataLoader
            if args.load_sphere is not None:
                loader_args = (args.filename, args.center, args.particle,
                               pynbody.filt.Sphere(args.load_sphere[0], args.load_sphere[1:]))
            else:
                loader_args = (args.filename, args.center, args.particle)


        vis = visualizer.Visualizer(data_loader_class=loader_class,
                                    data_loader_args=loader_args,
                                    colormap_name=args.colormap,
                                    periodic_tiling=args.tile,
                                    render_resolution=args.resolution)

        vis.quantity_name = args.quantity
        vis.canvas.show()

    from wgpu.gui import qt
    qt.run()

def topsy(snapshot: pynbody.snapshot.SimSnap, quantity: str | None = None, **kwargs):
    from . import visualizer, loader
    vis = visualizer.Visualizer(data_loader_class=loader.PynbodyDataInMemory,
                                data_loader_args=(snapshot,),
                                **kwargs)
    vis.quantity_name = quantity
    return vis

def _test(nparticle=config.TEST_DATA_NUM_PARTICLES_DEFAULT, **kwargs):
    from . import visualizer, loader
    vis = visualizer.Visualizer(data_loader_class=loader.TestDataLoader,
                                data_loader_args=(nparticle,),
                                **kwargs)
    return vis