"""topsy - An astrophysics simulation visualization package based on webgpu, using pynbody for reading data"""

from __future__ import annotations

__version__ = "0.5.0"

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
    argparser.add_argument('--hdr', help="[Experimental] Enable HDR rendering", action="store_true")
    argparser.add_argument('--rgb', help="[Experimental] Enable RGB->UVI rendering for stars", action="store_true")

    argparser.add_argument("--load-sphere", nargs='+', help="Load a sphere of particles with the given "
                                                          "radius and, optionally, centre in simulation units. "
                                                          "e.g. --load-sphere 5.0 to load a sphere of radius 5.0 about"
                                                          "the centre of the simulation, or 5.0 3.0 1.0 2.0 to load a "
                                                          "sphere of radius 5.0 about the point (3.0, 1.0, 2.0)."
                                                          "Supported only for swift simulations. Units are simulation units.",
                            metavar=("_"),
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

        this_args = argparser.parse_args(args[:split_index])

        if this_args.load_sphere is not None and len(this_args.load_sphere) != 1 and len(this_args.load_sphere) != 4:
                argparser.error("Invalid number of arguments for --load-sphere. Must be 1 or 4.")
        arg_batches.append(this_args)
        args = args[split_index+1:]


    return arg_batches

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
                match args.load_sphere:
                    case (r,):
                        loader_args = (args.filename, args.center, args.particle, pynbody.filt.Sphere(r))
                    case (r, x, y, z):
                        loader_args = (args.filename, args.center, args.particle,
                                       pynbody.filt.Sphere(r, (x, y, z)))
                    case _:
                        argparser.error("Invalid number of arguments for --load-sphere. Must be 1 or 4.")


            else:
                loader_args = (args.filename, args.center, args.particle)


        vis = visualizer.Visualizer(data_loader_class=loader_class,
                                    data_loader_args=loader_args,
                                    colormap_name=args.colormap,
                                    hdr=args.hdr,
                                    periodic_tiling=args.tile,
                                    render_resolution=args.resolution,
                                    rgb=args.rgb)

        vis.quantity_name = args.quantity
        vis.canvas.show()

    from rendercanvas import qt
    qt.loop.run()

def topsy(snapshot: pynbody.snapshot.SimSnap, quantity: str | None = None, **kwargs):
    from . import visualizer, loader
    vis = visualizer.Visualizer(data_loader_class=loader.PynbodyDataInMemory,
                                data_loader_args=(snapshot,),
                                **kwargs)
    vis.quantity_name = quantity
    return vis

def _test(nparticle=config.TEST_DATA_NUM_PARTICLES_DEFAULT, **kwargs):
    from . import visualizer, loader, drawreason
    vis = visualizer.Visualizer(data_loader_class=loader.TestDataLoader,
                                data_loader_args=(nparticle,),
                                data_loader_kwargs={'with_cells': kwargs.pop('with_cells', False)},
                                **kwargs)
    vis.draw(reason=drawreason.DrawReason.EXPORT)
    return vis
