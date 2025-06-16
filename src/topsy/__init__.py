"""topsy - An astrophysics simulation visualization package based on webgpu, using pynbody for reading data"""

from __future__ import annotations

__version__ = "0.7.0"

import argparse
import logging
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pynbody
    from .visualizer import Visualizer


from . import config

logger = None

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
    argparser.add_argument("--bivariate", "-b", help="[Experimental] Enable bivariate rendering", action="store_true")
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
    if logger is not None:
        return
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(ch)

def main():
    all_args = parse_args()

    for args in all_args:
        vis = load(args.filename, center=args.center, resolution=args.resolution,
                    particle=args.particle, tile=args.tile, rgb=args.rgb,
                    sphere_radius=args.load_sphere[0] if args.load_sphere is not None else None,
                    sphere_center=tuple(args.load_sphere[1:]) if args.load_sphere is not None and len(args.load_sphere) == 4 else None,
                    hdr=args.hdr, bivariate=args.bivariate)
        vis.quantity_name = args.quantity
        vis.canvas.show()

    from rendercanvas import qt # has to be imported here so that underlying qt toolkit has been autoselected
    qt.loop.run()

def topsy(snapshot: pynbody.snapshot.SimSnap, quantity: str | None = None, **kwargs):
    from . import visualizer, loader
    vis = visualizer.Visualizer(data_loader_class=loader.PynbodyDataInMemory,
                                data_loader_args=(snapshot,),
                                **kwargs)
    vis.quantity_name = quantity
    return vis

def load(filename: str, center: str = "none", particle: str = "gas", rgb: bool = False,
         resolution: int = config.DEFAULT_RESOLUTION, tile: bool = False,
         sphere_radius: float | None = None, sphere_center: tuple[float, float, float] | None = None,
         hdr: bool = False, bivariate: bool = False) -> Visualizer:
    """
    Load a simulation file (currently using pynbody) and return a visualizer object.

    Parameters
    ----------

    filename : str
        Path to the simulation file. You can also specify test://<N> to generate a test dataset with N particles.

    center : str
        Centering method. Can be 'halo-<N>', 'all', 'zoom' or 'none'.

    particle : str
        Particle type to visualize. Default is 'gas'; other options include 'dm' and 'star'.

    resolution : int
        Resolution of the visualization in pixels.

    sphere_radius : float | None
        If specified, load a sphere of particles with the given radius. Units are simulation units.

    sphere_center : tuple[float, float, float] | None
        If specified, load a sphere of particles with the given center. Units are simulation units.
        Must be a tuple of three floats (x, y, z).

    rgb : bool
        If True, enable RGB->UVI rendering for stars. Default is False.

    bivariate : bool
        If True, enable bivariate rendering. Default is False.

    hdr : bool
        If True, try enabling HDR rendering (only valid when rgb=True). Default is False.

    tile : bool
        If True, wrap and tile the simulation box using its periodicity. Default is False.


    Returns
    -------
    visualizer.Visualizer
        A visualizer object that can be used to render the simulation data.

    """
    from . import visualizer, loader
    setup_logging()
    
    if "test://" in filename:
        loader_class = loader.TestDataLoader
        try:
            n_part = int(float(filename[7:]))  # going through float allows scientific notation
        except ValueError:
            n_part = config.TEST_DATA_NUM_PARTICLES_DEFAULT
        logger.info(f"Using test data with {n_part} particles")
        loader_args = (n_part,)
    else:
        import pynbody
        loader_class = loader.PynbodyDataLoader
        if sphere_radius is not None:
            if sphere_center is not None:
                loader_args = (filename, center, particle, pynbody.filt.Sphere(sphere_radius, sphere_center))
            else:
                loader_args = (filename, center, particle, pynbody.filt.Sphere(sphere_radius))
        else:
            loader_args = (filename, center, particle)

    vis = visualizer.Visualizer(data_loader_class=loader_class,
                                data_loader_args=loader_args,
                                hdr=hdr,
                                periodic_tiling=tile,
                                render_resolution=resolution,
                                rgb=rgb, bivariate=bivariate)

    return vis

def test(nparticle=config.TEST_DATA_NUM_PARTICLES_DEFAULT, **kwargs) -> Visualizer:
    from . import visualizer, loader
    vis = visualizer.Visualizer(data_loader_class=loader.TestDataLoader,
                                data_loader_args=(nparticle,),
                                data_loader_kwargs={'with_cells': kwargs.pop('with_cells', False),
                                                    'periodic': kwargs.get('periodic_tiling', False)},
                                **kwargs)
    return vis




_force_is_jupyter = False

def is_jupyter():
    """Determine whether the user is executing in a Jupyter Notebook / Lab.

    This has been pasted from an old version of wgpu.gui.auto.is_jupyter; the function was removed"""
    global _force_is_jupyter
    if _force_is_jupyter:
        return True
    from IPython import get_ipython
    try:
        ip = get_ipython()
        if ip is None:
            return False
        if ip.has_trait("kernel"):
            return True
        else:
            return False
    except NameError:
        return False
    
def force_jupyter():
    """Force the return from is_jupyter() to be True; used in testing"""
    global _force_is_jupyter
    _force_is_jupyter = True
