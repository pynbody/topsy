topsy
===========

This package visualises simulations, and is an add-on to the pynbody analysis package.

At the moment, it's a bit of a toy project, but it already works quite well with zoom 
(or low resolution) simulations. The future development path will depend on the level
of interest from the community.

Installing
----------

You will need python 3.8 or later. You can then install `topsy` using `pip` 
as usual:

```
pip install topsy
```

This will install topsy and its dependencies (including `pynbody` itself) into
your current python environment.

As usual, you can also install direct from github, e.g.

```
pip install git+https://github.com/pynbody/topsy
```

Or clone the repository and install for development using

```
pip install -e .
```

from inside the cloned repository.

If you want to play with `topsy` without disturbing your current installation,
I recommend using `venv`:

```
# create a toy environment
python -m venv visualiser-env

# activate the new environment
source visualiser-env/bin/activate 

# install
pip install topsy

... other commands ...

# get your old environment back:
deactivate 
```

For more information about venv, see its 
[tutorial page](https://docs.python.org/3/library/venv.html).


Trying it out
-------------

The package provides one simple command called `topsy`, to be 
called straight from your shell. Pass `topsy` the path to the
simulation that you wish to visualise. 

You can (and probably should) also
tell it what to center on using the `-c` flag, to which valid arguments are:

* `-c none` (just loads the file without changing the centering) 
* `-c halo-1` (uses the shrink sphere center of halo 1; or you can change 1 to any other number)
* `-c zoom` (uses the shrink sphere center on the highest resolution particles, without loading a halo catalogue)
* `-c all` (uses the shrink sphere center on all particles in the file)

By default, it will show you dark matter particles. To change this pass `-p gas` to show gas particles or `-p star` for stars.

By default, it uses matplotlib's `twilight_shifted` colormap. To change this pass, for example, `-m viridis`, or the name
of any other [matplotlib colormap](https://matplotlib.org/stable/tutorials/colors/colormaps.html#sequential).

Controls in the main window
---------------------------

If everything works, a window will pop up with a beautiful rendering of your simulation. Make sure the window
is in focus (for some reason on MacOS I sometimes have to switch to another application then back to 
python to get this to work). Then you can use the following controls:

* To spin around the centre, drag the mouse. 
* To zoom in and out, use the mouse scroll wheel. 
* To rescale the colours to an appropriate range for the current view, press `r`(ange)
* To return the view to the original orientation and zoom, press `h`(ome)
* To save a snapshot of the current image as a pdf press `s`(ave)