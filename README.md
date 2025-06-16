https://github.com/user-attachments/assets/b185c3b8-8658-4f7d-96de-e976959e7ad6


topsy
=====

[![Build Status](https://github.com/pynbody/topsy/actions/workflows/build-test.yaml/badge.svg)](https://github.com/pynbody/topsy/actions)

This package visualises simulations, and is an add-on to the [pynbody](https://github.com/pynbody/pynbody) analysis package.
Its name nods to the [TIPSY](https://github.com/N-BodyShop/tipsy) project.
It is built using [wgpu](https://wgpu.rs), which is a future-facing GPU standard (with thanks to the [python wgpu bindings](https://wgpu-py.readthedocs.io/en/stable/guide.html)).

At the moment, `topsy` is experimental, but has proven to work well in a variety of environments.  It is mainly
developed and optimized on Apple M-series chips, but has also been shown to work on NVidia GPUs. 

The future development path will depend on the level of interest from the community.

Installing
----------

You will need python 3.11 or later, running in a UNIX variant (basically MacOS, Linux or if you're on Windows you need [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)). You can then install `topsy` using `pip` 
as usual:

```
pip install topsy
```

This will install topsy and its dependencies (including `pynbody` itself) into
your current python environment. (If it fails, check that you have python 3.11
or later, and `pip` is itself up-to-date using `pip install -U pip`.)

### Alternative 1: install into isolated environment using pipx

You can also install `topsy` into its own isolated environment using [pipx](https://pypi.org/project/pipx/):

```
pipx install topsy
```

The command line tool will now be available, but you won't have access to the `topsy` package from your existing python environment. This can be useful if you don't want to risk disturbing anything.

### Alternative 2: install into new environment using venv and pip 

If you want to play with `topsy` without disturbing your existing installation, but also want to be able to use `topsy` from python scripts or jupyter etc, I recommend using `venv`:

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

### Alternative 3: install unreleased versions or contribute to development

As usual, you can also install direct from github, e.g.

```
pip install git+https://github.com/pynbody/topsy
```

Or clone the repository and install for development using

```
pip install -e .
```

from inside the cloned repository.




Trying it out
-------------

### Very quick start

Once `topsy` is installed, if you just want to try it out and you don't have a 
suitable simulation snapshot to hand, you can download some
from the [tangos tutorial datasets (5.1GB)](https://zenodo.org/records/5959983/files/tutorial_changa.tar.gz?download=1).
You need to untar them (`tar -xzf tutorial_changa.tar.gz` from your command line), then
you can type `topsy pioneer50h128.1536gst1.bwK1.000832` to visualise that file's
dark matter content.

### More detailed description

If using from the command line, pass `topsy` the path to the simulation that you wish to visualise. 

You can (and probably should) also
tell it what to center on using the `-c` flag, to which valid arguments are:

* `-c none` (just loads the file without changing the centering) 
* `-c halo-1` (uses the shrink sphere center of halo 1; or you can change 1 to any other number)
* `-c zoom` (uses the shrink sphere center on the highest resolution particles, without loading a halo catalogue)
* `-c all` (uses the shrink sphere center on all particles in the file)

By default, it will show you dark matter particles. To change this pass `-p gas` to show gas particles or `-p star` for 
stars. Note that the particle type _cannot_ be changed once the window is open (although you can open a separate window for each particle type; see below).

If your particles have other quantities defined on them (such as `temp` for gas particles), you can view the 
density-weighted average quantity by passing `-q temp`. The quantity to visualise can also be changed by selecting it via the main window controls
(see below).  

To open more than one visualisation window on different files or with different parameters, you can
pass multiple groups of parameters separated by `+`, for example to see separate views of the gas and
dark matter you could launch `topsy` with:

```
topsy -c halo-1 -p gas my_simulation + -c halo-1 -p dm my_simulation
```

You can choose to link the rotation/zoom of multiple views using the toolbar (see below).

Using SSPs
----------

If you have stars in your simulation, you can try rendering using pynbody's SSP tables, using the command-line
flag `--rgb`, e.g.

```
topsy -c halo-1 -p s --rgb my_simulation 
```

Even better, if you have an HDR display (e.g. recent Macbook Pros), you can use the `--hdr` flag to render in HDR mode.  
Note in HDR mode that the magnitude range specified applies to the SDR range, i.e. HDR brightnesses extend beyond the specified maximum surface brightness limit. The exact brightest magntiude that can be displayed will depend on your display hardware.


Controls in the main window
---------------------------

The view in the `topsy` window can be manipulated as follows:

* To spin around the centre, **drag** the mouse.
* To zoom in and out, use the mouse **scroll** wheel.
* To move the centre, **double click** on a target (topsy will determine its depth), or **shift-drag** to move in x-y plane.
* To rescale the colours to an appropriate range for the current view, press `r`(ange)
* To return the view to the original orientation and zoom, press `h`(ome)

There is also a toolbar at the bottom of the window with some buttons:

* <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/record.png?raw=true" style="width: 1em;">
  - start recording actions (rotations, scalings, movements and more). Press again to stop. 
* <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/movie.png?raw=true" style="width: 1em;">
  - render the recorded actions into an mp4 file. You will be prompted about various options and a filename.
* <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/load_script.png?raw=true" style="width: 1em;">
  <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/save_script.png?raw=true" style="width: 1em;">
  - load and save the recorded actions to a file for later use.
* <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/camera.png?raw=true" style="width: 1em;">
  - save a snapshot of the current view to an image file.
* <img src="https://github.com/pynbody/topsy/blob/c69e08e6e8d29cd93b6e8224796de4eec6d0c667/src/topsy/canvas/icons/linked.png?raw=true" style="width: 1em;">
  - link this window to other topsy windows, so that rotating, scaling or moving one does the same to the other
* <img src="https://github.com/pynbody/topsy/blob/b516b3e15aeefcc78ecb4d8b52009f6243da7020/src/topsy/canvas/qt/icons/rgb.png?raw=true" style="width: 1em;"> - open colormap control; this lets you select the min/max values, the quantity to visualise, and the matplotlib colormap. (When in RGB / SSP mode, you just get to set the surface brightness range and a gamma value.) 

Using from jupyter
------------------

Thanks to [jupyter-rfb](https://jupyter-rfb.readthedocs.io/en/stable/), it is possible to use `topsy` within a jupyter notebook. 

To open a topsy view within your jupyter notebook, try

```python
import topsy 
topsy.load("/path/to/simulation", particle="gas")
```
Note that you can interact with this widget in exactly the same way as the native window produced by `topsy`. Most of
the same options you can pass on the command line are also available via this `load` function (type 
`help(topsy.load)` for details).
