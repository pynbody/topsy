import time 
import ipywidgets as widgets
from playwright.sync_api import Page
import pytest 

from IPython.display import display


from typing import Callable 

import topsy, topsy.canvas.jupyter


def poll_until_true(assertion: Callable, timeout=2, iteration_delay=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if assertion():
            return True
        time.sleep(iteration_delay)
    return False

@pytest.fixture
def jupyter_vis(solara_test):
    vis = topsy.test(100, canvas_class = topsy.canvas.jupyter.VisualizerCanvas)
    display(vis)
    return vis


def test_colormap_name_select(jupyter_vis, page_session: Page):
    vis = jupyter_vis

    assert vis.colormap.get_parameter('colormap_name') == "twilight_shifted"

    sel = page_session.locator("select:has-text('twilight_shifted')")
    sel.wait_for()
    sel.select_option("twilight")

    assert poll_until_true(lambda: vis.colormap.get_parameter('colormap_name') == "twilight")

    assert vis.quantity_name == None

def test_quantity_name_select(jupyter_vis, page_session: Page):
    sel = page_session.locator(f"select:has-text('{topsy.config.PROJECTED_DENSITY_NAME}')")
    cb = page_session.locator("input[type='checkbox']")

    sel.wait_for()
    cb.wait_for()

    assert cb.is_checked()
    sel.select_option("test-quantity")


    assert poll_until_true(lambda: jupyter_vis.quantity_name == "test-quantity")

    # check that log quantity is no longer selected    
    assert poll_until_true(lambda: not cb.is_checked())

def test_alter_range(jupyter_vis, page_session: Page):
    vis = jupyter_vis

    min_slider = page_session.locator("div.noUi-handle-lower")
    max_slider = page_session.locator("div.noUi-handle-upper")

    min_slider.wait_for()
    max_slider.wait_for()

    vmin_orig = vis.colormap.get_parameter('vmin')
    vmax_orig = vis.colormap.get_parameter('vmax')

    # Use keyboard to move the slider instead of drag_to to avoid pointer event interception issues
    min_slider.focus()
    page_session.keyboard.press("ArrowLeft")
    page_session.keyboard.press("ArrowLeft")
    max_slider.focus()
    page_session.keyboard.press("ArrowRight")
    page_session.keyboard.press("ArrowRight")

    assert poll_until_true(lambda: vis.colormap.get_parameter('vmin') < vmin_orig)
    assert poll_until_true(lambda: vis.colormap.get_parameter('vmax') > vmax_orig)

def test_rgb_map(solara_test, page_session: Page):
    vis = topsy.test(100, canvas_class = topsy.canvas.jupyter.VisualizerCanvas, rgb=True)
    display(vis)

    # at the moment we just check this actually gives the alternative panel
    sel = page_session.locator("text=gamma")
    sel.wait_for()

    