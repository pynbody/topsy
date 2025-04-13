import pytest

from topsy import progressive_render, config
from topsy.drawreason import DrawReason


def test_initial_recommendations():
    # Test the initial recommendation for a small number of particles
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER//2)
    render_progression.start_frame(DrawReason.INITIAL_UPDATE)
    assert render_progression.get_block(0.0) == (0, config.INITIAL_PARTICLES_TO_RENDER//2)

    # Test the initial recommendation for a large number of particles
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER*2)
    render_progression.start_frame(DrawReason.INITIAL_UPDATE)
    assert render_progression.get_block(0.0) == (0, config.INITIAL_PARTICLES_TO_RENDER)

def test_export_recommendations():
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER * 2)
    render_progression.start_frame(DrawReason.EXPORT)
    assert render_progression.get_block(0.0) == (0, config.INITIAL_PARTICLES_TO_RENDER*2)
    render_progression.end_block(0.1)
    assert render_progression.get_block(1.0) is None

def test_progression():
    # Test the progression of recommendations
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate rendering a block of particles
    start_index, num_to_render = render_progression.get_block(0.0)
    assert start_index == 0
    assert num_to_render == 100

    # Simulate ending the block and reporting time taken
    render_progression.end_block(0.0005)

    # Check the next recommendation
    start_index, num_to_render = render_progression.get_block(0.0005)
    assert start_index == 100
    assert num_to_render == 900

    render_progression.end_block(0.0006)

    # Check the end of the frame
    assert render_progression.get_block(0.0006) is None

def test_timeout_and_progression():
    # Test the timeout and progression of recommendations
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate a long time elapsed
    block = render_progression.get_block(0.0)
    assert block is not None

    render_progression.end_block(1.0) # far too long!
    # Check that the next recommendation is None
    block = render_progression.get_block(1.0)
    assert block is None

    sf = render_progression.end_frame_get_scalefactor()
    assert sf == 10.0

    assert render_progression.needs_refine()

    render_progression.start_frame(DrawReason.REFINE)
    start, num = render_progression.get_block(0.0)
    assert start == 100
    assert num == int(100/config.TARGET_FPS) # took one second to render 100 particles, so recommendation should be this


def test_always_one_block():
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    block = render_progression.get_block(1.0) # simulate a long time elapsed, but should still render at least one block
    assert block is not None

def test_no_render_on_presentation_change():
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate rendering the first frame
    t = 0.0
    while (block := render_progression.get_block(t)) is not None:
        t+=1e-5
        render_progression.end_block(t)

    render_progression.end_frame_get_scalefactor()
    assert not render_progression.needs_refine()

    render_progression.start_frame(DrawReason.PRESENTATION_CHANGE)

    block = render_progression.get_block(0.0)
    assert block is None

    render_progression.end_frame_get_scalefactor()
    assert not render_progression.needs_refine()

def test_no_frame_exception():
    render_progression = progressive_render.RenderProgression(1000, 100)
    with pytest.raises(RuntimeError):
        render_progression.get_block(0.0)

def test_export():
    """Test that export always recommends the full resolution"""

    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.EXPORT)
    block = render_progression.get_block(0.0)
    assert block == (0, 1000)

def test_always_one_particle():
    """Test that always at least one particle is recommended (even if it takes a long time)"""

    render_progression = progressive_render.RenderProgression(1000, 3)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate a long time elapsed
    block = render_progression.get_block(0.0)
    assert block is not None

    render_progression.end_block(1.0) # far too long!

    assert render_progression.get_block(1.0) is None # end of this frame
    render_progression.end_frame_get_scalefactor()
    assert render_progression.needs_refine()

    # Check that the refinement consists of at least one particle, even though technically
    # we would have rounded this down to zero particles
    render_progression.start_frame(DrawReason.REFINE)
    block = render_progression.get_block(1.0)
    assert block == (3,1)
