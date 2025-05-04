import numpy as np
import pytest

from topsy import progressive_render, config
from topsy.drawreason import DrawReason

def _get_single_block(blocks):
    """Helper function to convert from list of blocks to single block"""
    assert len(blocks)==2
    assert len(blocks[0])==1
    assert len(blocks[1])==1
    return blocks[0][0], blocks[1][0]

def test_initial_recommendations():
    # Test the initial recommendation for a small number of particles
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER//2)
    render_progression.start_frame(DrawReason.INITIAL_UPDATE)
    assert _get_single_block(render_progression.get_block(0.0)) == (0, config.INITIAL_PARTICLES_TO_RENDER//2)

    # Test the initial recommendation for a large number of particles
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER*2)
    render_progression.start_frame(DrawReason.INITIAL_UPDATE)
    assert _get_single_block(render_progression.get_block(0.0)) == (0, config.INITIAL_PARTICLES_TO_RENDER)

def test_export_recommendations():
    render_progression = progressive_render.RenderProgression(config.INITIAL_PARTICLES_TO_RENDER * 2)
    render_progression.start_frame(DrawReason.EXPORT)
    assert _get_single_block(render_progression.get_block(0.0)) == (0, config.INITIAL_PARTICLES_TO_RENDER*2)
    render_progression.end_block(0.1)
    assert render_progression.get_block(1.0) is None

def test_progression():
    # Test the progression of recommendations
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate rendering a block of particles
    start_index, num_to_render = _get_single_block(render_progression.get_block(0.0))
    assert start_index == 0
    assert num_to_render == 100

    # Simulate ending the block and reporting time taken
    render_progression.end_block(0.5/config.TARGET_FPS)

    # Check the next recommendation. We're half way through.
    start_index, num_to_render = _get_single_block(render_progression.get_block(0.5/config.TARGET_FPS))
    assert start_index == 100
    assert num_to_render == 50 # hasn't updated the expected rendering number, just looks at time remaining

    render_progression.end_block(1./config.TARGET_FPS)

    # Check the end of the frame
    assert render_progression.get_block(1./config.TARGET_FPS) is None

    # We've rendered 150 particles.
    assert render_progression.end_frame_get_scalefactor() == 1000./150


def test_timeout_and_progression():
    # Test the timeout and progression of recommendations
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate a long time elapsed
    block = _get_single_block(render_progression.get_block(0.0))
    assert block is not None

    render_progression.end_block(1.0)  # far too long!
    # Check that the next recommendation is None
    block = render_progression.get_block(1.0)
    assert block is None

    sf = render_progression.end_frame_get_scalefactor()
    assert sf == 10.0

    assert render_progression.needs_refine()

    render_progression.start_frame(DrawReason.REFINE)
    start, num = _get_single_block(render_progression.get_block(0.0))
    assert start == 100
    assert num == int(100/config.TARGET_FPS) # took one second to render 100 particles, so recommendation should be this


def test_always_one_block():
    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.CHANGE)

    block = _get_single_block(render_progression.get_block(1.0)) # simulate a long time elapsed, but should still render at least one block
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
        _get_single_block(render_progression.get_block(0.0))

def test_export():
    """Test that export always recommends the full resolution"""

    render_progression = progressive_render.RenderProgression(1000, 100)
    render_progression.start_frame(DrawReason.EXPORT)
    block = _get_single_block(render_progression.get_block(0.0))
    assert block == (0, 1000)

def test_always_one_particle():
    """Test that always at least one particle is recommended (even if it takes a long time)"""

    render_progression = progressive_render.RenderProgression(1000, 3)
    render_progression.start_frame(DrawReason.CHANGE)

    # Simulate a long time elapsed
    block = _get_single_block(render_progression.get_block(0.0))
    assert block is not None

    render_progression.end_block(1.0)  # far too long!

    assert render_progression.get_block(1.0) is None # end of this frame
    render_progression.end_frame_get_scalefactor()
    assert render_progression.needs_refine()

    # Check that the refinement consists of at least one particle, even though technically
    # we would have rounded this down to zero particles
    render_progression.start_frame(DrawReason.REFINE)
    block = _get_single_block(render_progression.get_block(1.0))
    assert block == (3,1)

@pytest.fixture
def cell_progressive_render(num_particles=100000, num_side=10, num_part_to_take = 100):
    np.random.seed(1337)
    pos = np.random.uniform(0.0, 1.0, (num_particles, 3))

    cell_layout, order = progressive_render.CellLayout.from_positions(pos, 0.0, 1.0, num_side)
    pos = pos[order]

    render_progression = progressive_render.RenderProgressionWithCells(cell_layout, len(pos), num_part_to_take)

    return render_progression, pos

def test_blocks_with_layout(cell_progressive_render):

    render_progression, pos = cell_progressive_render
    cell_layout = render_progression._cell_layout

    total_particles = 0

    rendered = np.zeros(len(pos), dtype=np.int32)

    render_progression.start_frame(DrawReason.CHANGE)
    first_render = True

    while True:

        block = render_progression.get_block(0.0)

        for start, length in zip(*block):
            # find which cell this block belongs to
            cell_index = cell_layout.cell_index_from_offset(start)
            assert length!=0 # should not return zero length blocks
            cell_index_at_end = cell_layout.cell_index_from_offset(start+length-1)
            assert cell_index == cell_index_at_end
            total_particles+=length
            rendered[start:start+length]+=1

        if first_render:
            assert total_particles > 95 and total_particles < 105

        render_progression.end_block(0.0001)
        render_progression.end_frame_get_scalefactor()

        if render_progression.needs_refine():
            first_render = False
            render_progression.start_frame(DrawReason.REFINE)
        else:
            break

    assert (rendered==1).all()

    # check that render_progression returns None at end of frame

    render_progression.start_frame(DrawReason.CHANGE)
    npart = 0
    while (block := render_progression.get_block(0.0)):
        starts, lens = block
        npart+=lens.sum()
        render_progression.end_block(0.0)
    assert npart == len(pos)

def test_spatial_limits(cell_progressive_render):
    render_progression, pos = cell_progressive_render

    render_progression.select_sphere((0.5, 0.5, 0.5), 0.1)

    render_progression.start_frame(DrawReason.CHANGE)
    rendered = np.zeros(len(pos), dtype=np.int32)
    while (block:=render_progression.get_block(0.0)):
        for start, length in zip(*block):
            rendered[start:start+length]+=1

        render_progression.end_block(0.0)

    assert rendered.max() == 1

    r = np.linalg.norm(pos-0.5, axis=1)
    rendered_r = r[rendered==1]
    unrendered_r = r[rendered==0]

    assert (rendered_r<0.4).all()
    assert (unrendered_r>0.1).all()

def test_export_very_large():
    num_renders = 5
    render_progression = progressive_render.RenderProgression(config.MAX_PARTICLES_PER_EXPORT_RENDERCALL * num_renders)
    render_progression.start_frame(DrawReason.EXPORT)

    for blocknum in range(num_renders):
        # pretend we have a crazy long-running render. We should render the whole thing, but in a series
        # of blocks
        block = render_progression.get_block(100.0*blocknum)
        assert block is not None
        assert block[0][0] == config.MAX_PARTICLES_PER_EXPORT_RENDERCALL * blocknum
        assert block[1][0] == config.MAX_PARTICLES_PER_EXPORT_RENDERCALL
        render_progression.end_block(100.0 * (blocknum + 1))

    assert render_progression.get_block(100.0*num_renders) is None # finished now!

    clear = render_progression.start_frame(DrawReason.EXPORT)

    assert clear

