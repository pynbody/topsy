import numpy as np
from topsy import cell_layout

def test_randomization():
    offsets = np.array([0, 10, 30])
    lengths = np.array([10, 20, 20])
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0,0.0,0.0]])
    cl = cell_layout.CellLayout(centres, offsets, lengths)

    order = cl.randomize_within_cells()
    assert (order[:10]<10).all()
    assert ((order[10:30]<30) & (order[10:30]>=10)).all()
    assert ((order[30:50]<50) & (order[30:50]>=30)).all()

    assert (order != np.arange(50)).any()

def test_from_positions():
    Npart = 10000
    Nside = 10
    Ncells_to_test = 100
    min_pos = -1.0
    max_pos = 1.0

    np.random.seed(1337)

    pos = np.random.uniform(min_pos, max_pos, (Npart, 3))

    cl, order = cell_layout.CellLayout.from_positions(pos, min_pos, max_pos, Nside)

    pos = pos[order]

    for test_cell in np.random.randint(0, Nside**3, Ncells_to_test):

        # get the cell slice
        cell_slice = cl.cell_slice(test_cell)

        # get the positions of the particles in the cell
        test_pos = pos[cell_slice]

        # get the cell centre
        cell_centre = cl._centres[test_cell]

        cell_size = (max_pos - min_pos) / Nside

        # check that all the particles are within the cell
        assert ((test_pos > (cell_centre - 0.5*cell_size)) & (test_pos < (cell_centre + 0.5*cell_size))).all()



