import numpy as np
import pytest
import wgpu

from topsy import split_buffers

@pytest.fixture
def device():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()

@pytest.fixture
def sb(max_per_buffer=15, num_particles=50):
    return split_buffers.SplitBuffers(num_particles, max_per_buffer)

def test_global_to_split(sb):

    assert sb.global_to_split(0, 10) == ([0], [0], [10])
    assert sb.global_to_split(0, 20) == ([0, 1], [0, 0], [15, 5])
    assert sb.global_to_split(0, 45) == ([0, 1, 2], [0, 0, 0], [15, 15, 15])
    assert sb.global_to_split(15, 10) == ([1], [0], [10])
    assert sb.global_to_split(14, 2) == ([0, 1], [14, 0], [1, 1])
    assert sb.global_to_split(20, 20) == ([1, 2], [5, 0], [10, 10])
    assert sb.global_to_split(49, 1) == ([3], [4], [1])
    assert sb.global_to_split(0, 50) == ([0, 1, 2, 3], [0, 0, 0, 0], [15, 15, 15, 5])

    with pytest.raises(ValueError):
        # too long:
        sb.global_to_split(0, 100)

    with pytest.raises(ValueError):
        sb.global_to_split(49,2)

def test_global_to_split_monotonic(sb):
    def generate_test_case():
        starts = np.random.randint(0, 50, size=5)
        starts = np.sort(starts)
        lengths = np.diff(starts)
        starts = starts[:-1]
        lengths = np.random.randint(lengths+1)

        # remove any cases i with lengths[i] = 0 from the starts and lengths lists:
        mask = lengths!=0
        starts = starts[mask]
        lengths = lengths[mask]

        return (starts, lengths)

    np.random.seed(1337)
    for i in range(100):
        starts, lengths = generate_test_case()

        # faster version to be used in code:
        results = sb.global_to_split_monotonic(starts, lengths)

        # slow version to act as check:
        results_slow = [([], []) for _ in range(sb._num_buffers)]

        for global_s, global_l in zip(starts, lengths):
            local_bufs, local_starts, local_lengths = sb.global_to_split(global_s, global_l)
            for lb, ls, ll in zip(local_bufs, local_starts, local_lengths):
                results_slow[lb][0].append(ls)
                results_slow[lb][1].append(ll)

        assert results == results_slow


def test_create_buffers(device, sb):
    # Create a buffer with 4 buffers of size 15 and one of size 5
    buffers = sb.create_buffers(device, 4, wgpu.BufferUsage.UNIFORM)
    assert len(buffers) == 4
    assert all([buf.size == 15*4 for buf in buffers[:-1]])
    assert buffers[-1].size == 5*4

def test_write_buffers(device, sb):
    buffers = sb.create_buffers(device, 4, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    data = np.arange(50, dtype=np.float32)

    with pytest.raises(ValueError):
        # wrong number of buffers
        sb.write_buffers(device, buffers[:-1], data)

    with pytest.raises(ValueError):
        # wrong number of particles
        sb.write_buffers(device, buffers, data[:-1])

    # should succeed
    sb.write_buffers(device, buffers, data)