"""
Test that runs tasks, where each task is allocated a set of processes.
"""

# Standard library
import random

import pytest

# Use full imports so we can run this with mpiexec externally
from schwimmbad.tests import TEST_MPI  # noqa
from schwimmbad.tests.test_pools import _task_parallel_function,isclose
from schwimmbad.mpi import MPIPool# noqa
import schwimmbad.mpi as schwimm_mpi
from schwimmbad import choose_pool
import sys

@pytest.mark.skip(True, reason="WTF")
def test_mpi_task_parallel():
    assert MPIPool.enabled()
    MPI  = schwimm_mpi.MPI
    world_comm = MPI.COMM_WORLD
    rank = world_comm.Get_rank()
    world_size = world_comm.size
    master_comm = world_comm.Create(world_comm.group.Incl([rank]))
    with choose_pool(mpi=True,processes=4,comm=world_comm) as pool:
        all_tasks = [[random.randint(0,100000) for i in range(1000)]]
        # test map alone
        for tasks in all_tasks:
            results = pool.map(_task_parallel_function, tasks)
            for r1, r2 in zip(results, [_task_parallel_function(x,master_comm) for x in tasks]):
                print("r1 {} r2 {}".format(r1,r2))
                assert isclose(r1, r2)
            assert len(results) == len(tasks)


if __name__ == '__main__':
    test_mpi_task_parallel()
