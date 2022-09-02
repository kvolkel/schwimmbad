# Standard library
import time
import random

# Project
from ..serial import SerialPool
from ..multiprocessing import MultiPool
from ..jl import JoblibPool
from ..mpi import MPIPool

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _function(x):
    """
    Wastes a random amount of time, returns the number 42.01
    """
    time.sleep(random.random()*4E-4 + 1E-4)
    return 42.01


def _batch_function(batch):
    """
    Wastes a random amount of time, returns the number 42.01 - accepts a batch
    of tasks instead of a single task
    """
    results = []
    for x in batch:
        time.sleep(random.random()*4E-4 + 1E-4)
        results.append(42.01)
    return results


def _task_parallel_function(x,task_comm=None):
    import sys
    assert task_comm!=None
    """
    Calculates the series 0+1+2...+x-1 across the task communicator task_comm
    """
    task_rank = task_comm.Get_rank()
    numbers_per_rank = x//task_comm.size
    last_rank_extra = x%task_comm.size
    rank_start = task_rank*numbers_per_rank
    rank_end =task_rank*numbers_per_rank+numbers_per_rank
    if task_rank==task_comm.size-1:
        rank_end+=last_rank_extra
    #print("x {} start {} end {} rank {}".format(x,rank_start,rank_end,task_comm.rank))
    sys.stdout.flush()
    rank_sum = sum(range(rank_start,rank_end))

    gathered_sum = task_comm.gather(rank_sum,root=0)

    if task_rank==0:
        return sum(gathered_sum)

    return None
    


class PoolTestBase(object):

    all_tasks = [[random.random() for i in range(1000)]]

    def test_init_close(self):
        pool = self.PoolClass()
        pool.close()

    def _make_pool(self):
        return self.PoolClass()

    def test_map(self):
        pool = self._make_pool()

        for tasks in self.all_tasks:
            results = pool.map(_function, tasks)
            count = 0
            for r1,r2 in zip(results, [_function(x) for x in tasks]):
                assert isclose(r1, r2)
                count += 1

            assert count == len(tasks) # to make sure iterations through the loop happen!

        pool.close()

    def test_map_callback(self):
        pool = self._make_pool()

        for tasks in self.all_tasks:

            mylist = []
            def _callback(x):
                assert isclose(x, 42.01)
                mylist.append(x)

            results = pool.map(_function, tasks, callback=_callback)
            for r1,r2 in zip(results, [_function(x) for x in tasks]):
                assert isclose(r1, r2)
            assert len(mylist) == len(tasks)

        pool.close()

class TestSerialPool(PoolTestBase):
    def setup(self):
        self.PoolClass = SerialPool

class TestMultiPool(PoolTestBase):
    def setup(self):
        self.PoolClass = MultiPool

class TestJoblibPool(PoolTestBase):
    def setup(self):
        self.PoolClass = JoblibPool
