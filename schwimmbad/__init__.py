# coding: utf-8
"""
Contributions by:
- Peter K. G. Williams
- Júlio Hoffimann Mendes
- Dan Foreman-Mackey

Implementations of four different types of processing pools:

    - MPIPool: An MPI pool.
    - MultiPool: A multiprocessing for local parallelization.
    - SerialPool: A serial pool, which uses the built-in ``map`` function

"""
import pkg_resources

__version__ = pkg_resources.require(__package__)[0].version
__author__ = "Adrian Price-Whelan <adrianmpw@gmail.com>"

# Standard library
import sys
import logging
log = logging.getLogger(__name__)
_VERBOSE = 5

from .multiprocessing import MultiPool
from .serial import SerialPool
from .mpi import MPIPool, MPI
from .jl import JoblibPool

def choose_pool(mpi=False, processes=1, comm=None, **kwargs):
    """
    Choose between the different pools given options from, e.g., argparse.

    Parameters
    ----------
    mpi : bool, optional
        Use the MPI processing pool, :class:`~schwimmbad.mpi.MPIPool`. By
        default, ``False``, will use the :class:`~schwimmbad.serial.SerialPool`.
    processes : int, optional
        Use the multiprocessing pool,
        :class:`~schwimmbad.multiprocessing.MultiPool`, with this number of
        processes. By default, ``processes=1``, will use the
        :class:`~schwimmbad.serial.SerialPool`.
    comm: 
        top level communicator for setting up parallel tasks. By default, ``None``, will just use 1 process for every task.
    **kwargs
        Any additional kwargs are passed in to the pool class initializer
        selected by the arguments.
    """
    if mpi:
        if not MPIPool.enabled():
            raise SystemError("Tried to run with MPI but MPIPool not enabled.")
        if comm==None:
            pool = MPIPool(**kwargs) #this is the standard case, 1 process per task
        else:
            if processes+1==comm.size:
                return MPIPool(comm=comm,**kwargs) #just run with a main comm
            #allow helper processes per task
            number_main_processes = processes
            #construct the main communicator
            main_group = comm.group.Incl(range(0,number_main_processes+1))
            main_comm = comm.Create(main_group)
            #construct task-local communicators 
            task_proc_index = (number_main_processes+1)
            #distributed remaining processes for each task
            procs_per_task = (comm.size-(number_main_processes+1))//(number_main_processes)
            if (comm.size-(number_main_processes+1))%number_main_processes!=0:
                log.info("Total ranks ({}) - 1 is not a multiple of number of tasks ({})".format(comm.size,number_main_processes))
            #set up the task_communicator for each rank
            extra_procs = (comm.size-(number_main_processes+1))%(number_main_processes)
            for index, i in enumerate(range(task_proc_index,comm.size,procs_per_task)): 
                task_main_rank = index+1
                if task_main_rank > number_main_processes: break
                end_point = i+procs_per_task
                if task_main_rank==number_main_processes: end_point+=extra_procs
                task_comm = comm.Create(comm.group.Incl([task_main_rank]+list(range(i,end_point))))
                if comm.Get_rank() == task_main_rank or (comm.Get_rank()>=i and  comm.Get_rank()<i+end_point):
                    final_task_comm=task_comm
                if comm.rank==0:
                    final_task_comm=task_comm
            task_comm=final_task_comm
            pool = MPIPool(comm=main_comm,task_comm=task_comm)
        log.info("Running with MPI on {0} cores".format(pool.size))
        return pool

    elif processes != 1 and MultiPool.enabled():
        log.info("Running with MultiPool on {0} cores".format(processes))
        return MultiPool(processes=processes, **kwargs)

    else:
        log.info("Running with SerialPool")
        return SerialPool(**kwargs)
