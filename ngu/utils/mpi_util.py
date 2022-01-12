# Code reference:
# https://github.com/openai/random-network-distillation/blob/master/mpi_util.py
import numpy as np
from mpi4py import MPI


def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n + 1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]


def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis + 1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), comm=None, use_mpi=True):
        self.mean = np.zeros(shape, 'float64')
        self.use_mpi = use_mpi
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm

    def update(self, x):
        if self.use_mpi:
            batch_mean, batch_std, batch_count = mpi_moments(x, axis=0, comm=self.comm)
        else:
            batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
