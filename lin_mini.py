from typing import Tuple
import numpy as np

# try to import numba module
# noinspection PyBroadException
try:
    from numba import jit
    HAS_NUMBA = True
except Exception as _:
    jit = None
    HAS_NUMBA = False

__NAME__ = 'linear_minimization.py'

# must catch if we do not have the jit decorator and define our own
if not HAS_NUMBA:
    def jit(**options):
        # don't use options but they are required to match jit definition
        _ = options

        # define decorator
        def decorator(func):
            # define wrapper
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            # return wrapper
            return wrapper
        # return decorator
        return decorator


# Set "nopython" mode for best performance, equivalent to @nji
@jit(nopython=True)
def lin_mini(vector: np.ndarray, sample: np.ndarray, mm: np.ndarray,
             v: np.ndarray, sz_sample: Tuple[int], case: int,
             recon: np.ndarray, amps: np.ndarray,
             no_recon: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear minimization of sample with vector, iwth input adapted for @jit

    Used internally in linear_minimization - you probably should
    use the linear_minization function instead of this directly

    :param vector: vector of N elements
    :param sample: sample: matrix N * M each M column is adjusted in
                   amplitude to minimize the chi2 according to the input vector
    :param mm: zero filled vector for filling size = M
    :param v: zero filled vector for filling size = M
    :param sz_sample: tuple the shape of the sample (N, M)
    :param case: int, if case = 1 then vector.shape[0] = sample.shape[1],
                 if case = 2 vector.shape[0] = sample.shape[0]
    :param recon: zero filled vector size = N, recon output
    :param amps: zero filled vector size = M, amplitudes output
    :param no_recon: boolean if True does not calculate recon
                     (output = input for recon)

    :returns: amps, recon
    """
    # do not set function name here -- cannot use functions here
    # case 1
    if case == 1:
        # fill-in the co-variance matrix
        for i in range(sz_sample[0]):
            for j in range(i, sz_sample[0]):
                mm[i, j] = np.sum(sample[i, :] * sample[j, :])
                # we know the matrix is symetric, we fill the other half
                # of the diagonal directly
                mm[j, i] = mm[i, j]
            # dot-product of vector with sample columns
            v[i] = np.sum(vector * sample[i, :])
        # if the matrix cannot we inverted because the determinant is zero,
        # then we return a NaN for all outputs
        if np.linalg.det(mm) == 0:
            amps = np.zeros(sz_sample[0]) + np.nan
            recon = np.zeros_like(v)
            return amps, recon
        # invert coveriance matrix
        inv = np.linalg.inv(mm)
        # retrieve amplitudes
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]
        # reconstruction of the best-fit from the input sample and derived
        # amplitudes
        if not no_recon:
            for i in range(sz_sample[0]):
                recon += amps[i] * sample[i, :]
        return amps, recon
    # same as for case 1 but with axis flipped
    if case == 2:
        # fill-in the co-variance matrix
        for i in range(sz_sample[1]):
            for j in range(i, sz_sample[1]):
                mm[i, j] = np.sum(sample[:, i] * sample[:, j])
                # we know the matrix is symetric, we fill the other half
                # of the diagonal directly
                mm[j, i] = mm[i, j]
            # dot-product of vector with sample columns
            v[i] = np.sum(vector * sample[:, i])
        # if the matrix cannot we inverted because the determinant is zero,
        # then we return a NaN for all outputs
        if np.linalg.det(mm) == 0:
            return amps, recon
        # invert coveriance matrix
        inv = np.linalg.inv(mm)
        # retrieve amplitudes
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]
        # reconstruction of the best-fit from the input sample and derived
        # amplitudes
        if not no_recon:
            for i in range(sz_sample[1]):
                recon += amps[i] * sample[:, i]
        return amps, recon


def linear_minimization(vector: np.ndarray, sample: np.ndarray,
                        no_recon: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper function that sets everything for the @jit later
    In particular, we avoid the np.zeros that are not handled
    by numba, size of input vectors and sample to be adjusted

    :param vector: 2d matrix that is (N x M) or (M x N)
    :param sample: 1d vector of length N
    :param no_recon: bool, if True does not calculate recon
    :return:
    """
    # set function name
    func_name = __NAME__ + 'linear_minimization(I)'
    # get sample and vector shapes
    sz_sample = sample.shape  # 1d vector of length N
    sz_vector = vector.shape  # 2d matrix that is N x M or M x N
    # define which way the sample is flipped relative to the input vector
    if sz_vector[0] == sz_sample[0]:
        case = 2
    elif sz_vector[0] == sz_sample[1]:
        case = 1
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        print(emsg)
        raise ValueError(emsg.format(func_name))
    # ----------------------------------------------------------------------
    # Part A) we deal with NaNs
    # ----------------------------------------------------------------------
    # set up keep vector
    keep = None
    # we check if there are NaNs in the vector or the sample
    # if there are NaNs, we'll fit the rest of the domain
    isnan = (np.sum(np.isnan(vector)) != 0) or (np.sum(np.isnan(sample)) != 0)
    # ----------------------------------------------------------------------
    # case 1: sample is not flipped relative to the input vector
    if case == 1:
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=0))
            # redefine the input vector to avoid NaNs
            vector = vector[keep]
            sample = sample[:, keep]
            # re-find shapes
            sz_sample = sample.shape  # 1d vector of length N
        # matrix of covariances
        mm = np.zeros([sz_sample[0], sz_sample[0]])
        # cross-terms of vector and columns of sample
        vec = np.zeros(sz_sample[0])
        # reconstructed amplitudes
        amps = np.zeros(sz_sample[0])
        # reconstruted fit
        recon = np.zeros(sz_sample[1])
    # ----------------------------------------------------------------------
    # case 2: sample is flipped relative to the input vector
    elif case == 2:
        # same as for case 1, but with axis flipped
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=1))
            vector = vector[keep]
            sample = sample[keep, :]
            # re-find shapes
            sz_sample = sample.shape  # 1d vector of length N
        mm = np.zeros([sz_sample[1], sz_sample[1]])
        vec = np.zeros(sz_sample[1])
        amps = np.zeros(sz_sample[1])
        recon = np.zeros(sz_sample[0])
    # ----------------------------------------------------------------------
    # should not get here (so just repeat the raise from earlier)
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        raise ValueError(emsg.format(func_name))

    # ----------------------------------------------------------------------
    # Part B) pass to optimized linear minimization
    # ----------------------------------------------------------------------
    # pass all variables and pre-formatted vectors to the @jit part of the code
    amp_out, recon_out = lin_mini(vector, sample, mm, vec, sz_sample,
                                  case, recon, amps, no_recon=no_recon)
    # ----------------------------------------------------------------------
    # if we had NaNs in the first place, we create a reconstructed vector
    # that has the same size as the input vector, but pad with NaNs values
    # for which we cannot derive a value
    if isnan:
        recon_out2 = np.zeros_like(keep) + np.nan
        recon_out2[keep] = recon_out
        recon_out = recon_out2

    return amp_out, recon_out
