import numpy as np

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pysparse as pys


def cs1d(y, Phi, Psi=None, optim='OMP', k=1000, tol=1e-8, osshape=None, verbose=True):
    r"""Solves the 1-d Compressive Sensing problem

    for sparse signal :math:`\bm x`

    .. math::
       {\bm y} = {\bm \Phi}{\bm x} + {\bf n},

    for non-sparse signal :math:`{\bm x} = {\bm \Psi}{\bm z}`

    .. math::
       {\bm y} = {\bm \Phi \Psi}{\bm z} + {\bm n},


    see https://iridescent.ink/aitrace/SignalProcessing/Sparse/LinearCompressiveSensing/index.html for details.

    Arguments
    ----------------------
    y : ndarray
        the mesurements :math:`{\bm y}`, if ndarray, each colum is a mesurement 
    Phi : 2darray
        the mesurement matrix :math:`{\bm \Phi}`

    Keyword Arguments
    ----------------------
    Psi : 2darray
        the dictionary :math:`{\bm \Psi}` (default: {None})
    optim : str
        optimization method, OMP, LASSO (default: {'OMP'})
    k : integer
        sparse degree for OMP, max iter for LASSO (default: size of :math:`{\bm x}` )
    tol : float
        tolerance of error (LASSO) (default: {1e-8})
    osshape : tuple
        orignal signal shape, such as an H-W image (default: {None})
    verbose : bool
        show log info (default: {True})

    Returns
    ----------------------
    x : ndarray
        reconstructed signal with sieof osshape

    """

    print("================in cs1d================")
    if y is None:
        print("===No raw data!")

    if verbose:
        print("===Type of Phi, y: ", Phi.dtype, y.dtype)
    cplxFlag = False

    if Psi is not None:
        # =================step1: Construct===========
        if verbose:
            print("===Construct sensing matrix: A = Phi*D...")
        A = np.matmul(Phi, Psi)
        Phi = None
        if verbose:
            print("===Done!...")
    else:
        A = Phi
        Phi = None

    if np.iscomplex(A).any() or np.iscomplex(y).any():
        cplxFlag = True
        if verbose:
            print("===Convert complex to real...")
        y = np.concatenate((np.real(y), np.imag(y)), axis=0)
        ReA = np.real(A)
        ImA = np.imag(A)
        A1 = np.concatenate((ReA, -ImA), axis=1)
        A2 = np.concatenate((ImA, ReA), axis=1)
        A = np.concatenate((A1, A2), axis=0)
        if verbose:
            print("===Done!")

    # if np.ndim(y) > 1:
    #     y = y.flatten()

    if optim is 'OMP':
        print(A.shape, y.shape)
        z = pys.romp(
            y, A, k=k, alpha=1e-6, normalize=False, tol=1e-16, verbose=verbose)
    if cplxFlag:
        z = np.split(z, 2)
        z = z[0] + 1j * z[1]
        if verbose:
            print("===size of z: ", z.shape)

    if Psi is not None:
        x = np.matmul(Psi, z)
    else:
        x = z
        z = None

    if osshape is not None:
        x = np.reshape(x, osshape)
        if verbose:
            print("===shape of x: ", x.shape)
            print("===Done!")
    return x
