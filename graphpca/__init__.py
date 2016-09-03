# __init__.py.py
#

import logging

import networkx as nx
import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger(__name__)


def reduce_graph(nx_graph, output_dim, add_supernode=False):
    """
    Run PCA on the ETCD of the input NetworkX graph

    We skip calculating the actual ETCD for efficiency. The ETCD is given by
    the Moore-Penrose pseudoinverse of the Laplacian of the input graph. The
    input graph is G, the Laplacian is L, and its pseudoinverse is pinv(L). We
    actually only care about the eigenvectors associated with the top
    output_dim eigenvalues. Therefore we use the fact that::

        eigvals(pinv(A)) == [1/e for e in eigvals(A) if e != 0 else e]

    and the corresponding eigenvectors are the same. Further, we only care
    about the top output_dim eigenpairs of pinv(L), which correspond to the
    smallest nonzero eigenvalues of L. We use `scipy.sparse.linalg.eigs` with
    `which=SM` to calculate eigenpairs, which includes zero eigenpairs.
    Therefore in order to calculate the smallest nonzero eigenpairs we need
    to calculate the smallest ``output_dim + nullity`` eigenpairs. We compute
    the nullity using the convenient fact that the nullity of L is equal
    to the number of connected components in G.

    Parameters
    ----------
    nx_graph : :class:`nx.Graph` or :class:`nx.DiGraph`
        The graph to be reduced
    output_dim : int
        The number of dimensions to reduce to
    add_supernode : bool
        If True, adds a node to the graph that is connected to every other node
        in the graph. This reduces the nullspace of the Laplacian to 1, making
        there many fewer eigenpairs that need to be computed. The cost is minor
        information loss.

    Returns
    -------
    :class:`numpy.ndarray`
        The reduced data in output_dim dimensions

    """
    LOG.debug('Entering reduce_graph')
    assert output_dim < len(nx_graph)
    LOG.info('Calculating Laplacian L')
    L = nx.laplacian_matrix(nx_graph).astype('d')
    LOG.debug('L.shape: {}'.format(L.shape))
    if add_supernode:
        L = _add_supernode_to_laplacian(L)
    LOG.info('Calculating nullity of L as connected components of nx_graph')
    nullity = nx.number_connected_components(nx_graph)
    LOG.info('Calculating smallest eigenvalues of L & corresponding eigenvectors')
    (E, U) = _sane_eigendecomp(L, output_dim + nullity, which='SM')
    LOG.debug('Eigenvalues: {}'.format(E))
    LOG.info('Assembling PCA result')
    # Remove the 0 eigenvalues and corresponding eigenvectors
    # Use tolerance value 10 x from numpy.linalg.matrix_rank
    tol = E.max() * max(L.shape) * np.finfo(float).eps * 10
    LOG.debug('Using tolerance {}'.format(tol))
    zero_indexes = [i for i in range(len(E)) if abs(E[i]) < tol]
    E = np.delete(E, zero_indexes)
    U = np.delete(U, zero_indexes, axis=1)
    # If we added a supernode, now remove it
    if add_supernode:
        E = E[:-1]
        U = U[:-1, :]
    # Invert eigenvalues to get largest eigenvalues of L-pseudoinverse
    Ep = 1/E
    LOG.debug('Filtered & Inverted Eigenvalues: {}'.format(Ep))
    # Orient Eigenvectors
    _orient_eigenvectors(U)
    # Assemble into the right structure
    X = np.zeros((output_dim, len(nx_graph)))
    sqrtEp = np.sqrt(Ep)
    for i in range(output_dim):
        X[i, :] = sqrtEp[i] * U[:, i]
    return X


def _add_supernode_to_laplacian(L):
    L_padded = np.ones([n+1 for n in L.shape])
    L_padded[:-1, :-1] = L.todense()
    return L_padded


def _orient_eigenvectors(U):
    for i in range(U.shape[1]):
        if U[0, i] < 0.0:
            U[:, i] = - U[:, i]
    return U


def _sane_eigendecomp(M, output_dim, which='SM'):
    if M.shape[0] < 1000:
        return _small_eigendecomp(M, output_dim, which=which)
    else:
        if which == 'SM':
            # Use shift-invert method to calculate smallest eigenpairs.
            # Use very small sigma since `sigma=0.0` fails with
            #    RuntimeError: Factor is exactly singular
            return _big_eigendecomp(M, output_dim, sigma=0.00001, which='LM')
        else:
            return _big_eigendecomp(M, output_dim, which=which)


def _small_eigendecomp(M, output_dim, which):
    LOG.debug('Using _small_eigendecomp')
    if scipy.sparse.issparse(M):
        M = M.todense()
    E, U = scipy.linalg.eigh(M)
    # Cut out eigenpairs
    if which == 'SM':
        E = E[:output_dim]
        U = U[:, :output_dim]
        U = _orient_eigenvectors(U)
    elif which == 'LM':
        E = E[E.shape[0] - output_dim:]
        U = U[:, U.shape[1] - output_dim:]
        U = _orient_eigenvectors(U)
    else:
        raise NotImplementedError('Unknown setting for `which`: {}'.format(which))
    return E, U


def _big_eigendecomp(M, output_dim, tol=0.000000001, _attempt=0, **kwargs):
    LOG.debug('Using _big_eigendecomp')
    try:
        E, U = scipy.sparse.linalg.eigsh(M, output_dim, tol=tol, **kwargs)
        U = _orient_eigenvectors(U)
        return E, U
    except ArpackNoConvergence, e:
        if _attempt > 2:
          LOG.error('Eigendecomp did not converge. Bailing.')
          raise e
        LOG.info(e)
        new_tol = tol * 10
        LOG.info('Eigendecomp failed to converge, retrying with tolerance {}'.format(new_tol))
        return _big_eigendecomp(M, output_dim, tol=new_tol, _attempt=_attempt+1)


def naive_reduce_graph(nx_graph, output_dim):
    """
    Run PCA on the ETCD of a NetworkX graph using a slow but precise method

    This is the method that calculates the actual ETCD. It calculates the
    Moore-Penrose pseudoinverse of the Laplacian of the input graph. We return
    the first output_dim dimensions of the ETCD, ordered by decreasing
    eigenvalue.

    This method starts to take a very, very long time as graph size reaches
    into the thousands due to the matrix inversion.

    Parameters
    ----------
    nx_graph : :class:`nx.Graph` or :class:`nx.DiGraph`
        The graph to be reduced
    output_dim : int
        The number of dimensions to reduce to

    Returns
    -------
    :class:`numpy.ndarray`
        The reduced data in output_dim dimensions
    """
    LOG.debug('Entering naive_reduce_graph')
    L = nx.laplacian_matrix(nx_graph).astype('f').todense()
    LOG.info('Calculating Moore-Penrose inverse of the Laplacian L')
    Li = np.linalg.pinv(L)
    LOG.info('Calculating largest eigenvalues of L-inverse & corresponding eigenvectors')
    (E, U) = _sane_eigendecomp(Li, output_dim, which='LM')
    # Flip so largest eigen first
    E = E[::-1]
    U = np.fliplr(U)
    LOG.debug('Eigenvalues: {}'.format(E))
    LOG.info('Assembling PCA result')
    # Assemble into the right structure
    X = np.zeros((output_dim, len(nx_graph)))
    sqrtE = np.sqrt(E)
    for i in range(output_dim):
        X[i, :] = sqrtE[i] * U[:, i]
    return X


def plot_2d(pca_output_2d, colormap_name='winter'):
    import matplotlib.pyplot as plt
    x = pca_output_2d[0, :]
    y = pca_output_2d[1, :]
    colormap = plt.get_cmap(colormap_name)
    colors = colormap(np.linspace(0, 1, (len(x))))
    plt.scatter(x, y, c=colors)
    plt.show()
    return plt


def draw_graph(nx_graph, naive=False):
    """
    Draws the input graph on two axes with lines between the nodes

    Positions of the nodes are determined with reduce_graph, of course.

    Parameters
    ----------
    nx_graph : :class:`nx.Graph` or :class:`nx.DiGraph`
        The graph to be plotted
    """
    import matplotlib.pyplot as plt
    if naive:
        reduced_2 = naive_reduce_graph(nx_graph, 2)
    else:
        reduced_2 = reduce_graph(nx_graph, 2)
    for edge in nx_graph.edges():
        plt.plot([reduced_2[0, edge[0]], reduced_2[0, edge[1]]],
                 [reduced_2[1, edge[0]], reduced_2[1, edge[1]]],
                 'b-')
    plot_2d(reduced_2)
