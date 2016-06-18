import networkx as nx
import numpy
import scipy.io
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence


def reduce_from_matlab(mat_path, output_dim):
    mat = scipy.io.loadmat(mat_path)
    A = mat['ans'].todense()
    G = nx.from_numpy_matrix(A)
    return reduce_graph(G, output_dim)


def reduce_graph(nx_graph, output_dim):
    assert output_dim < len(nx_graph)
    print 'Calculating Laplacian L'
    L = nx.laplacian_matrix(nx_graph).astype('f')
    print 'Calculating smallest eigenvalues of L & corresponding eigenvectors'
    (E, U) = retry_eigendecomp(L, output_dim + 1, which='SM')
    print 'Assembling PCA result'
    # Remove the 0 eigenvalue and corresponding eigenvector
    assert abs(E[0]) < 0.0001, E
    E = E[1:]
    U = U[:, 1:]
    # Invert eigenvalues to get largest eigenvalues of L-pseudoinverse
    Ep = 1/E
    # Assemble into the right structure
    X = numpy.zeros((output_dim, len(nx_graph)))
    sqrtEp = numpy.sqrt(Ep)
    for i in range(output_dim):
        X[i, :] = sqrtEp[i] * U[:, i]
    return X


def retry_eigendecomp(M, output_dim, tol=0, _attempt=0, **kwargs):
    try:
        return scipy.sparse.linalg.eigs(M, output_dim, tol=tol, **kwargs)
    except ArpackNoConvergence, e:
        if _attempt > 2:
          print 'Eigendecomp did not converge. Bailing.'
          raise e
        print e
        if tol == 0:
            tol = 0.000000001
        new_tol = tol * 10
        print 'Eigendecomp failed to converge, retrying with tolerance {}'.format(new_tol)
        return retry_eigendecomp(M, output_dim, tol=new_tol, _attempt=_attempt+1)


def plot_2d(pca_output_2d):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    x = pca_output_2d[0, :]
    y = pca_output_2d[1, :]
    colors = cm.cool(len(x))
    plt.scatter(x, y, c=colors)
    plt.show()
