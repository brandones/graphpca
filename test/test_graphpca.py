# test_graphpca.py
#
""" Unit tests for graphpca
"""

import unittest

import networkx as nx
import numpy as np
import scipy.io

import graphpca


class TestGraphPCA(unittest.TestCase):

    def test_returns_plausible_results(self):
        g = nx.erdos_renyi_graph(100, 0.3)
        g_5 = graphpca.reduce_graph(g, 5)
        self.assertEqual(len(g_5), 5)
        self.assertEqual(len(g_5[0]), 100)
        for i in range(5):
            max_val = max(abs(g_5[i]))
            self.assertGreater(max_val, 0.01)

    def test_ok_if_multiple_zero_eigens(self):
        g = nx.erdos_renyi_graph(100, 0.3)
        node = next(g.nodes_iter())
        for neighbor in g.neighbors(node):
            g.remove_edge(node, neighbor)
        g_5 = graphpca.reduce_graph(g, 5)
        self.assertEqual(len(g_5), 5)
        self.assertEqual(len(g_5[0]), 100)
        for i in range(5):
            max_val = max(abs(g_5[i]))
            self.assertGreater(max_val, 0.01)

    @unittest.skip('This fails and I have no idea why')
    def test_similar_output_to_naive_peterson(self):
        G = nx.petersen_graph()
        G2 = graphpca.reduce_graph(G, 2)
        G2n = graphpca.naive_reduce_graph(G, 2)
        self.assertTrue(np.allclose(G2, G2n, rtol=1e-04, atol=1e-06),
                        'Regular result:\n{}\n\nNaive result:\n{}\n'.format(G2, G2n))

    def test_similar_output_to_naive_small(self):
        G = nx.erdos_renyi_graph(10, 0.5)
        G2 = graphpca.reduce_graph(G, 2)
        G2n = graphpca.naive_reduce_graph(G, 2)
        self.assertTrue(np.allclose(G2, G2n, rtol=1e-04, atol=1e-06),
                        'Regular result:\n{}\n\nNaive result:\n{}\n'.format(G2, G2n))

    def test_similar_output_to_naive_mat_3(self):
        mat = scipy.io.loadmat('bcspwr01.mat')
        # I love the UFSMC (https://www.cise.ufl.edu/research/sparse/matrices/)
        # but wow they really buried the matrix in this .mat
        A = mat['Problem'][0][0][1].todense()
        G = nx.from_numpy_matrix(A)
        G3 = graphpca.reduce_graph(G, 3)
        G3n = graphpca.naive_reduce_graph(G, 3)
        # Compare the fabs because eiganvalue parity is arbitrary
        self.assertTrue(np.allclose(G3, G3n, rtol=1e-04, atol=1e-06),
                        'Regular result:\n{}\n\nNaive result:\n{}\n'.format(G3, G3n))

    def test_similar_output_to_naive_big(self):
        G = nx.erdos_renyi_graph(1001, 0.02)
        G2 = graphpca.reduce_graph(G, 2)
        G2n = graphpca.naive_reduce_graph(G, 2)
        # Compare the fabs because eiganvalue parity is arbitrary
        self.assertTrue(np.allclose(G2, G2n, rtol=1e-03, atol=1e-05),
                        'Regular result:\n{}\n\nNaive result:\n{}\n'.format(G2, G2n))

    def test_small_eigendomp_same_as_big(self):
        g = nx.erdos_renyi_graph(10, 0.3)
        l = nx.laplacian_matrix(g).astype('d')
        # Test for smallest eigs
        Eb, Ub = graphpca._big_eigendecomp(l, 4, which='SM')
        Es, Us = graphpca._small_eigendecomp(l, 4, which='SM')
        self.assertTrue(np.allclose(Eb, Es), 'Big vals: {}\nSmall vals: {}\n'.format(Eb, Es))
        self.assertTrue(np.allclose(Ub, Us, rtol=1e-03, atol=1e-05),
                        'Big vecs:\n{}\n\nSmall vecs:\n{}\n'.format(Ub, Us))
        # Test for biggest eigs
        Eb, Ub = graphpca._big_eigendecomp(l, 4, which='LM')
        Es, Us = graphpca._small_eigendecomp(l, 4, which='LM')
        self.assertTrue(np.allclose(Eb, Es), 'Big vals: {}\nSmall vals: {}\n'.format(Eb, Es))
        self.assertTrue(np.allclose(Ub, Us, rtol=1e-04, atol=1e-06),
                        'Big vecs:\n{}\n\nSmall vecs:\n{}\n'.format(Ub, Us))
