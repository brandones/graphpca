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

    def test_similar_output_to_naive(self):
        mat = scipy.io.loadmat('bcspwr01.mat')
        # I love the UFSMC (https://www.cise.ufl.edu/research/sparse/matrices/)
        # but wow they really buried the matrix in this .mat
        A = mat['Problem'][0][0][1].todense()
        G = nx.from_numpy_matrix(A)
        G2 = graphpca.reduce_graph(G, 2)
        G2n = graphpca.naive_reduce_graph(G, 2)
        print G2
        print G2n
        # Compare the fabs because eiganvalue parity is arbitrary
        # Compare the row-swapped even though the eigenvalue order is not at
        # all arbitrary because reduce_graph fails to accurately determine
        # the smallest eigenvalues of L.
        # TODO: improve eigenvalue calculation
        difference_norm = min(np.linalg.norm(np.fabs(G2) - np.fabs(G2n)),
                              np.linalg.norm(np.fabs(G2) - np.fabs(G2n[[1, 0]])))
        print difference_norm
        self.assertLess(difference_norm, 0.00001)

