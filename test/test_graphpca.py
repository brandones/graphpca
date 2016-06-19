# test_graphpca.py
#
""" Unit tests for graphpca
"""

import unittest

import networkx as nx
import numpy as np

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
