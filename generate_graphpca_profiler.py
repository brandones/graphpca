# -*- coding: utf-8 -*-
#
# generate_graphpca_profiler.py
#
# Copyright 2016 Socos LLC
#

import os

import matplotlib.pyplot as plt
import networkx as nx
import graphpca

import timeit


def main():
    # n_vs_t = profile_n()
    # plt.plot(*n_vs_t)
    # plt.show()
    d_vs_t = profile_d()
    plt.plot(*d_vs_t)
    plt.show()


def profile_n():
    n_vs_t = [[], []]
    print 'Timing graphpca(G, 5) on Erdos-Renyi Graph nx.fast_gnp_random_graph(n, p)'
    print '\t'.join(('n', 'p', 't (ms)'))
    n_range = [int(pow(10.0, i/2.0)) for i in range(2, 11)]
    p_range = [0.2, 0.2] + [2 * pow(10.0, -i/2.0) for i in range(2, 9)]
    for n, p in zip(n_range, p_range):
        g = nx.fast_gnp_random_graph(n, p)
        tic = timeit.default_timer()
        graphpca.reduce_graph(g, 3)
        toc = timeit.default_timer()
        print '\t'.join((str(n), str(p), str((toc - tic) * 1000)))
        n_vs_t[0].append(n)
        n_vs_t[1].append(toc - tic)
    return n_vs_t


def generate_profile_file(iterations=9, d=3, steps_per_10_factor=2):
    code = ['"""THIS FILE IS GENERATED. COMPUTERS CAN WRITE CODE NOW TOO."""',
            '',
            'import timeit',
            'import networkx as nx',
            'import graphpca',
            '',
            'print "Timing graphpca(G, 3) on Erdos-Renyi Graph nx.fast_gnp_random_graph(n, p)"',
            'print "\t".join(("n", "p", "t (ms)"))']
    fcn_names = []
    s = steps_per_10_factor
    n_range = [int(pow(10.0, float(i)/s)) for i in range(s, iterations + s)]
    p_range = [0.2] * s + [2 * pow(10.0, -float(i)/s) for i in range(s, iterations)]
    for n, p in zip(n_range, p_range):
        fcn_name = 'profile_{}'.format(n)
        fcn_names.append(fcn_name)
        code.extend([
            '',
            'def {}():'.format(fcn_name),
            '    tic = timeit.default_timer()',
            '    graphpca.reduce_graph(nx.fast_gnp_random_graph({}, {}), {})'.format(n, p, d),
            '    toc = timeit.default_timer()',
            '    print "\t".join((str({}), str({}), str((toc - tic) * 1000)))'.format(n, p),
        ])
    code.append('')
    code.extend(['{}()'.format(fcn_name) for fcn_name in fcn_names])

    with open('profile_graphpca.py', 'w') as f:
        f.writelines([l + '\n' for l in code])


def profile_d():
    d_vs_t = [[], []]
    print 'Timing graphpca(G, d) on Erdos-Renyi Graph nx.fast_gnp_random_graph(1000, 0.02)'
    print '\t'.join(('d', 't (ms)'))
    g = nx.fast_gnp_random_graph(1000, 0.02)
    for d in range(1, 950, 30):
        tic = timeit.default_timer()
        graphpca.reduce_graph(g, d)
        toc = timeit.default_timer()
        print '\t'.join((str(d), str((toc - tic) * 1000)))
        d_vs_t[0].append(d)
        d_vs_t[1].append(toc - tic)
    return d_vs_t


if __name__ == '__main__':
    main()

