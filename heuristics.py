'''

'''
from __future__ import division
__author__ = 'sivanov'
import itertools
import collections


def ffc(G, L, S, K):
    """
    First Friends Common heuristic is extracting K most common features among the first
    friends of the subscribed set S.
    :param G:
    :param L:
    :param S:
    :param K:
    :return:
    """
    print 'Starting First Friends Common heuristic...'
    print 'Selected nodes: ',
    ff = set()
    for node in S:
        ff.update(G[node])
    common = dict()
    for friend in ff:
        for feat in L[friend]:
            common[feat] = common.get(feat, 0) + 1
    chosen = sorted(common.iteritems(), key=lambda (dk, dv): dv, reverse=True)[:K]
    print ' '.join([f for (f, l) in chosen])
    return [f for (f, l) in chosen]


def mcf(L, K):
    """
    Most Common Features selects K most common features in the network.
    :param G:
    :param L:
    :param S:
    :param L:
    :return:
    """
    print 'Starting First Friends Common heuristic...'
    print 'Selected nodes: ',
    F = list(itertools.chain.from_iterable(L.values()))
    print ' '.join([f for (f, c) in collections.Counter(F).most_common(K)])
    return [f for (f, c) in collections.Counter(F).most_common(K)]


if __name__ == "__main__":
    console = []