'''

'''
from __future__ import division
__author__ = 'sivanov'
import itertools
import collections
import random


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
    ff = set()
    for node in S:
        ff.update(G[node])
    common = dict()
    for friend in ff:
        for feat in L[friend]:
            common[feat] = common.get(feat, 0) + 1
    chosen = sorted(common.iteritems(), key=lambda (dk, dv): dv, reverse=True)[:K]
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
    F = list(itertools.chain.from_iterable(L.values()))
    return [f for (f, c) in collections.Counter(F).most_common(K)]


def rf(L, K):
    """
    Random selects K random features.
    :param L:
    :param K:
    :return:
    """
    F = set(list(itertools.chain.from_iterable(L.values())))
    return random.sample(F, K)


if __name__ == "__main__":
    console = []