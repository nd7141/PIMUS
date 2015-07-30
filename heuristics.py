'''

'''
from __future__ import division
__author__ = 'sivanov'

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


if __name__ == "__main__":
    console = []