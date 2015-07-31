'''
Comparison of PIMUS to IM
'''
from __future__ import division
__author__ = 'sivanov'

#TODO add run_ic and run_lt functions
def run_ic(G, S, R):
    pass

def max_node(G, S, R, IC):
    max_u = None
    max_spread = 0
    for u in G:
        if IC:
            spread = run_ic(G, S + [u], R)
            if spread > max_spread:
                max_u = u
                max_spread = spread
    assert max_u is not None
    return max_u

def greedy_im(G, k, R, IC=True):
    S = []
    while len(S) < k:
        node = max_node(G, S, R, IC)
        S.append(node)
    return S


if __name__ == "__main__":
    console = []