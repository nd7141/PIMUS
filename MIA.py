# author: sivanov
# date: 27 Oct 2015
from __future__ import division
import networkx as nx
import pandas as pd
import time
import random
import math
from collections import Counter
from itertools import chain, combinations
import multiprocessing as mp

def read_graph(filename, directed=True, sep=' ', header = None):
    """
    Create networkx graph using pandas.
    :param filename: every line (u, v)
    :param directed: boolean
    :param sep: separator in file
    :return
    """
    df = pd.read_csv(filename, sep=sep, header = header)
    if directed:
        G = nx.from_pandas_dataframe(df, 0, 1, create_using=nx.DiGraph())
    else:
        G = nx.from_pandas_dataframe(df, 0, 1)
    return G

def read_graph2(filename, directed=False):
    """
    Create networkx graph reading file.
    :param filename: every line (u, v)
    :param directed: boolean
    :return:
    """
    if not directed:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    with open(filename) as f:
        for line in f:
            d = line.split()
            G.add_edge(int(d[0]), int(d[1]))
    return G

def add_graph_attributes(G, filename):
    """
    Add features as node attributes and construct feature -> edges
    :param G: networkx graph
    :param filename: (u f1 f2 f3 ...); u is mandatory in graph -- the node
    f1 f2 f3 ... - arbitrary number of features
    :return: Ef: dictionary f -> edges that it affects
    """
    Ef = dict() # feature -> edges
    Nf = dict() # node -> features
    with open(filename) as f:
        for line in f:
            d = line.split()
            u = int(d[0])
            features = d[1:]
            for f in features:
                Ef.setdefault(f, []).extend(G.in_edges(u)) # add feature-dependent edges
            G.node[u]['Fu'] = features
            Nf[u] = features
    return Ef, Nf

def read_probabilities(filename, sep=' '):
    """
    Creates a dataframe object indexed by endpoints of edge (u, v)
    :param filename: (u, v, p)
    :param sep: separator in the file
    :return:
    """
    df = pd.read_csv(filename, sep=sep, header = None)
    return df.set_index([0, 1])

def increase_probabilities(G, B, Q, F, E, P):
    """
    Increase probabilities of edges E depending on selected features F. Returns previous probabilities of changed edges.
    :param G: graph that has nodes attributes as features for nodes
    :param B: dataframe indexed by two endpoints of edge: base probabilities on edges
    :param Q: dataframe indexed by two endpoints of edge: product probabilities on edges
    :param F: selected set of features
    :param E: edges that require update
    :param K: number of required features
    :param P: final probabilities on edges (updated only Ef)
    :return:
    """
    changed = dict() # changed edges and its previous probabilities
    for e in E:
        changed[e] = float(P.loc[e]) # remember what edges changed
        hF = len(set(F).intersection(G.node[e[1]]['Fu']))/len(G.node[e[1]]['Fu']) # function h(F)
        q = float(Q.loc[e])
        b = float(B.loc[e])
        P.loc[e] = min(hF*q + b, 1) # final probabilities p = h(F)*q + b
    return changed

def decrease_probabilities(changed, P):
    """
    Decrease probabilities of changed edges.
    :param changed: edge (u,v) -> probability
    :param P: dataframe with probabilities on edges
    :return:
    """
    for e in changed:
        P.loc[e] = changed[e]

def calculate_MC_spread(G, S, P, I):
    """
    Returns influence spread in IC model from S with features F using I Monte-Carlo simulations.
    :param G: networkx graph
    :param S: list of seed set
    :param P: dataframe of probabilities
    :param I: integer of number of MC iterations
    :return: influence spread
    """
    spread = 0.
    for _ in range(I):
        # print 'I:', _,
        activated = dict(zip(G.nodes(), [False]*len(G)))
        for node in S:
            activated[node] = True
        T = [node for node in S]
        i = 0
        while i < len(T):
            v = T[i]
            for neighbor in G[v]:
                if not activated[neighbor]:
                    prob = float(P.loc[v, neighbor])
                    if random.random() < prob:
                        activated[neighbor] = True
                        T.append(neighbor)
            i += 1
        # print len(T)
        spread += len(T)
    return spread/I

def greedy(G, B, Q, Ef, S, Phi, K, I):
    """
    Return best features to PIMUS problem using greedy algorithm.
    :param G: networkx graph
    :param B: dataframe of base probabilities
    :param Q: dataframe of product probabilities
    :param Ef: dictionary feature -> edges
    :param S: list of seed set
    :param Phi: set of all features
    :param K: integer of number of required features
    :param I: integer number of Monte-Carlo simulations
    :return: F: list of K best features
    """
    P = B.copy()
    F = []
    influence = dict()
    while len(F) < K:
        max_spread = -1
        print 'len(F):', len(F)
        for f in Phi.difference(F):
            changed = increase_probabilities(G, B, Q, F + [f], Ef[f], P)
            spread = calculate_MC_spread(G, S, P, I)
            if spread > max_spread:
                max_spread = spread
                max_feature = f
            decrease_probabilities(changed, P)
        print 'Selected', max_feature
        F.append(max_feature)
        influence[len(F)] = max_spread
        increase_probabilities(G, B, Q, F + [max_feature], Ef[max_feature], P)
    return F, influence

def explore(G, P, S, theta):
    """
    Creates in-arborescences for nodes reachable from S.
    :param G: networkx graph
    :param P: dataframe of edge probabilities
    :param S: list seed set
    :param theta: float parameter controlling size of arborescence
    :return:
    """

    Ain = dict()
    for v in S:
        MIPs = {v: []} # shortest paths of edges to nodes from v
        crossing_edges = set([out_edge for out_edge in G.out_edges([v]) if out_edge[1] not in S + [v]])
        edge_weights = dict()
        dist = {v: 0} # shortest paths from the root v

        while crossing_edges:
            # Dijkstra's greedy criteria
            min_dist = float("Inf")
            sorted_crossing_edges = sorted(crossing_edges) # to break ties consistently
            for edge in sorted_crossing_edges:
                if edge not in edge_weights:
                    edge_weights[edge] = -math.log(float(P.loc[edge]))
                edge_weight = edge_weights[edge]
                if dist[edge[0]] + edge_weight < min_dist:
                    min_dist = dist[edge[0]] + edge_weight
                    min_edge = edge
            # check stopping criteria
            if min_dist < -math.log(theta):
                dist[min_edge[1]] = min_dist
                MIPs[min_edge[1]] = MIPs[min_edge[0]] + [min_edge]
                # update crossing edges
                crossing_edges.difference_update(G.in_edges(min_edge[1]))
                crossing_edges.update([out_edge for out_edge in G.out_edges(min_edge[1])
                                       if (out_edge[1] not in MIPs) and (out_edge[1] not in S)])
            else:
                break
        for u in MIPs:
            if u not in S:
                if u in Ain:
                    Ain[u].add_edges_from(MIPs[u])
                else:
                    Ain[u] = nx.DiGraph()
                    Ain[u].add_edges_from(MIPs[u])
    return Ain

def calculate_ap(u, Ain_v, S, P):
    """
    Calculate activation probability of u in in-arborescence of v.
    :param u: node in networkx graph
    :param Ain_v: networkx graph
    :param S: list of seed set
    :param P: dataframe of edge probabilities
    :return: float of activation probability
    """
    if u in S:
        return 1
    elif not Ain_v.in_edges(u):
        return 0
    else:
        prod = 1
        for e in Ain_v.in_edges(u):
            w = e[0]
            ap_w = calculate_ap(w, Ain_v, S, P)
            prod *= (1 - ap_w*float(P.loc[e]))
        return 1 - prod

def update(Ain, S, P):
    """
    Returns influence spread in IC model from S using activation probabilities in in-arborescences.
    :param Ain: dictionary of node -> networkx in-arborescence
    :param S: list of seed set
    :param P: dataframe of edge probabilities
    :return:
    """
    return sum([calculate_ap(u, Ain[u], S, P) for u in Ain])

def get_pi(G, Ain, S):
    """
    Get participating edges.
    :param G: networkx graph
    :param Ain: in-arborescences
    :param S: Seed set
    :return:
    """
    Pi_nodes = set(Ain.keys() + S)
    Pi = set()
    for u in Pi_nodes:
        Pi.update(G.in_edges(u))
        Pi.update(G.out_edges(u))
    return Pi

def explore_update(G, B, Q, S, K, Ef, theta):
    """
    Explore-Update algorithm.
    :param G: networkx graph
    :param B: dataframe base probabilities
    :param Q: dataframe product probabilities
    :param S: list of seed set
    :param K: integer of number of required features
    :param Ef: dictionary of feature to edges
    :param theta: float threshold parameter
    :return F: list of selected features
    """
    P = B.copy() # initialize edge probabilities

    Ain = explore(G, P, S, theta)
    Pi = get_pi(G, Ain, S)
    print 'Ain:', len(Ain)

    F = []
    Phi = set(Ef.keys())

    count = 0
    while len(F) < K:
        print '***************len(F): {}'.format(len(F))
        max_feature = None
        max_spread = -1
        for f in Phi.difference(F):
            e_intersection = Pi.intersection(Ef[f])
            # print 'intersection', len(e_intersection)
            if e_intersection:
                changed = increase_probabilities(G, B, Q, F + [f], Ef[f], P)
                Ain = explore(G, P, S, theta)
                spread = update(Ain, S, P)
                if spread > max_spread:
                    max_spread = spread
                    max_feature = f
                decrease_probabilities(changed, P)
            else:
                print 'Missing feature', f
                count += 1
        if max_feature:
            F.append(max_feature)
            increase_probabilities(G, B, Q, F, Ef[max_feature], P)
            Ain = explore(G, P, S, theta)
            Pi = get_pi(G, Ain, S)
            print 'Ain:', len(Ain)
        else:
            raise ValueError, 'Not found max_feature. F: {}'.format(F)
    print 'Total number of omissions', count
    return F

def calculate_spread(G, B, Q, S, F, Ef, I):
    """
    Calculate spread for given feature set F.
    :param G: networkx graph
    :param B: dataframe base probabilities
    :param Q: dataframe product probabilities
    :param S: list of seed set
    :param F: list of selected features
    :param Ef: dictionary of feature to edges
    :param I: integer number of MC calculations
    :return: float average number of influenced nodes
    """
    P = B.copy()
    E = []
    for f in F:
        E.extend(Ef[f])
    increase_probabilities(G, B, Q, F, E, P)

    return calculate_MC_spread(G, S, P, I)

def read_groups(filename):
    """
    Reads groups' memberships.
    :param filename: string each line is group and its members
    :return: dictionary group to members
    """
    groups = dict()
    with open(filename) as f:
        for line in f:
            d = line.split()
            members = map(int, d[1:])
            groups[d[0]] = members
    return groups

def number_of_combinations(n, k):
    d = {1: 1}
    for i in range(2, n+1):
        d[i] = d[i-1]*i
    return float(d[n])/(d[k]*d[n-k])

def brute_force(G, B, Q, S, K, Ef, I):
    Phi = set(Ef.keys())
    combs = combinations(Phi, K)
    max_spread = -1
    max_F = []
    print 'Total', number_of_combinations(len(Phi), K)
    for i, f_set in enumerate(combs):
        print i, f_set,
        start = time.time()
        spread = calculate_spread(G, B, Q, S, f_set, Ef, I)
        print spread, time.time() - start
        if spread > max_spread:
            max_F = f_set
            max_spread = spread
    return max_F, max_spread

class CalcSpread(object):
    def __init__(self, G, B, Q, S, Ef, I):
        self.G = G
        self.B = B
        self.Q = Q
        self.S = S
        self.Ef = Ef
        self.I = I
    def c_spread(self, F):
        return calculate_spread(G, B, Q, S, F, Ef, I)

def calc_spread((idx, F)):
    start = time.time()
    spread = C.c_spread(F)
    finish = time.time() - start
    print idx, F, spread, finish
    return F, spread

def top_edges(Ef, K):
    return map(lambda (k, v): k, sorted(Ef.items(), key= lambda (k, v): len(v), reverse=True)[:K])

def top_nodes(Nf, K):
    return map(lambda (k, v): k, Counter(chain.from_iterable(Nf.values())).most_common(K))

if __name__ == "__main__":

    model = "mv"
    print model

    G = read_graph('datasets/wv.txt')
    Ef, Nf = add_graph_attributes(G, 'datasets/wikivote_mem.txt')
    Phi = set(Ef.keys())

    B = read_probabilities('datasets/wikivote_{}.txt'.format(model))
    Q = read_probabilities('datasets/wikivote_{}.txt'.format(model))

    print 'Phi: {}'.format(len(Ef))

    groups = read_groups('datasets/wikivote_com.txt')
    #
    # I = 1
    # S = groups['9']
    # print len(S), S
    #
    # for u in S:
    #     edges = G.in_edges(u)
    #     G.remove_edges_from(edges)
    #



    # # greedy spread
    # filename1 = "datasets/greedy/gnutella_greedy_range_{}.txt".format(model)
    # filename2 = "datasets/greedy/gnutella_greedy_selected_{}.txt".format(model)
    #
    # with open(filename2) as f:
    #     F = f.readlines()[0].split()
    #
    # print len(F), F
    #
    # with open(filename1, 'w') as f1:
    #     for K in range(5, 15, 5):
    #         spread = calculate_spread(G, B, Q, S, F[:K], Ef, I)
    #         f1.write("{}\n".format(spread))



    # experiment 3
    # selected_groups = ['10', '26', '133', '75', '61', '135', '72', '73', '25']
    #
    # filename1 = "datasets/experiment3/gnutella_results_greedy_{}.txt".format(model)
    # filename2 = "datasets/experiment3/gnutella_time_greedy_{}.txt".format(model)
    #
    # I = 500
    # K = 50
    # for gr in selected_groups:
    #     S = groups[gr]
    #
    #     for u in S:
    #         edges = G.in_edges(u)
    #         G.remove_edges_from(edges)

        # f1.write("{} ".format(len(S)))
        # f2.write("{} ".format(len(S)))

        # # explore-update
        # start = time.time()
        # F = explore_update(G, B, Q, S, K, Ef, 1./40)
        # finish = time.time() - start
        # f2.write("{} ".format(finish))
        # print 'EU', sorted(F), time.time() - start
        # start = time.time()
        # eu_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        # print 'EU spread:', eu_spread, time.time() - start
        # f1.write("{} ".format(eu_spread))
        #
        # # top edges
        # start = time.time()
        # F = map(lambda (k, v): k, sorted(Ef.items(), key= lambda (k, v): len(v), reverse=True)[:K])
        # finish = time.time() - start
        # f2.write("{}\n".format(finish))
        # print 'Top edges:', sorted(F), time.time() - start
        # tope_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        # print 'Top edges spread:', tope_spread
        # f1.write("{}\n".format(tope_spread))

        # greedy
        # top edges
        # start = time.time()
        # F = greedy(G, B, Q, Ef, S, Phi, K, I)
        # finish = time.time() - start
        # # f2.write("{}\n".format(finish))
        # print 'Greedy:', sorted(F), time.time() - start
        # greedy_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        # print 'Top edges spread:', greedy_spread
        # # f1.write("{}\n".format(greedy_spread))
        #
        # with open(filename1, 'a') as f1, open(filename2, 'a') as f2:
        #     f1.write("{} {}\n".format(len(S), greedy_spread))
        #     f2.write("{} {}\n".format(len(S), finish))


    # experiment 4

    # S = groups['25']
    # for u in S:
    #     edges = G.in_edges(u)
    #     G.remove_edges_from(edges)
    #
    # filename1 = "datasets/experiment4/gnutella_theta_spread.txt"
    # filename2 = "datasets/experiment4/gnutella_theta_time.txt"
    #
    # I = 500
    # K = 50
    # for theta in [1./10, 1./20, 1./40, 1./80, 1./160, 1./320, 1./640, 1./1280]:
    #     print 'theta=', theta
    #     start = time.time()
    #     F = explore_update(G, B, Q, S, K, Ef, theta)
    #     finish = time.time() - start
    #     with open(filename2, 'a') as f2:
    #         f2.write("{}\n".format(finish))
    #     eu_spread = calculate_spread(G, B, Q, S, F, Ef, I)
    #     with open(filename1, 'a') as f1:
    #         f1.write("{}\n".format(eu_spread))

    # experiment 5
    filename1 = "datasets/experiment5/wikivote_time_{}.txt".format(model)
    filename2 = "datasets/experiment5/wikivote_results_{}.txt".format(model)
    filename3 = "datasets/experiment5/wikivote_results_greedy_{}.txt".format(model)

    S = groups['9'] #todo change the number of a group
    print '9:', len(S)
    for u in S:
        edges = G.in_edges(u)
        G.remove_edges_from(edges)

    K = 50
    I = 100
    start = time.time()
    F, greedy_influence = greedy(G, B, Q, Ef, S, Phi, K, I)
    finish = time.time()
    print F, finish - start
    with open(filename1, 'a') as f1:
        f1.write("{} ".format(finish - start))
    with open(filename3, 'a') as f3:
        for size in range(5, 51, 5):
            f3.write("{}\n".format(greedy_influence[size]))

    start = time.time()
    eu_selected = explore_update(G, B, Q, S, K, Ef, 1./40)
    finish = time.time()
    with open(filename1, 'a') as f1:
        f1.write("{} ".format(finish - start))

    for K in range(5, 51, 5):
        print 'K:', K
        F = eu_selected[:K]
        eu_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        with open(filename2, 'a') as f2:
            f2.write("{} ".format(eu_spread))

        F = top_edges(Ef, K)
        tope_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        with open(filename2, 'a') as f2:
            f2.write("{} ".format(tope_spread))

        F = top_nodes(Nf, K)
        topn_spread = calculate_spread(G, B, Q, S, F, Ef, I)
        with open(filename2, 'a') as f2:
            f2.write("{}\n".format(topn_spread))


    # K = 10
    #
    # L = 500
    # dct = dict()
    # with open("gnutella_mv_opt2_top.txt") as f:
    #     for ix, line in enumerate(f):
    #         d = line.split()
    #         dct[tuple(sorted(d[1:]))] = (ix, float(d[0]))
    #         if ix == L:
    #             break

    # C = CalcSpread(G, B, Q, S, Ef, I)
    # pool = mp.Pool(4)
    # start = time.time()
    # map_lst = pool.map(calc_spread, enumerate(dct))
    # finish = time.time() - start
    # print 'Total time to brute-force', finish
    # sorted_maps = sorted(map_lst, key = lambda (_, s): s, reverse=True)
    # print sorted_maps[:10]
    # with open('gnutella_mv_opt2_top.txt', 'w') as f1:
    #     for F, sp in sorted_maps:
    #         f1.write("{} ".format(sp))
    #         for f in F:
    #             f1.write("{} ".format(f))
    #         f1.write("\n")

    # dct3 = dict()
    # for F in dct:
    #     print F,
    #     start = time.time()
    #     spread = calculate_spread(G, B, Q, S, F, Ef, I)
    #     print time.time() - start
    #     dct3[F] = spread
    #
    # sorted_top = sorted(dct3.items(), key = lambda (k,v): v, reverse=True)
    # with open("gnutella_mv_opt2_top.txt", "w") as f:
    #     for F, sp in sorted_top:
    #         f.write("{} {}\n".format(sp, " ".join(F)))

    # dct2 = dict()
    # with open("gnutella_mv_others.txt") as f:
    #     for line in f:
    #         d = line.split()
    #         dct2[tuple(sorted(d[1:]))] = float(d[0])
    #
    # for k in dct2:
    #     print k, dct[k]


    # C = CalcSpread(G, B, Q, S, Ef, I)
    # combs = combinations(Phi, K)
    # pool = mp.Pool(4)
    # start = time.time()
    # map_lst = pool.map(calc_spread, enumerate(combs))
    # finish = time.time() - start
    # print 'Total time to brute-force', finish
    # sorted_maps = sorted(map_lst, key = lambda (_, s): s, reverse=True)
    # print sorted_maps[:10]
    # with open('gnutella_mv_opt2.txt', 'w') as f1:
    #     for F, sp in sorted_maps:
    #         f1.write("{} ".format(sp))
    #         for f in F:
    #             f1.write("{} ".format(f))
    #         f1.write("\n")

    # with open('datasets/gnutella_eu_selected2_wc.txt') as f:
    #     selected_eu = f.readlines()[0].split()
    #
    # with open('datasets/gnutella_greedy_selected2_wc.txt') as f:
    #     selected_greedy = f.readlines()[0].split()
    #

    # filename1 = "datasets/greedy/gnutella_greedy_selected_{}.txt".format(model)
    # filename2 = "datasets/greedy/gnutella_greedy_time_{}.txt".format(model)
    # filename3 = "datasets/greedy/gnutella_greedy_results_{}.txt".format(model)
    #
    # with open(filename1, 'w') as f1, open(filename2, 'w') as f2:
    #     K = 15
    #     start = time.time()
    #     F = greedy(G, B, Q, Ef, S, Phi, K, I)
    #     finish = time.time() - start
    #     print 'Greedy:', F, finish
    #
    #     f1.write(" ".join(F))
    #     f2.write("{}".format(finish))
    #
    # with open(filename3, 'w') as f1:
    #     for K in range(5, 15, 5):
    #         print K,
    #         greedy_spread = calculate_spread(G, B, Q, S, F[:K], Ef, I)
    #         print greedy_spread
    #         f1.write("{}\n".format(greedy_spread))

    # with open("gnutella_mv_.txt", 'w') as f1:
    #     for K in [50]:
    #         print 'K', K
    #         # greedy algorithm
    #         start = time.time()
    #         F = greedy(G, B, Q, Ef, S, Phi, K, I)
    #         # F = selected_greedy[:K]
    #         finish = time.time() - start
    #         # f1.write("{}".format(finish))
    #         print 'Greedy:', F, finish
            # greedy_spread = calculate_spread(G, B, Q, S, F, Ef, I)
            # print 'Greedy spread:', greedy_spread
            # f1.write("{} ".format(greedy_spread))
            # f1.write(" ".join(sorted(F)))
            # f1.write("\n")

            # with open('datasets/gnutella_greedy_selected2_wc.txt', 'w') as f:
            #     f.write(" ".join(F))

            # # EU algorithm
            # start = time.time()
            # F2 = explore_update(G, B, Q, S, K, Ef, 1./40)
            # # F2 = selected_eu[:K]
            # finish = time.time() - start
            # # f1.write("{} ".format(finish))
            # print 'EU', sorted(F2), time.time() - start
            # start = time.time()
            # eu_spread = calculate_spread(G, B, Q, S, F2, Ef, I)
            # print 'EU spread:', eu_spread, time.time() - start
            # f1.write("{} ".format(eu_spread))
            # f1.write(" ".join(sorted(F2)))
            # f1.write("\n")
            #
            # # with open('datasets/gnutella_eu_selected2_wc.txt', 'w') as f:
            # #     f.write(" ".join(F2))
            #
            # # top edges
            # start = time.time()
            # F = map(lambda (k, v): k, sorted(Ef.items(), key= lambda (k, v): len(v), reverse=True)[:K])
            # finish = time.time() - start
            # # f2.write("{} ".format(finish))
            # print 'Top edges:', sorted(F), time.time() - start
            # tope_spread = calculate_spread(G, B, Q, S, F, Ef, I)
            # print 'Top edges spread:', tope_spread
            # f1.write("{} ".format(tope_spread))
            # f1.write(" ".join(sorted(F)))
            # f1.write("\n")
            # #
            # # # top nodes
            # start = time.time()
            # F = map(lambda (k, v): k, Counter(chain.from_iterable(Nf.values())).most_common(K))
            # finish = time.time() - start
            # # f2.write("{}\n".format(finish))
            # print 'Top nodes:', sorted(F), time.time() - start
            # topn_spread = calculate_spread(G, B, Q, S, F, Ef, I)
            # print 'Top nodes spread:', topn_spread
            # f1.write("{}".format(topn_spread))
            # f1.write(" ".join(sorted(F)))
            # f1.write("\n")
    # print calculate_spread(G, S, B, Q, F, Ef, 100)
    # print calculate_spread(G, S, B, Q, F2, Ef, 100)




    # theta = 1./2000
    # S = [0, 5, 10]
    # start = time.time()
    # Ain = explore(G, P, S, theta)
    # print time.time() - start
    # print len(Ain)
    #
    # print update(Ain, S, P)
    # print calculate_MC_spread(G, S, P, 10)

    console = []