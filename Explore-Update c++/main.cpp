#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <unordered_set>
#include <time.h>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/graph/topological_sort.hpp>
#include <unordered_map>
#include <ctime>

using namespace std;

struct vertex_info {int label;};

typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::bidirectionalS> DiGraph;
typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::bidirectionalS, vertex_info> SubGraph;
typedef boost::graph_traits<SubGraph>::vertex_descriptor vertex_t;
typedef boost::graph_traits<SubGraph>::edge_descriptor edge_t;
typedef boost::graph_traits<DiGraph>::vertex_iterator vertex_iter;
typedef boost::graph_traits<DiGraph>::edge_iterator edge_iter;
typedef boost::graph_traits<DiGraph>::out_edge_iterator out_edge_iter;
typedef boost::graph_traits<DiGraph>::in_edge_iterator in_edge_iter;
//typedef map<pair<int, int>, double> edge_prob;
typedef map<edge_t, double> prob_e;

void print_vertices(DiGraph G) {
    pair<vertex_iter, vertex_iter> vp;
    for (vp = boost::vertices(G); vp.first != vp.second; ++vp.first)
        cout << *vp.first << " " << *vp.second << endl;
    cout << endl;
}

void print_edges(DiGraph G) {
    edge_iter ei, edge_end;
    for (boost::tie(ei, edge_end) = edges(G); ei != edge_end; ++ei) {
        cout << source(*ei, G) << " " << target(*ei, G) << endl;
    }
}

void print_degree(DiGraph G) {
    vertex_iter vi, v_end;
    int out_d, in_d, count=0;
    for (boost::tie(vi, v_end) = boost::vertices(G); vi != v_end; ++vi) {
        in_d = boost::in_degree(*vi, G);
        out_d = boost::out_degree(*vi, G);
        cout << *vi << " " << out_d << " " << in_d << endl;
    }
}

void print_node_edges(DiGraph G) {
    out_edge_iter ei, e_end;
    in_edge_iter qi, q_end;
    vertex_iter vi, v_end;
    for (boost::tie(vi, v_end) = boost::vertices(G); vi != v_end; ++vi) {
        cout << *vi << "--->";
        for (boost::tie(ei, e_end) = out_edges(*vi, G); ei!=e_end; ++ei) {
            cout << target(*ei, G) << " ";
        }
        cout << endl;
        cout << *vi << "<---";
        for (boost::tie(qi, q_end) = in_edges(*vi, G); qi!=q_end; ++qi) {
            cout << source(*qi, G) << " ";
        }
        cout << endl;
        cout << endl;
    }
}

void print_size(DiGraph G) {
    cout << num_vertices(G) << endl;
    cout << num_edges(G) << endl;
}

DiGraph read_graph(string graph_filename) {
    cout << graph_filename << endl;
    ifstream infile(graph_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }

    unordered_map<int, int> unordered_mapped;
    int u, v;
    int node_count=0;
    pair<DiGraph::edge_descriptor, bool> edge_insertion;
    DiGraph G;

    while (infile >> u >> v) {
        if (unordered_mapped.find(u) == unordered_mapped.end()) {
            unordered_mapped[u] = node_count;
            node_count++;
        }
        if (unordered_mapped.find(v) == unordered_mapped.end()) {
            unordered_mapped[v] = node_count;
            node_count++;
        }
        edge_insertion=boost::add_edge(unordered_mapped[u], unordered_mapped[v], G);
        if (!edge_insertion.second) {
            std::cout << "Unable to insert edge\n";
        }
    }
    return G;
}

void read_features(string feature_filename, DiGraph G, unordered_map<int, vector<int> > &Nf, unordered_map<int, vector<edge_t> > &Ef) {

    string line;
    vector<string> line_splitted;
    int u, f;
    in_edge_iter ei, e_end;

    ifstream infile(feature_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    while(getline(infile, line)) {
        boost::split(line_splitted, line, boost::is_any_of(" "));
        u = stoi(line_splitted[0]);
        vector<int> u_features;
        for (int i=1; i < line_splitted.size(); ++i) {
            f = stoi(line_splitted[i]);
            u_features.push_back(f);
        }
        for (auto & feat: u_features) {
            for (boost::tie(ei, e_end) = in_edges(u, G); ei!=e_end; ++ei) {
                Ef[feat].push_back(*ei);
            }
        }
        Nf[u] = u_features;
    }
}

void read_probabilities(string prob_filename, prob_e &P, DiGraph G) {
    ifstream infile(prob_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    int u, v;
    double p;
    while (infile >> u >> v >> p) {
        P[boost::edge(u, v, G).first] = p;
    }
}


void read_groups(string group_filename, unordered_map<int, unordered_set<int> > &groups) {
    ifstream infile(group_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    string line;
    vector<string> line_splitted;

    while (getline(infile, line)) {
        boost::split(line_splitted, line, boost::is_any_of(" "));
        unordered_set<int> nodes;
        for (int i = 1; i < line_splitted.size(); ++i) {
            nodes.insert(stoi(line_splitted[i]));
        }
        groups[stoi(line_splitted[0])] = nodes;
    }
}

prob_e increase_probabilities(DiGraph G, prob_e B, prob_e Q, unordered_map<int, vector<int> > Nf, vector<int> F,
                                 vector<edge_t> E, prob_e &P) {
    prob_e changed;
    double q,b,h;
    int t;
    double intersect;
    vector<int> F_target;
    for (auto &edge: E) {
        changed[edge] = P[edge];
        q = Q[edge]; b = B[edge];
//        find intersection
//        solution found here: http://stackoverflow.com/a/24337598/2069858
        t = target(edge, G);
        F_target = Nf[t];
        sort(F_target.begin(), F_target.end());
        sort(F.begin(), F.end());
        unordered_set<int> s(F_target.begin(), F_target.end());
        intersect = count_if(F.begin(), F.end(), [&](int k) {return s.find(k) != s.end();});
        h = intersect/F_target.size();
        P[edge] = h*q + b;
    }
    return changed;
}

void decrease_probabilities(prob_e changed, prob_e &P) {
    for (auto &item: changed) {
        P[item.first] = item.second;
    }
}

double calculate_spread (DiGraph G, prob_e B, prob_e Q, unordered_map<int, vector<int> > Nf, set<int> S,
                        vector<int> F, unordered_map<int, vector<edge_t> > Ef, int I) {

    prob_e Prob;
    Prob.insert(B.begin(), B.end());

    vector<edge_t> E;
    for (int i =0; i<F.size(); ++i) {
        for (int j=0; j < Ef[F[i]].size(); ++j) {
            E.push_back(Ef[F[i]][j]);
        }
    }

    increase_probabilities(G, B, Q, Nf, F, E, Prob);

    double spread=0;
    pair<vertex_iter, vertex_iter> vp;
    unordered_map<int, bool> activated;
    vector<int> T;
    int u, v;
    double p;
    out_edge_iter ei, e_end;
    for (int it=0; it < I; ++it) {
        for (vp = boost::vertices(G); vp.first != vp.second; ++vp.first) {
            u = (int)*vp.first;
            activated[u] = false;
        }
        for (auto &node: S) {
            activated[node] = false;
            T.push_back(node);
        }
        int count = 0;
        while (count < T.size()) {
            u = T[count];
            for (boost::tie(ei, e_end) = out_edges(u, G); ei!=e_end; ++ei) {
                v = target(*ei, G);
                if (not activated[v]) {
                    p = Prob[boost::edge(u,v,G).first];
                    double r = ((double) rand() / (RAND_MAX));
                    if (r < p) {
                        activated[v] = true;
                        T.push_back(v);
                    }
                }
            }
            ++count;
        }
        spread += T.size();
        T.clear();
    }
    return spread/I;
}

pair<vector<int>, unordered_map<int, double> >  greedy(DiGraph G, prob_e B, prob_e Q, set<int> S, unordered_map<int,
        vector<int> > Nf, unordered_map<int, vector<edge_t> > Ef, vector<int> Phi, int K, int I) {

    vector<int> F;
    prob_e P;
    unordered_map<int, bool> selected;
    prob_e changed;
    double spread, max_spread;
    int max_feature;
    unordered_map<int, double> influence;

    P.insert(B.begin(), B.end());

    while (F.size() < K) {
        max_spread = -1;
        printf("it = %i; ", (int)F.size() + 1);
        fflush(stdout);
        for (auto &f: Phi) {
            if (not selected[f]) {
                F.push_back(f);
                changed = increase_probabilities(G, B, Q, Nf, F, Ef[f], P);
                spread = calculate_spread(G, B, Q, Nf, S, F, Ef, I);
                if (spread > max_spread) {
                    max_spread = spread;
                    max_feature = f;
                }
                decrease_probabilities(changed, P);
                F.pop_back();
            }
        }
        F.push_back(max_feature);
        selected[max_feature] = true;
        printf("f = %i; spread = %.2f\n", max_feature, max_spread);
        increase_probabilities(G, B, Q, Nf, F, Ef[max_feature], P);
        influence[F.size()] = max_spread;
    }
    return make_pair(F, influence);
}

bool less_edge(DiGraph G, edge_t edge1, edge_t edge2) {
    int u1 = source(edge1, G), v1 = target(edge1, G),
            u2 = source(edge2, G), v2 = target(edge2, G);
    if (u1 < u2 or (u1 == u2 and v1 < v2))
        return true;
    return false;

}

unordered_map<int, set<edge_t> > explore(DiGraph G, prob_e P, unordered_set<int> S, double theta) {

    double max_num = numeric_limits<double>::max();
    double min_dist;
    bool b;
    edge_t min_edge, mip_edge;
    pair<DiGraph::edge_descriptor, bool> edge_insertion;
    int V = num_vertices(G);
    map<edge_t, double> edge_weights;
    out_edge_iter ei, e_end;
    in_edge_iter qi, q_end;
    unordered_map<int, double> dist;
    set<edge_t> crossing_edges;
    unordered_map<int, vector<edge_t> > MIPs;
    unordered_map<int, set<edge_t> > Ain_edges;

    cout << "nodes:" << endl;
    for (auto &v: S) {
        cout << v << endl;
        MIPs[v] = {};
        dist[v] = 0;
        for (boost::tie(ei, e_end) = out_edges(v, G); ei!=e_end; ++ei) {
            crossing_edges.insert(*ei);
        }

        while (true) {
            if (crossing_edges.size() == 0)
                break;

            min_dist = max_num;
            boost::tie(min_edge, b) = boost::edge(V+1, V+1, G);

            for (auto &edge: crossing_edges) {
                if (edge_weights.find(edge) == edge_weights.end()) {
                    edge_weights[edge] = -log(P[edge]);
                }

                if (edge_weights[edge] + dist[source(edge, G)] < min_dist or
                        (edge_weights[edge] + dist[source(edge, G)] == min_dist and less_edge(G, edge, min_edge))) {
                    min_dist = edge_weights[edge] + dist[source(edge, G)];
                    min_edge = edge;
                }
            }
            if (min_dist <= -log(theta)) {
                dist[target(min_edge, G)] = min_dist;
                MIPs[target(min_edge, G)] = MIPs[source(min_edge, G)];
                MIPs[target(min_edge, G)].push_back(min_edge);
                for (auto &edge: MIPs[target(min_edge, G)]) {
                    Ain_edges[target(min_edge, G)].insert(edge);
                }

                for (boost::tie(qi, q_end) = in_edges(target(min_edge, G), G); qi!=q_end; ++qi) {
                    crossing_edges.erase(*qi);
                }
                for (boost::tie(ei, e_end) = out_edges(target(min_edge, G), G); ei!=e_end; ++ei) {
                    int end2 = target(*ei, G);
                    if (MIPs.find(end2) == MIPs.end()) {
                        crossing_edges.insert(*ei);
                    }
                }
            }
            else
                break;
        }
        dist.clear();
        crossing_edges.clear();
        MIPs.clear();
    }
    return Ain_edges;
}

SubGraph make_subgraph(DiGraph G, set<edge_t> Ain_edges_v, int root) {
    SubGraph Ain_v;
    int u, v, count=0;
    unordered_map<int, int> unordered_mapped;
    edge_t e; bool b;
    vertex_t vertex;

    unordered_mapped[root] = count;
    vertex = boost::add_vertex(Ain_v);
    Ain_v[vertex].label = root;
    count++;
    for (auto &edge: Ain_edges_v) {
        u = source(edge, G);
        v = target(edge, G);
        if (unordered_mapped.find(u) == unordered_mapped.end()) {
            unordered_mapped[u] = count;
            vertex = boost::add_vertex(Ain_v);
            Ain_v[vertex].label = u;
            count++;
        }
        if (unordered_mapped.find(v) == unordered_mapped.end()) {
            unordered_mapped[v] = count;
            vertex = boost::add_vertex(Ain_v);
            Ain_v[vertex].label = v;
            count++;
        }
        boost::tie(e, b) = boost::add_edge(unordered_mapped[u], unordered_mapped[v], Ain_v);
        if (not b)
            cout << "Unable to insert an edge in Ain_v" << endl;
    }
    return Ain_v;
}

double calculate_ap(DiGraph G, vertex_t u, SubGraph Ain_v, unordered_set<int> S, prob_e P) {
    if (S.find(Ain_v[u].label) != S.end())
        return 1;
    else {
        double prod = 1, ap_node, p;
        in_edge_iter qi, q_end;
        vertex_t node;
        edge_t e;
        bool b;
        for (boost::tie(qi, q_end)=in_edges(u, Ain_v); qi!=q_end; ++qi) {
            node = source(*qi, Ain_v);
            ap_node = calculate_ap(G, node, Ain_v, S, P);
//            p = P[make_pair(Ain_v[node].label, Ain_v[u].label)];
            boost::tie(e, b) = boost::edge(Ain_v[node].label, Ain_v[u].label, G);
            p = P[e];
            prod *= (1 - ap_node*p);
        }
        return 1 - prod;
    }
}

double calculate_ap2(DiGraph G, SubGraph Ain_v, unordered_set<int> S, prob_e P) {
    vector<vertex_t> topology;
    unordered_map<vertex_t, double> ap;
    double prod;
    in_edge_iter qi, q_end;

    topological_sort(Ain_v, back_inserter(topology));

    clock_t start = clock();
    for (vector<vertex_t>::reverse_iterator ii=topology.rbegin(); ii!=topology.rend(); ++ii) {
        if (S.find(Ain_v[*ii].label) != S.end()) {
            ap[*ii] = 1;
        }
        else {
            prod = 1;
            for (boost::tie(qi, q_end)=in_edges(*ii, Ain_v); qi!=q_end; ++qi) {
                pair<edge_t, bool> edge_G = boost::edge(Ain_v[source(*qi, Ain_v)].label, Ain_v[*ii].label, G);
                prod *= (1 - ap[source(*qi, Ain_v)]*P[edge_G.first]);
            }
            ap[*ii] = 1 - prod;
        }
    }
    return 1 - prod;
}

// TODO redo. Refactor Ain_edges
double update(DiGraph G, unordered_map<int, set<edge_t> > Ain_edges, unordered_set<int> S, prob_e P) {
    double total = 0, path_prob;
    unordered_set<int> mip;
    bool pathed;
    for (auto &item: Ain_edges) {
        pathed = true;
        path_prob = 1;
//      optimization for simple paths
        set<edge_t> edges = item.second;
        for (const auto &e: edges) {
            if (mip.find(target(e, G)) != mip.end()) {
                pathed = false;
                break;
            }
            else {
                mip.insert(target(e, G));
                path_prob *= P[e];
            }
        }
        if (pathed) {
            total += path_prob;
        }
        else {
            SubGraph Ain_v = make_subgraph(G, Ain_edges[item.first], item.first);
            total += calculate_ap2(G, Ain_v, S, P);
        }
    }
    return total;
}

set<edge_t> get_pi(DiGraph G, unordered_map<int, set<edge_t> > Ain_edges, unordered_set<int> S) {
    set<edge_t> Pi;
    out_edge_iter ei, e_end;
    in_edge_iter qi, q_end;
    set<int> Pi_nodes;

    Pi_nodes.insert(S.begin(), S.end());
    for (auto &item: Ain_edges) {
        Pi_nodes.insert(item.first);
    }

    for (auto &node: Pi_nodes) {
        for (boost::tie(ei, e_end) = out_edges(node, G); ei!=e_end; ++ei) {
            Pi.insert(*ei);
        }
        for (boost::tie(qi, q_end) = in_edges(node, G); qi!=q_end; ++qi) {
            Pi.insert(*qi);
        }
    }
    return Pi;
}

vector<int> explore_update(DiGraph G, prob_e B, prob_e Q, prob_e P, unordered_set<int> S, unordered_map<int,vector<int> > Nf,
                           unordered_map<int, vector<edge_t>> Ef, vector<int> Phi, int K, double theta) {


    vector<int> F;
    unordered_map<int, set<edge_t> > Ain_edges;
    set<edge_t> Pi;
    int max_feature;
    double max_spread, spread;
    unordered_map<int, bool> selected;
    bool intersected;
    prob_e changed;
    int omissions = 0;
    clock_t begin, finish;

    Ain_edges = explore(G, P, S, theta);
    Pi = get_pi(G, Ain_edges, S);

    while (F.size() < K) {
        cout << F.size() << ": ";
        max_feature = -1;
        max_spread = -1;
        for (auto &f: Phi) {
            cout << f << " ";
            fflush(stdout);
            if (not selected[f]) {
                intersected = false;
                for (auto &edge: Ef[f]) {
                    if (Pi.find(edge) != Pi.end()) {
                        intersected = true;
                        break;
                    }
                }
                if (intersected) {
                    F.push_back(f);
                    changed = increase_probabilities(G, B, Q, Nf, F, Ef[f], P);
                    Ain_edges = explore(G, P, S, theta);
                    begin = clock();
                    spread = update(G, Ain_edges, S, P);
                    if (spread > max_spread) {
                        max_spread = spread;
                        max_feature = f;
                    }
                    decrease_probabilities(changed, P);
                    F.pop_back();
                }
                else {
                    ++omissions;
                }
            }
        }
        cout << endl;
        F.push_back(max_feature);
        selected[max_feature] = true;
        increase_probabilities(G, B, Q, Nf, F, Ef[max_feature], P);
    }
    cout << "Total number of omissions: " << omissions << endl;
    return F;
}


int main(int argc, char* argv[]) {
    srand(time(NULL));
    // read parameters from command-line
    if (argc > 1) {
        const string path = argv[1]; // prefix path to directory with necessary files
        const long int V = atoi(argv[2]); // number of nodes
        const int K = atoi(argv[4]);         // number of features
    }

    unordered_map<int, vector<int> > Nf;
    unordered_map<int, vector<edge_t> > Ef;
    prob_e B, Q, P;
    unordered_map<int, unordered_set<int> > groups;
    vector<int> F;
    unordered_set<int> S;
    int I, K;
    unordered_map<int, double> influence;
    double theta;
    in_edge_iter qi, q_end;

    DiGraph G = read_graph("datasets/gnutella.txt");
    read_features("datasets/gnutella_mem.txt", G, Nf, Ef);
    read_probabilities("datasets/gnutella_mv.txt", B, G);
    read_probabilities("datasets/gnutella_mv.txt", Q, G);
    read_probabilities("datasets/gnutella_mv.txt", P, G);
    read_groups("datasets/gnutella_com.txt", groups);

    vector<int> Phi;
    for (auto &item: Ef) {
        Phi.push_back(item.first);
    }

//    setup
    S = groups[2];
    for (auto &node: S) {
        boost::clear_in_edges(node, G);
    }
    I = 100;
    K = 2;
    theta = 1./320;
    cout << "I: " << I << endl;
    cout << "K: " << K << endl;

//    greedy algorithm
//    clock_t begin = clock();
//    boost::tie(F, influence) = greedy(G, B, Q, S, Nf, Ef, Phi, K, I);
//    clock_t finish = clock();
//    printf("Time = %.4f sec.", (double) (finish - begin)/CLOCKS_PER_SEC);
//    cout << " F = ";
//    for (int i = 0; i < F.size(); ++i)
//        cout << F[i] << " ";
//    cout << endl;
//    for (auto &item: influence) {
//        printf("%i %f\n", item.first, item.second);
//    }

    unordered_map<int, set<edge_t> > Ain_edges;
    clock_t begin, finish;
    begin = clock();
    Ain_edges = explore(G, P, S, theta);
    cout << Ain_edges.size() << endl;
    begin = clock();
    double total = update(G, Ain_edges, S, P);
    finish = clock();
    cout << (double) (finish - begin)/(CLOCKS_PER_SEC) << " " << total << endl;

//    cout << "Start explore-update" << endl;
//    clock_t start;
//    start = clock();
//    F = explore_update(G, B, Q, P, S, Nf, Ef, Phi, K, theta);
//    cout << "Time Explore-Update: " << (double) (clock() - start)/(CLOCKS_PER_SEC) << endl;
//    for (auto &f: F)
//        cout << f << " ";
//    cout << endl;

    return 0;
}