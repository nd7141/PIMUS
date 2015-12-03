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
typedef map<pair<int, int>, double> edge_prob;

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

void read_features(string feature_filename, DiGraph G, unordered_map<int, vector<int> > &Nf, unordered_map<int, vector<pair<int, int> > > &Ef) {

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
                Ef[feat].push_back(make_pair(source(*ei, G), target(*ei, G)));
            }
        }
        Nf[u] = u_features;
    }
}

void read_probabilities(string prob_filename, edge_prob &P) {
    ifstream infile(prob_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    int u, v;
    double p;
    while (infile >> u >> v >> p) {
        P[make_pair(u, v)] = p;
    }
}


void read_groups(string group_filename, unordered_map<int, set<int> > &groups) {
    ifstream infile(group_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    string line;
    vector<string> line_splitted;

    while (getline(infile, line)) {
        boost::split(line_splitted, line, boost::is_any_of(" "));
        set<int> nodes;
        for (int i = 1; i < line_splitted.size(); ++i) {
            nodes.insert(stoi(line_splitted[i]));
        }
        groups[stoi(line_splitted[0])] = nodes;
    }
}

edge_prob increase_probabilities(DiGraph G, edge_prob B, edge_prob Q, unordered_map<int, vector<int> > Nf, vector<int> F,
                                 vector<pair<int, int> > E, edge_prob &P) {
    edge_prob changed;
    double q,b,h;
    int target;
    double intersect;
    vector<int> F_target;
    for (auto &edge: E) {
        changed[edge] = P[edge];
        q = Q[edge]; b = B[edge];
//        find intersection
//        solution found here: http://stackoverflow.com/a/24337598/2069858
        target = edge.second;
        F_target = Nf[target];
        sort(F_target.begin(), F_target.end());
        sort(F.begin(), F.end());
        unordered_set<int> s(F_target.begin(), F_target.end());
        intersect = count_if(F.begin(), F.end(), [&](int k) {return s.find(k) != s.end();});
        h = intersect/F_target.size();
        P[edge] = h*q + b;
    }
    return changed;
}

void decrease_probabilities(edge_prob changed, edge_prob &P) {
    for (auto &item: changed) {
        pair<int, int> edge = item.first;
        double p = item.second;
        P[edge] = p;
    }
}

double calculate_spread (DiGraph G, edge_prob B, edge_prob Q, unordered_map<int, vector<int> > Nf, set<int> S,
                        vector<int> F, unordered_map<int, vector<pair<int, int> > > Ef, int I) {

    edge_prob Prob;
    Prob.insert(B.begin(), B.end());

    vector<pair<int, int> > E;
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
                    p = Prob[make_pair(u, v)];
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

pair<vector<int>, unordered_map<int, double> >  greedy(DiGraph G, edge_prob B, edge_prob Q, set<int> S, unordered_map<int,
        vector<int> > Nf, unordered_map<int, vector<pair<int, int> > > Ef, vector<int> Phi, int K, int I) {

    vector<int> F;
    edge_prob P;
    unordered_map<int, bool> selected;
    edge_prob changed;
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

unordered_map<int, set<pair<int, int> > > explore(DiGraph G, edge_prob P, set<int> S, double theta) {

    double max_num = numeric_limits<double>::max();
    double min_dist;
    pair<int, int> min_edge, mip_edge;
    pair<DiGraph::edge_descriptor, bool> edge_insertion;
    int V = num_vertices(G);
    map<pair<int, int>, double> edge_weights;
    out_edge_iter ei, e_end;
    in_edge_iter qi, q_end;
    unordered_map<int, double> dist;
    set<pair<int, int> > crossing_edges;
    unordered_map<int, vector<pair<int, int> > > MIPs;
    unordered_map<int, set<pair<int, int> > > Ain_edges;


    for (auto &v: S) {
        MIPs[v] = {};
        dist[v] = 0;
        for (boost::tie(ei, e_end) = out_edges(v, G); ei!=e_end; ++ei) {
            crossing_edges.insert(make_pair(source(*ei, G), target(*ei, G)));
        }

        while (true) {
            if (crossing_edges.size() == 0)
                break;

            min_dist = max_num;
            min_edge = make_pair(V+1, V+1);

            for (auto &edge: crossing_edges) {
                if (edge_weights.find(edge) == edge_weights.end()) {
                    edge_weights[edge] = -log(P[edge]);
                }
                if (edge_weights[edge] + dist[edge.first] < min_dist or
                        (edge_weights[edge] + dist[edge.first] == min_dist and edge <= min_edge)) {
                    min_dist = edge_weights[edge] + dist[edge.first];
                    min_edge = edge;
                }
            }
            if (min_dist <= -log(theta)) {
                dist[min_edge.second] = min_dist;
                MIPs[min_edge.second] = MIPs[min_edge.first];
                MIPs[min_edge.second].push_back(min_edge);
                for (auto &edge: MIPs[min_edge.second]) {
                    Ain_edges[min_edge.second].insert(edge);
                }

                for (boost::tie(qi, q_end) = in_edges(min_edge.second, G); qi!=q_end; ++qi) {
                    crossing_edges.erase(make_pair(source(*qi, G), target(*qi, G)));
                }
                for (boost::tie(ei, e_end) = out_edges(min_edge.second, G); ei!=e_end; ++ei) {
                    int end2 = target(*ei, G);
                    if (MIPs.find(end2) == MIPs.end()) {
                        crossing_edges.insert(make_pair(min_edge.second, end2));
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

SubGraph make_subgraph(set<pair<int, int> > Ain_edges_v, int root) {
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
        u = edge.first; v = edge.second;
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

double calculate_ap(vertex_t u, SubGraph Ain_v, set<int> S, edge_prob P) {
    if (S.find(Ain_v[u].label) != S.end())
        return 1;
    else {
        double prod = 1, ap_node, p;
        in_edge_iter qi, q_end;
        vertex_t node;
        for (boost::tie(qi, q_end)=in_edges(u, Ain_v); qi!=q_end; ++qi) {
            node = source(*qi, Ain_v);
            ap_node = calculate_ap(node, Ain_v, S, P);
            p = P[make_pair(Ain_v[node].label, Ain_v[u].label)];
            prod *= (1 - ap_node*p);
        }
        return 1 - prod;
    }
}

double calculate_ap2(SubGraph Ain_v, set<int> S, edge_prob P) {
    vector<vertex_t> topology;
    topological_sort(Ain_v, back_inserter(topology));
    unordered_map<vertex_t, double> ap;
    double p, prod=1;
    in_edge_iter qi, q_end;
    vertex_t node;

    for (vector<vertex_t>::reverse_iterator ii=topology.rbegin(); ii!=topology.rend(); ++ii) {
        if (S.find(Ain_v[*ii].label) != S.end())
            ap[*ii] = 1;
        else {
            prod = 1;
            for (boost::tie(qi, q_end)=in_edges(*ii, Ain_v); qi!=q_end; ++qi) {
                node = source(*qi, Ain_v);
                p = P[make_pair(Ain_v[node].label, Ain_v[*ii].label)];
                prod *= (1 - ap[node]*p);
            }
            ap[*ii] = 1 - prod;
        }
    }
    return 1 - prod;
}

double update(unordered_map<int, set<pair<int, int> > > Ain_edges, set<int> S, edge_prob P) {
    double total = 0, tmp=1;
    unordered_map<int, int> in_degrees;
    bool pathed = true;
    clock_t begin, finish;
    double timed = 0;
    for (auto &item: Ain_edges) {
//         micro optimization
        set<pair<int, int> > edges = item.second;
        for (auto &e: edges) {
            ++in_degrees[e.second];
            if (in_degrees[e.second] > 1) {
                pathed = false;
                break;
            }
            tmp *= P[e];
        }
        if (pathed) {
            total += tmp;
        }
        else {
            begin = clock();
            SubGraph Ain_v = make_subgraph(Ain_edges[item.first], item.first);
            timed += (double) (clock() - begin)/(CLOCKS_PER_SEC);
            total += calculate_ap2(Ain_v, S, P);
        }
    }
    cout << "Time spent on making graph is " << timed << endl;
    return total;
}

set<pair<int, int> > get_pi(DiGraph G, unordered_map<int, set<pair<int, int> > > Ain_edges, set<int> S) {
    set<pair<int, int> > Pi;
    out_edge_iter ei, e_end;
    in_edge_iter qi, q_end;
    vertex_iter vi, v_end;
    set<int> Pi_nodes;

    Pi_nodes.insert(S.begin(), S.end());
    for (auto &item: Ain_edges) {
        Pi_nodes.insert(item.first);
    }

    for (auto &node: Pi_nodes) {
        for (boost::tie(ei, e_end) = out_edges(node, G); ei!=e_end; ++ei) {
            Pi.insert(make_pair(source(*ei, G), target(*ei, G)));
        }
        for (boost::tie(qi, q_end) = in_edges(node, G); qi!=q_end; ++qi) {
            Pi.insert(make_pair(source(*qi, G), target(*qi, G)));
        }
    }
    return Pi;
}

vector<int> explore_update(DiGraph G, edge_prob B, edge_prob Q, edge_prob P, set<int> S, unordered_map<int,vector<int> > Nf,
                           unordered_map<int, vector<pair<int, int> > > Ef, vector<int> Phi, int K, double theta) {


    vector<int> F;
    unordered_map<int, set<pair<int, int> > > Ain_edges;
    set<pair<int, int> > Pi;
    int max_feature;
    double max_spread, spread;
    unordered_map<int, bool> selected;
    bool intersected;
    edge_prob changed;
    clock_t begin, finish;

    Ain_edges = explore(G, P, S, theta);
    Pi = get_pi(G, Ain_edges, S);

    while (F.size() < K) {
        cout << F.size() << ": ";
        fflush(stdout);
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
                begin = clock();
                if (intersected) {
                    F.push_back(f);
                    changed = increase_probabilities(G, B, Q, Nf, F, Ef[f], P);
                    Ain_edges = explore(G, P, S, theta);
                    begin = clock();
                    spread = update(Ain_edges, S, P);
                    finish = clock();
                    cout << "Time to update: " << (double) (finish - begin)/(CLOCKS_PER_SEC) << endl;
                    if (spread > max_spread) {
                        max_spread = spread;
                        max_feature = f;
                    }
                    decrease_probabilities(changed, P);
                    F.pop_back();
                }
            }
        }
        cout << endl;
        F.push_back(max_feature);
        selected[max_feature] = true;
//        printf("f = %i; spread = %.2f\n", max_feature, max_spread);
        increase_probabilities(G, B, Q, Nf, F, Ef[max_feature], P);
    }
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
    unordered_map<int, vector<pair<int, int> > > Ef;
    edge_prob B, Q, P;
    unordered_map<int, set<int> > groups;
    vector<int> F;
    set<int> S;
    int I, K;
    unordered_map<int, double> influence;
    double theta;
    in_edge_iter qi, q_end;

    DiGraph G = read_graph("datasets/gnutella.txt");
    read_features("datasets/gnutella_mem.txt", G, Nf, Ef);
    read_probabilities("datasets/gnutella_mv.txt", B);
    read_probabilities("datasets/gnutella_mv.txt", Q);
    read_probabilities("datasets/gnutella_mv.txt", P);
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

    unordered_map<int, set<pair<int, int> > > Ain_edges;
    clock_t begin, finish;
    begin = clock();
    Ain_edges = explore(G, P, S, theta);
    begin = clock();
    double total = update(Ain_edges, S, P);
    finish = clock();
    cout << (double) (finish - begin)/(CLOCKS_PER_SEC) << " " << total << endl;

//    cout << "Start explore-update" << endl;
//    F = explore_update(G, B, Q, P, S, Nf, Ef, Phi, K, theta);
//    for (auto &f: F)
//        cout << f << " ";
//    cout << endl;

    return 0;
}