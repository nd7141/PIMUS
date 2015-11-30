#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <unordered_set>
#include <time.h>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>

using namespace std;

typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::bidirectionalS> DiGraph;
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

    map<int, int> mapped;
    int u, v;
    int node_count=0;
    pair<DiGraph::edge_descriptor, bool> edge_insertion;
    DiGraph G;

    while (infile >> u >> v) {
        if (mapped.find(u) == mapped.end()) {
            mapped[u] = node_count;
            node_count++;
        }
        if (mapped.find(v) == mapped.end()) {
            mapped[v] = node_count;
            node_count++;
        }
        edge_insertion=boost::add_edge(mapped[u], mapped[v], G);
        if (!edge_insertion.second) {
            std::cout << "Unable to insert edge\n";
        }
    }
    return G;
}

void read_features(string feature_filename, DiGraph G, map<int, vector<int> > &Nf, map<int, vector<pair<int, int> > > &Ef) {

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


void read_groups(string group_filename, map<int, vector<int> > &groups) {
    ifstream infile(group_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    string line;
    vector<string> line_splitted;

    while (getline(infile, line)) {
        boost::split(line_splitted, line, boost::is_any_of(" "));
        vector<int> nodes;
        for (int i = 1; i < line_splitted.size(); ++i) {
            nodes.push_back(stoi(line_splitted[i]));
        }
        groups[stoi(line_splitted[0])] = nodes;
    }
}

edge_prob increase_probabilities(DiGraph G, edge_prob B, edge_prob Q, map<int, vector<int> > Nf, vector<int> F,
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

double calculate_spread (DiGraph G, edge_prob B, edge_prob Q, map<int, vector<int> > Nf, vector<int> S,
                        vector<int> F, map<int, vector<pair<int, int> > > Ef, int I) {

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
    map<int, bool> activated;
    vector<int> T;
    int u, v;
    double p;
    out_edge_iter ei, e_end;
    for (int it=0; it < I; ++it) {
        for (vp = boost::vertices(G); vp.first != vp.second; ++vp.first) {
            u = *vp.first;
            activated[u] = false;
        }
        for (int j=0; j < S.size(); ++j) {
            activated[S[j]] = false;
            T.push_back(S[j]);
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

vector<int> greedy(DiGraph G, edge_prob B, edge_prob Q, vector<int> S, map<int, vector<int> > Nf,
                   map<int, vector<pair<int, int> > > Ef, vector<int> Phi, int K, int I) {
    vector<int> F;
    edge_prob P;
    map<int, bool> selected;
    edge_prob changed;
    double spread, max_spread;
    int max_feature;
    map<int, double> influence;

    P.insert(B.begin(), B.end());

    int count = 0;
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
        printf("f = %i; spread = %.2f\n", max_feature, max_spread);
        increase_probabilities(G, B, Q, Nf, F, Ef[max_feature], P);
        influence[F.size()] = max_spread;
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

    map<int, vector<int> > Nf;
    map<int, vector<pair<int, int> > > Ef;
    edge_prob B, Q, P;
    map<int, vector<int> > groups;
    vector<int> F;
    vector<int> S;
    int I, K;

    DiGraph G = read_graph("datasets/gnutella.txt");
    read_features("datasets/gnutella_mem.txt", G, Nf, Ef);
    read_probabilities("datasets/gnutella_mv.txt", B);
    read_probabilities("datasets/gnutella_mv.txt", Q);
    read_groups("datasets/gnutella_com.txt", groups);

    vector<int> Phi;
    for (auto &item: Ef) {
        Phi.push_back(item.first);
    }

    S = groups[1];
    I = 5;
    K = 2;

    F = greedy(G, B, Q, S, Nf, Ef, Phi, K, I);
    cout << "F = ";
    for (int i = 0; i < F.size(); ++i)
        cout << F[i] << " ";


    cout << "Spread: " << calculate_spread(G, B, Q, Nf, S, F, Ef, I) << endl;

    return 0;
}