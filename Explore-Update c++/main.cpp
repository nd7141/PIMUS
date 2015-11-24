#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <fstream>

using namespace std;

typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::directedS> DiGraph;
typedef typename boost::graph_traits<DiGraph>::edge_descriptor Edge;
typedef pair<DiGraph::edge_descriptor, bool> Edge_insertion;



DiGraph read_graph(string graph_filename) {
    cout << graph_filename << endl;
    ifstream infile(graph_filename);
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }

    map<int, int> mapped;
    int u, v;
    int node_count=0;
    Edge_insertion edge_insertion;
    DiGraph G(1);

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

    // Iterate through the vertices and print them out
    typedef boost::graph_traits<DiGraph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;
    for (vp = boost::vertices(G); vp.first != vp.second; ++vp.first)
        cout << vp << endl;
//        std::cout << G[*vp.first] << " " << G[*vp.first] << endl;
    std::cout << std::endl;

    return G;
}

double calculate_spread() {
    return 0;
}

vector<int> greedy (DiGraph G, int K) {
    vector<int> F;

    while (F.size() < K) {
        K--; // change to greedy algorithm
    }

    return F;
}


int main(int argc, char* argv[]) {
    // read parameters from command-line
    if (argc > 1) {
        const string path = argv[1]; // prefix path to directory with necessary files
        const long int V = atoi(argv[2]); // number of nodes
        const int K = atoi(argv[4]);         // number of features
    }


    DiGraph G = read_graph("datasets/gnutella.txt");

    return 0;
}