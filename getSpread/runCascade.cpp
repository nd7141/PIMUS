// Calculate Independent Cascade size
// Input: <filename of the weighted graph> <number of vertices> <filename of seeds> <max_iter>
// Example: Flickr_dir.txt 5000 K 100

#include <iostream>
#include <fstream> // for reading files
#include <cstdlib> // for atoi, rand
#include <map>
#include <sys/time.h> // for gettimeofday
#include <stdio.h> // printf()
#include <stdlib.h> // exit()
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <utility>

using namespace boost;
using namespace std;


double printTime(struct timeval &start, struct timeval &end){
	double t1=start.tv_sec+(start.tv_usec/1000000.0);
	double t2=end.tv_sec+(end.tv_usec/1000000.0);
	return t2-t1;
}

void readGraph(vector<vector<pair<int, double> > >& G, string dataset_f) {
    ifstream infile(dataset_f.c_str());
    if (infile==NULL){
        cout << "Unable to open the input file\n";
    }
    long int u, v;
    double p;
    pair<int, double> endpoint1, endpoint2;
    while (infile >> u >> v >> p){
    	endpoint1 = make_pair(u, p);
    	endpoint2 = make_pair(v, p);
        G[v].push_back(endpoint1);
        G[u].push_back(endpoint2);
    }
}

int getSeeds(vector<int>& S, string seeds_F) {
	ifstream infile(seeds_F);
	string line;
	int k=-1;
	while(getline(infile, line)) {
		int count = 0;
		tokenizer<> tok(line);
		for(tokenizer<>::iterator beg=tok.begin(); beg!=tok.end();++beg) {
			int v=lexical_cast<int>(*beg);
			// the first number in line is seed set size k
			if (count == 0) {
				k = v;
				count++;
			}
			// all others are nodes
			else {
				S.push_back(v);
				count++;
			}
		}
	}
	return k;
}

double calculateSpread(vector<vector<pair<int, double> > > G, vector<int> S, int V, int I) {
	double cascade_size = 0;
	int iter = 0;
	map<int, bool> activated;
	vector<int> T;
	while (iter < I) {
		// activate seeds
		for (int i = 0; i < V; ++i) {
			activated[i] = false;
		}
		for (vector<int>::iterator it = S.begin(); it != S.end(); ++it) {
			activated[*it] = true;
			T.push_back(*it);
		}
		// activate new nodes
		vector<int>::size_type ix = 0; // activated node index
		int u;
		while (ix < T.size()) {
			u = T[ix];
			for (vector<pair<int, double> >::iterator it = G[u].begin(); it != G[u].end(); ++it) {
				int v = it->first;
				double p = it->second;
				if (!activated[v]) {
					double random = (double) rand()/RAND_MAX;
					if (random <= p) {
						activated[v] = true;
						T.push_back(v);
					}
				}
			}
			ix++;
		}
		cascade_size += T.size();
		iter++;
		T.clear();
	}
	double spread = (double)cascade_size/(double)I;
	return spread;
}

int main(int argc, char* argv[]) {
	srand(time(NULL));
	struct timeval ex_start, ex_finish;
	gettimeofday(&ex_start, NULL);

	// read parameters from command-line
	const string dataset_f = argv[1]; // filename of the dataset
	const int V = atoi(argv[2]); // number of nodes
	const string seeds_F = argv[3]; // filename of seeds
	const int I = atoi(argv[4]);         // number of MC simulations
	const string spread_f = argv[5]; // filename for spread

	cout << "Graph: " << dataset_f << " I: " << I << endl;

	vector<vector<pair<int, double> > > G(V);
	readGraph(G, dataset_f);

	vector<int> S;
	int k = getSeeds(S, seeds_F);

	double spread = calculateSpread(G, S, V, I);

	cout << "k:" << k << " Average cascade size: " << spread << endl;

	std::ofstream myfile;
	myfile.open(spread_f, ios_base::app);
	myfile << k << " " << spread << endl;
	myfile.close();

	gettimeofday(&ex_finish, NULL);
	cout << "* Execution time: " << printTime(ex_start, ex_finish) << " sec." << endl;
	return 0;
}