'''
http://delivery.acm.org/10.1145/2730000/2723734/p1539-tang.pdf?ip=89.106.174.122&id=2723734&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=667647771&CFTOKEN=97444750&__acm__=1434964493_0e76cda204d26fbea8a507089b868b5f
'''
from __future__ import division
__author__ = 'sivanov'
import os, sys, time, json

def getNM(filename):
    nodes = set()
    edges = 0
    with open(filename) as f:
        for line in f:
            edges += 1
            nodes.update(map(int, line.split()[:2]))
    return len(nodes), edges

if __name__ == "__main__":
    start2exec = time.time()

    if len(sys.argv) != 5:
        assert ValueError, 'command: python IMM_wrapper.py folder eps I undirected_dataset'

    # dataset_folder should contain graph_ic.inf file with a graph u v p

    folder = sys.argv[1] # path to the folder with dataset
    eps = sys.argv[2] # epsilon
    I = sys.argv[3] # number of MC simulations
    dataset = sys.argv[4]

    IMM_time = []
    Spread_time = []
    begin_seed = 1
    end_seed = 10
    step_seed = 1

    # create directory if not exist
    directory = "data/IMM_data/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # remove the output file if exists
    output = directory + 'spread.txt'
    if os.path.exists(output):
        os.remove(output)

    if not os.path.exists(folder + 'graph_ic.inf'):
        print folder
        raise ValueError, 'The folder with data should contain file graph_ic.inf with graph'

    # create file attribute.txt
    N, M = getNM(folder + 'graph_ic.inf')
    with open(folder+'/attribute.txt', 'w+') as f:
        f.write('n=%s\n' %N)
        f.write('m=%s\n' %M)

    # remove the time file if exists
    timing = directory + 'wiki_multivalency_time.txt'
    if os.path.exists(timing):
        os.remove(timing)

    # os.system('make -C ./getPossibleWorlds/')
    os.system('make -C ./comparison/IMM/')

    k_range = range(begin_seed, end_seed, step_seed)
    for k in k_range:
        if os.path.exists(directory + 'seeds%s.txt' %(k)):
            os.remove(directory + 'seeds%s.txt' %(k))
        start = time.time()
        os.system('./comparison/IMM/imm_discrete -dataset %s -epsilon %s -k %s  -model IC -seeds %s' %(folder, eps, k, directory + 'seeds%s.txt' %(k)))
        IMM_time.append(time.time() - start)
        os.system('./getSpread/runCascade %s %s %s %s %s' %(dataset, N, directory + 'seeds%s.txt' %(k), I, output))

    finish2exec = time.time() - start2exec

    # write timing to file
    with open(timing, 'w+') as f:
        for i in range(len(IMM_time)):
            f.write("%s %s\n" %(k_range[i], IMM_time[i]))

    # print '* To get worlds: %s sec' %finish2worlds
    print '* Total execution time: %s sec' %finish2exec

    console = []