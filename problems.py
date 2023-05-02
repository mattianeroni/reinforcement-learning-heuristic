import numpy as np 
import math 
import itertools 


def euclidean (node1, node2):
    return math.sqrt( math.pow(node1[1] - node2[1], 2) + math.pow(node1[2] - node2[2], 2) )



class TSP:
    # Names of the file where the benchmark problems are written
    benchmarks = (
        "A-n32-k5_input_nodes.txt",
        "A-n38-k5_input_nodes.txt",
        "A-n45-k7_input_nodes.txt",
        "A-n55-k9_input_nodes.txt",
        "A-n60-k9_input_nodes.txt",
        "A-n61-k9_input_nodes.txt",
        "A-n65-k9_input_nodes.txt",
        "A-n80-k10_input_nodes.txt",
        "B-n50-k7_input_nodes.txt",
        "B-n52-k7_input_nodes.txt",
        "B-n57-k9_input_nodes.txt",
        "B-n78-k10_input_nodes.txt",
        "E-n22-k4_input_nodes.txt",
        "E-n30-k3_input_nodes.txt",
        "E-n33-k4_input_nodes.txt",
        "E-n51-k5_input_nodes.txt",
        "E-n76-k10_input_nodes.txt",
        "E-n76-k14_input_nodes.txt",
        "E-n76-k7_input_nodes.txt",
        "F-n135-k7_input_nodes.txt",
        "F-n45-k4_input_nodes.txt",
        "F-n72-k4_input_nodes.txt",
        "M-n101-k10_input_nodes.txt",
        "M-n121-k7_input_nodes.txt",
        "P-n101-k4_input_nodes.txt",
        "P-n22-k8_input_nodes.txt",
        "P-n40-k5_input_nodes.txt",
        "P-n50-k10_input_nodes.txt",
        "P-n55-k15_input_nodes.txt",
        "P-n65-k10_input_nodes.txt",
        "P-n70-k10_input_nodes.txt",
        "P-n76-k4_input_nodes.txt",
        "P-n76-k5_input_nodes.txt"
    )


    def __init__(self):
        self.nodes = tuple()
        self.dists = None


    def readfile (self, filename, path = "./data/" ):

        with open(path + filename) as file:
            # Split each line in tokens
            rows = [" ".join(line.split()).split(" ") for line in file]

            self.nodes = tuple(
                np.asarray([i, int(float(tokens[0])), int(float(tokens[1])),])
            for i, tokens in enumerate(rows))

            self.dists = np.zeros((len(self.nodes), len(self.nodes)), dtype=np.float32)
            for i, j in itertools.combinations(range(len(self.nodes)), 2):
                dist = euclidean(self.nodes[i], self.nodes[j])
                self.dists[i, j] = dist 
                self.dists[j, i] = dist
            
            self.nodes = np.asarray(self.nodes, dtype=np.float32)
 
    
    def evaluate (self, solution):
        roll_solution = np.roll(solution, -1)
        return self.dists[solution, roll_solution].sum()
