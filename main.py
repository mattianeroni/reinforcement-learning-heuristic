from problems import TSP 
from large_neighbour_search import LargeNeighbourSearch
from operator_selectors import RandomSelector, RLSelector 
from models import LinearModel, QTrainer
import util

from operators.swap import (
    single_close_swap,
    single_swap,
    opt2,
    shuffle,
)

import random



if __name__ == '__main__':

    benchmarks = TSP.benchmarks
    tsp = TSP()
    
    benchmark = random.choice(benchmarks)

    # Problem instance 
    tsp.readfile(benchmark)
    N = len(tsp.nodes)
    print(f"Problem: {benchmark} - Number of nodes: {N}")

    operators = [
        single_close_swap,
        single_swap,
        opt2,
        shuffle,
    ]
    
    # Large Neighbour Search
    classic_large_neighbour = LargeNeighbourSearch(
        problem=tsp, 
        op_selector=RandomSelector(operators), 
        max_wrong_steps=200
    )

    # Large Neighbour Search with RL model
    model = LinearModel(N * 2 + 2, 256, len(operators))
    op_selector = RLSelector(
        operators=operators, 
        model=model, 
        trainer=QTrainer(model, lr=0.001, gamma=0.9), 
        epsilon=0.1, 
        max_memory=100_000, 
        batch_size=500
    )
    RL_heuristic = LargeNeighbourSearch(tsp, op_selector, max_wrong_steps=50)

    bests1, bests2 = [], []


    for _ in range(3000):
        RL_heuristic.step()
        classic_large_neighbour.step()
        bests1.append(RL_heuristic.current_best_cost)
        bests2.append(classic_large_neighbour.current_best_cost)
        util.plot(bests1, bests2)

    