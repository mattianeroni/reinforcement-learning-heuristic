import numpy as np 
import random 



class LargeNeighbourSearch:

    """ Large Neighbour Search metaheuristic """

    def __init__(self, problem, op_selector, acceptance=0.2, max_wrong_steps=200):
        self.problem = problem

        self.current_sol = problem.nodes[:,0].astype(np.int32)
        np.random.shuffle(self.current_sol)
        self.current_sol_cost = problem.evaluate(self.current_sol)
        self.current_best = self.current_sol.copy()
        self.current_best_cost = problem.evaluate(self.current_best)

        self.op_selector = op_selector
        self.op_selector.old_state = np.hstack((
            self.current_sol, 
            self.current_best, 
            self.current_sol_cost, 
            self.current_best_cost
        ))
        
        self.wrong_steps = 0
        self.max_wrong_steps = max_wrong_steps
        self.acceptance = acceptance

        self.sol_history = []
        self.best_history = []

    
    def step (self):
        op_selector, problem = self.op_selector, self.problem
        save_sol, save_best = self.sol_history.append, self.best_history.append 

        # Init the reward for the selector and the game over flag
        reward, game_over = 0, False

        # Pick an operator 
        state = np.hstack((
            self.current_sol, 
            self.current_best, 
            self.current_sol_cost, 
            self.current_best_cost
        ))
        operator = op_selector.get_operator(state)

        # Change the current solution
        new_sol = operator(self.current_sol)
        new_cost = problem.evaluate(self.current_sol)

        if new_cost < self.current_sol_cost or random.random() < self.acceptance:
            self.current_sol = new_sol
            self.current_sol_cost = new_cost

        self.current_sol_cost = new_cost
        save_sol(self.current_sol_cost)

        # Compare it to the best 
        if self.current_sol_cost < self.current_best_cost:
            self.current_best = self.current_sol
            self.current_best_cost = self.current_sol_cost
            self.wrong_steps = 0
            reward = 10
            save_best(self.current_best_cost)
        else:
            # Increase steps with no improvement
            self.wrong_steps += 1
            if self.wrong_steps > self.max_wrong_steps:
                self.wrong_steps = 0 
                reward = -10
                game_over = True


        # Pass informations to the operator selector
        state = np.hstack((
            self.current_sol, 
            self.current_best, 
            self.current_sol_cost, 
            self.current_best_cost
        ))
        op_selector.callback(state, reward, game_over)

        # Long training 

