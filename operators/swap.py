import random 
import numpy as np 
import itertools

def single_close_swap (solution):
    i = random.randint(0, len(solution) - 2)
    j = i + 1
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol 


def single_swap (solution):
    i = random.randint(0, len(solution) - 1)
    j = random.randint(0, len(solution) - 1)
    new_sol = solution.copy()
    new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
    return new_sol 


def shuffle (solution):
    new_sol = solution.copy()
    np.random.shuffle(new_sol)
    return new_sol


def opt2 (solution):
    i = random.randint(0, len(solution) - 1)
    j = random.randint(0, len(solution) - 1)
    i, j = min(i, j), max(i, j)
    new_sol = np.concatenate((solution[:i], np.flip(solution[i:j]), solution[j:] ))
    return new_sol 


