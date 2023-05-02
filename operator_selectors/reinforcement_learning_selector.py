import torch
import random
import numpy as np
import collections









class RLSelector:

    """ A Reinforcement Learning based Selector """

    def __init__(self, operators, model, trainer, epsilon=0.05, max_memory=100_000, batch_size=1000):
        """
        :param epsilon: Randomness factor
        """
        self.iter = 0
        self.max_memory = max_memory
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.operators = operators
        self.memory = collections.deque(maxlen=max_memory)
        self.model = model
        self.trainer = trainer

        self.score = 0
        self.record = 0
        self.old_state = None
        self.last_selection = np.zeros(len(self.operators))
        self.last_selection[0] = 1


    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))


    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)
    

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)


    def get_operator (self, state):
        self.last_selection = np.zeros(len(self.operators))

        # Random operator
        if random.random() < self.epsilon:
            move = random.randint(0, len(self.operators) - 1)
            self.last_selection[move] = 1
            return self.operators[move]
        
        # Predicted best operator
        prediction = self.model(torch.tensor(state, dtype=torch.float32))
        move = torch.argmax(prediction).item()
        self.last_selection[move] = 1
        return self.operators[move]


    def callback (self, state, reward, game_over):
        self.iter += 1
        self.train_short_memory(self.old_state, self.last_selection, reward, state, game_over)
        self.remember(self.old_state, self.last_selection, reward, state, game_over)

        if game_over:
            self.train_long_memory()

