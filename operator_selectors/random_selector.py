import random 

class RandomSelector:

    def __init__(self, operators):
        self.operators = operators 

    def get_operator (self, state):
        return random.choice(self.operators)

    def callback (self, state, reward, game_over):
        pass