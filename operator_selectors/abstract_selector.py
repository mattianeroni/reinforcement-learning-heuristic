import abc


class AbstractSelector (abc.ABC):

    @abc.abstractmethod 
    def get_operator(self, state):
        pass 


    @abc.abstractmethod 
    def callback (self, state, reward, game_over):
        pass