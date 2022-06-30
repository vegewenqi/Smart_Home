from abc import ABC, abstractmethod

from base_types import Agent, Env
from replay_buffer import BasicBuffer


class TrainableController(Agent, ABC):
    needs_training = True
    needs_data = True

    def __init__(self, env):
        super().__init__(env)
        self.cost_fn = self.env.cost_fn

    @abstractmethod
    def train(self, replay_buffer: BasicBuffer):
        """
        Trains the controller from experience
        """
        pass

    # @abstractmethod
    # def save(self, path):
    #     pass

    # @abstractmethod
    # def load(self, path):
    #     pass


class OptimizationController(Agent, ABC):
    addition_solninfo = True