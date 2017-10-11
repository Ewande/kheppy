from numpy.random import uniform, randint
import numpy as np

from kheppy.evocom.commons import Controller


class ControllerDE(Controller):

    def copy(self):
        return ControllerDE(self.weights, self.biases, self.fitness)
