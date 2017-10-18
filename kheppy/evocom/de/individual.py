from numpy.random import uniform, randint
import numpy as np

from kheppy.evocom.commons import Controller


class ControllerDE(Controller):

    def copy(self):
        return ControllerDE(self.weights, self.biases, self.fitness)

    def prepare_candidate(self, source_candidates, diff_weights):
        if len(source_candidates) == 2:
            snd, thrd = source_candidates
            diff_weight = diff_weights[0]
        else:
            snd, thrd = None, None
            diff_weight = None

        weights = [w_a + diff_weight * (w_b - w_c) for w_a, w_b, w_c in zip(self.weights, snd.weights, thrd.weights)]
        biases = [b_a + diff_weight * (b_b - b_c) for b_a, b_b, b_c in zip(self.biases, snd.biases, thrd.biases)]
        return ControllerDE(weights, biases)

    def binary_cross(self, c2, p_cross):
        weights = [np.where(uniform(0, 1, w.shape) < p_cross, w, w2) for w, w2 in zip(self.weights, c2.weights)]
        biases = [np.where(uniform(0, 1, b.shape) < p_cross, b, b2) for b, b2 in zip(self.biases, c2.biases)]

        return ControllerDE(weights, biases)
