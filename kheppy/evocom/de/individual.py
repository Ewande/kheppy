from numpy.random import uniform, randint
import numpy as np

from kheppy.evocom.commons import Controller


class ControllerDE(Controller):

    def copy(self):
        return ControllerDE(self.weights, self.biases, self.fitness)

    def add_diff_vector(self, fst_ind, snd_ind, diff_weight):
        """Create a new individual by adding weighted difference vector of fst_ind and snd_ind to this individual.

        Performed elementwise:
            new_ind = self + diff_weight * (fst_ind - snd_ind)

        :param fst_ind: first individual
        :param snd_ind: second individual
        :param diff_weight: differential weight

        :return: new ControllerDE object
        """

        weights = [wa + diff_weight * (wb - wc) for wa, wb, wc in zip(self.weights, fst_ind.weights, snd_ind.weights)]
        biases = [ba + diff_weight * (bb - bc) for ba, bb, bc in zip(self.biases, fst_ind.biases, snd_ind.biases)]
        return ControllerDE(weights, biases)

    def binary_cross(self, c2, p_cross):
        weights = [np.where(uniform(0, 1, w.shape) < p_cross, w, w2) for w, w2 in zip(self.weights, c2.weights)]
        biases = [np.where(uniform(0, 1, b.shape) < p_cross, b, b2) for b, b2 in zip(self.biases, c2.biases)]

        return ControllerDE(weights, biases)
