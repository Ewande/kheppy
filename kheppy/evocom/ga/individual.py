from numpy.random import uniform, randint
import numpy as np

from kheppy.evocom.commons import Controller


class ControllerGA(Controller):

    def copy(self):
        return ControllerGA(self.weights, self.biases, self.fitness)

    def mutate(self, prob):
        self.weights = [(w + uniform(-0.05, 0.05, w.shape) * (uniform(0, 1, w.shape) < prob)) for w in self.weights]
        self.biases = [(b + uniform(-0.05, 0.05, b.shape) * (uniform(0, 1, b.shape) < prob)) for b in self.biases]

    def cross(self, c2):
        c1 = self
        c1_new = ControllerGA()
        c2_new = ControllerGA()
        for w1, w2, b1, b2 in zip(c1.weights, c2.weights, c1.biases, c2.biases):
            w1f, w2f, b1f, b2f = w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()

            half_w = randint(0, len(w1f))  # int(len(w1f)/2)
            half_b = randint(0, len(b1f))  # int(len(b1f)/2)
            w1_new = np.reshape(np.append(w1f[:half_w], w2f[half_w:]), w1.shape)
            w2_new = np.reshape(np.append(w2f[:half_w], w1f[half_w:]), w1.shape)
            b1_new = np.reshape(np.append(b1f[:half_b], b2f[half_b:]), b1.shape)
            b2_new = np.reshape(np.append(b2f[:half_b], b1f[half_b:]), b1.shape)
            c1_new.weights.append(w1_new)
            c2_new.weights.append(w2_new)
            c1_new.biases.append(b1_new)
            c2_new.biases.append(b2_new)

        return c1_new, c2_new
