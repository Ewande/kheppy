from kheppy.evocom.commons.population import Population
from kheppy.evocom.ga.individual import ControllerGA
from numpy.random import choice, uniform, shuffle
import numpy as np


class PopulationGA(Population):

    def initialize(self, init_limits):
        self.pop = []
        for _ in range(self.pop_size):
            weights = self.network.random_weights_list(init_limits)
            biases = self.network.random_biases_list(init_limits)
            self.pop.append(ControllerGA(weights, biases))
        return self

    def cross(self, prob):
        shuffle(self.pop)
        for i in range(0, len(self.pop), 2):
            if uniform() < prob:
                c1_new, c2_new = self.pop[i].cross(self.pop[i + 1])
                self.pop.append(c1_new)
                self.pop.append(c2_new)

    def mutate(self, prob):
        for controller in self.pop:
            controller.mutate(prob)

    def select(self, sel_type):
        if isinstance(sel_type, int):
            new_pop = []
            for _ in range(self.pop_size):
                group = choice(len(self.pop), sel_type, replace=False).tolist()
                max_ind = np.argmax([self.pop[i].fitness for i in group])
                best = self.pop[group[max_ind]]
                new_pop.append(best.copy())
        else:
            cum_fit = np.cumsum([elem.fitness for elem in self.pop])
            draws = np.random.uniform(0, cum_fit[-1], self.pop_size)
            indices = np.searchsorted(cum_fit, draws)
            new_pop = [self.pop[ind].copy() for ind in indices]

        return PopulationGA(self.network, new_pop)
