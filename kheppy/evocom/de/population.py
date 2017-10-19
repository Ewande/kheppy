from kheppy.evocom.commons.population import Population
from kheppy.evocom.de.individual import ControllerDE
from numpy.random import choice


class PopulationDE(Population):

    def initialize(self, init_limits):
        self.pop = []
        for _ in range(self.pop_size):
            weights = self.network.random_weights_list(init_limits)
            biases = self.network.random_biases_list(init_limits)
            self.pop.append(ControllerDE(weights, biases))
        return self

    def get_candidate_pop(self, p_cross, diff_weight, mut_strat):
        candidates = []
        best = self.best() if mut_strat != 'rand' else None
        for i, ind in enumerate(self.pop):
            ind_list = [x for x in range(len(self.pop)) if x != i and self.pop[x] != best]
            a, b, c = choice(ind_list, 3, replace=False)

            if mut_strat == 'rand':
                cand = self.pop[a].add_diff_vector(self.pop[b], self.pop[c], diff_weight[0])
            elif mut_strat == 'best':
                cand = best.add_diff_vector(self.pop[b], self.pop[c], diff_weight[0])
            elif mut_strat == 'rand-to-best':
                cand = self.pop[a].add_diff_vector(self.pop[b], self.pop[c], diff_weight[0])
                cand = cand.add_diff_vector(best, self.pop[a], diff_weight[1])
            else:  # mut_strat == 'curr-to-best'
                cand = ind.add_diff_vector(self.pop[b], self.pop[c], diff_weight[0])
                cand = cand.add_diff_vector(best, ind, diff_weight[1])

            final_cand = ind.binary_cross(cand, p_cross)
            candidates.append(final_cand)

        return PopulationDE(self.network, candidates)
