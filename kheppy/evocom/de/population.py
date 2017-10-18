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
        best = self.best() if mut_strat == 'best' else None
        for i, elem in enumerate(self.pop):
            ind_list = [x for x in range(len(self.pop)) if x != i and self.pop[x] != best]
            if mut_strat == 'rand':
                a, b, c = choice(ind_list, 3, replace=False)
                fst, snd, thrd = self.pop[a], self.pop[b], self.pop[c]
            elif mut_strat == 'best':
                b, c = choice(ind_list, 2, replace=False)
                fst, snd, thrd = best, self.pop[b], self.pop[c]
            else:
                return None
            cand = fst.prepare_candidate([snd, thrd], [diff_weight])
            final_cand = fst.binary_cross(cand, p_cross)
            candidates.append(final_cand)

        return PopulationDE(self.network, candidates)
