from kheppy.evocom.commons import BaseAlgorithm
from kheppy.evocom.de.population import PopulationDE


class DiffEvolution(BaseAlgorithm):

    def __init__(self):
        super().__init__()

        self.de_params()

    def de_params(self, p_cross=0.75, diff_weight=1, mut_strat='rand'):
        """Set parameters specific to differential evolution.

        :param p_cross: (binary) crossover probability
        :param diff_weight: differential weight
        :param mut_strat: mutation strategy, 'rand' or 'best'

        :return: this DiffEvolution object
        """
        self.params['p_cross'] = p_cross
        self.params['diff_weight'] = diff_weight
        self.params['mut_strat'] = mut_strat
        return self

    def _get_init_pop(self):
        return PopulationDE(self.params['model'], self.params['pop_size']).initialize(self.params['param_init'])

    def _get_next_pop(self, pop):
        candidates = pop.get_candidate_pop(self.params['p_cross'], self.params['diff_weight'], self.params['mut_strat'])
        time = self._evaluate_pop(candidates)
        ffe = len(candidates.pop) * self.params['num_sim']

        final_list = [org if org.fitness >= cand.fitness else cand for org, cand in zip(pop.pop, candidates.pop)]
        return PopulationDE(pop.network, final_list), ffe, time

