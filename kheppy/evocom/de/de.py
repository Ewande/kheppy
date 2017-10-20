from kheppy.evocom.commons import BaseAlgorithm
from kheppy.evocom.de.population import PopulationDE


class DiffEvolution(BaseAlgorithm):

    def __init__(self):
        super().__init__()

        self.de_params()

    def de_params(self, p_cross=0.75, diff_weight=1, mut_strat='rand'):
        """Set parameters specific to the differential evolution.

        :param p_cross: (binary) crossover probability
        :param diff_weight: differential weight, float or list of floats (see mut_strat parameter)
        :param mut_strat: mutation strategy
            'rand' or
            'best' - uses 1 differential weight,

            'rand-to-best' or
            'curr-to-best' - uses 2 differential weights,

        :return: this DiffEvolution object
        """
        if mut_strat not in ['rand', 'best', 'rand-to-best', 'curr-to-best']:
            raise ValueError('Incorrect value of "mut_strat" parameter. See function description.')
        if mut_strat in ['rand-to-best', 'curr-to-best'] and (not isinstance(diff_weight, (list, tuple))
                                                              or len(diff_weight)) < 2:
            raise ValueError('Selected mutation strategy uses 2 differential weights but only 1 was provided.')

        self.params['p_cross'] = p_cross
        self.params['diff_weight'] = diff_weight if isinstance(diff_weight, (list, tuple)) else [diff_weight]
        self.params['mut_strat'] = mut_strat
        return self

    def _get_init_pop(self):
        return PopulationDE(self.params['model'], self.params['pop_size']).initialize(self.params['param_init'])

    def _get_next_pop(self, pop):
        candidates = pop.get_candidate_pop(self.params['p_cross'], self.params['diff_weight'], self.params['mut_strat'])
        to_evaluate = PopulationDE(candidates.network, candidates.pop)
        if self.params['pos'] != 'static' or pop.average_fitness() == 0:
            to_evaluate.pop += pop.pop
        time = self._evaluate_pop(to_evaluate)
        ffe = len(to_evaluate.pop) * self.params['num_sim']

        final_list = [org if org.fitness >= cand.fitness else cand for org, cand in zip(pop.pop, candidates.pop)]
        return PopulationDE(pop.network, final_list), ffe, time
