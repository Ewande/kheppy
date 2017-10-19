from kheppy.evocom.commons import BaseAlgorithm
from kheppy.evocom.ga.population import PopulationGA


class GeneticAlgorithm(BaseAlgorithm):

    def __init__(self):
        super().__init__()

        self.ga_params()

    def ga_params(self, p_mut=0.03, p_cross=0.75, sel_type=3):
        """Set parameters specific to the genetic algorithm.
        
        :param p_mut: mutation probability, float
        :param p_cross: crossover probability, float
        :param sel_type: selection type, 'rw' (roulette wheel) or int (tournament size)
        
        :return: this GeneticAlgorithm object
        """
        self.params['p_mut'] = p_mut
        self.params['p_cross'] = p_cross

        if isinstance(sel_type, int) or sel_type == 'rw':
            self.params['sel_type'] = sel_type
        else:
            raise ValueError('Unsupported selection type. See function docstring.')

        return self

    def _get_init_pop(self):
        return PopulationGA(self.params['model'], self.params['pop_size']).initialize(self.params['param_init'])

    def _get_next_pop(self, pop):
        pop.cross(self.params['p_cross'])
        pop.mutate(self.params['p_mut'])

        time = self._evaluate_pop(pop)
        ffe = len(pop.pop) * self.params['num_sim']

        next_pop = pop.select(self.params['sel_type'])
        return next_pop, ffe, time
