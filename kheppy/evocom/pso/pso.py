from kheppy.evocom.commons import BaseAlgorithm
from kheppy.evocom.pso.population import PopulationPSO


class PartSwarmOpt(BaseAlgorithm):

    def __init__(self):
        super().__init__()

        self.pso_params()

    def pso_params(self, inertia_weight=1, cognitive_param=2, social_param=2):
        self.params['inertia'] = inertia_weight
        self.params['cognitive'] = cognitive_param
        self.params['social'] = social_param
        return self

    def _get_init_pop(self):
        return PopulationPSO(self.params['model'], self.params['pop_size']).initialize(self.params['param_init'])

    def _get_next_pop(self, pop):
        time = self._evaluate_pop(pop)
        ffe = len(pop) * self.params['num_sim']

        pop.update_local_best()
        pop.update_global_best()
        pop.move_particles(self.params['inertia'], self.params['cognitive'], self.params['social'])
        return pop, ffe, time
