from timeit import default_timer as timer
import numpy as np
import os

from kheppy.core import SimList
from kheppy.evocom.commons import BaseAlgorithm
from kheppy.evocom.ga.population import PopulationGA
from kheppy.utils import timestamp


class GeneticAlgorithm(BaseAlgorithm):

    def __init__(self):
        super().__init__()

        self.sel_params()

    def evo_params(self, pop_size=100, p_mut=0.03, p_cross=0.75, epochs=100, param_init_limits=(-1, 1)):
        super().evo_params(pop_size, epochs, param_init_limits)
        self.params['p_mut'] = p_mut
        self.params['p_cross'] = p_cross
        return self

    def sel_params(self, sel_type=3):
        """Set parameters dedicated to selection process.
        
        :param sel_type: 'rw' (roulette wheel) or int (tournament size)
        
        :return: this GeneticAlgorithm object
        """
        if isinstance(sel_type, int) or sel_type == 'rw':
            self.params['sel_type'] = sel_type
            return self
        else:
            raise ValueError('Unsupported selection type. See function docstring.')

    def run(self, output_dir=None, num_proc=1, seed=42, verbose=False):
        np.random.seed(seed)
        with SimList(self.params['wd_path'], 2 * self.params['pop_size'] + 1, self.params['num_sim'],
                     self.params['robot_id']) as sim_list:

            if verbose:
                print('Using {} simulation(s) per controller.'.format(self.params['num_sim']))
                print('Preparing population...')
            pop = PopulationGA(self.params['model'], self.params['pop_size']).initialize(self.params['param_init'])
            best, no_change, i = None, 0, 0

            sim_list.shuffle_defaults(seed=seed)
            sim_list.reset_to_defaults()

            while i < self.params['epochs'] and no_change < self.params['stop']:

                if verbose:
                    print('Epoch {:>3} '.format(i + 1), end='', flush=True)
                start = timer()
                pop.cross(self.params['p_cross'])
                pop.mutate(self.params['p_mut'])
                time = pop.evaluate(sim_list, self.params['num_cycles'], self.params['steps'], self.params['max_speed'],
                                    self.params['fit_func'], self.params['agg_func'], num_proc)
                pop = pop.select(self.params['sel_type'])

                if verbose:
                    print('finished in {:>5.2f}s (simulation: {:>5.2f}s) | max fitness: {:.4f} | '
                          'average fitness: {:.4f} | min fitness: {:.4f}.'
                          .format(timer() - start, time, pop.best().fitness, pop.average_fitness(),
                                  pop.worst().fitness))
                self.reporter.put(['max', 'avg', 'min', 'start_pos'],
                                  [pop.best().fitness, pop.average_fitness(), pop.worst().fitness,
                                  [sim.get_robot_position() for sim in sim_list.default_sims]])

                if best is not None and pop.best().fitness - best.fitness < 0.0001:
                    no_change += 1
                else:
                    best = pop.best().copy()
                    no_change = 0
                i += 1

                self._prepare_positions(sim_list)

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.params['model'].save('{}ga_final_{}.nn'.format(output_dir, timestamp()), best.weights, best.biases)
            if verbose:
                print('Evolution finished after {} iterations.'.format(i))
            self.best = best
