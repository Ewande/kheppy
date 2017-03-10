from timeit import default_timer as timer
import numpy as np

from kheppy.core import SimList
from kheppy.evocom.ga.population import Population
from kheppy.utils import Reporter
from kheppy.utils import timestamp


class GeneticAlgorithm:

    def __init__(self):
        self.params = {}
        self.evo_params()
        self.sel_params()
        self.eval_params(model=None, fitness_func=None)
        self.sim_params(wd_path=None, robot_id=None)
        self.early_stopping(self.params['epochs'])
        self.reporter = Reporter(['best', 'avg', 'min', 'test'])
        self.best = None

    def evo_params(self, pop_size=100, p_mut=0.03, p_cross=0.75, epochs=100, param_init_limits=(-1, 1)):
        self.params['pop_size'] = pop_size
        self.params['p_mut'] = p_mut
        self.params['p_cross'] = p_cross
        self.params['epochs'] = epochs
        self.params['param_init'] = param_init_limits
        return self

    def sel_params(self, tournament_size=3):
        self.params['t_size'] = tournament_size
        return self

    def eval_params(self, model, fitness_func, num_cycles=80, steps_per_cycle=7, aggregate_func=np.mean,
                    num_sim=1, randomize_position=False):
        self.params['model'] = model
        self.params['fit_func'] = fitness_func
        self.params['num_cycles'] = num_cycles
        self.params['steps'] = steps_per_cycle
        self.params['agg_func'] = aggregate_func
        self.params['num_sim'] = num_sim
        self.params['rand_pos'] = randomize_position
        return self

    def sim_params(self, wd_path, robot_id, max_robot_speed=5):
        self.params['wd_path'] = wd_path
        self.params['robot_id'] = robot_id
        self.params['max_speed'] = max_robot_speed
        return self

    def early_stopping(self, epochs):
        self.params['stop'] = epochs
        return self

    def run(self, output_dir=None, num_proc=1, seed=42, verbose=False):
        np.random.seed(seed)
        with SimList(self.params['wd_path'], 2 * self.params['pop_size'], self.params['num_sim'],
                     self.params['robot_id']) as sim_list:

            if verbose:
                print('Using {} simulation(s) per controller.'.format(self.params['num_sim']))
                print('Preparing population...')
            pop = Population(self.params['model'], self.params['pop_size'], self.params['param_init'])
            best, no_change, i = None, 0, 0

            while i < self.params['epochs'] and no_change < self.params['stop']:
                if self.params['rand_pos']:
                    sim_list.randomize()
                else:
                    sim_list.reset()

                if verbose:
                    print('Epoch {:>3} '.format(i), end='', flush=True)
                start = timer()
                pop.cross(self.params['p_cross'])
                pop.mutate(self.params['p_mut'])
                time = pop.evaluate(sim_list, self.params['num_cycles'], self.params['steps'], self.params['max_speed'],
                                    self.params['fit_func'], self.params['agg_func'], num_proc)
                pop = pop.select(self.params['t_size'])

                ep_best = pop.best().copy()
                ep_best.reset_fitness()
                ep_best.evaluate(sim_list.init_sim.copy(), pop.network, 800, self.params['max_speed'],
                                 self.params['steps'], self.params['fit_func'], np.mean)

                if verbose:
                    print('finished in {:>5.2f}s (simulation: {:>5.2f}s) | best fitness: {:.4f} | '
                          'average fitness: {:.4f} | test position best fitness: {:.4f}.'
                          .format(timer() - start, time, pop.best().fitness, pop.average_fitness(), ep_best.fitness))
                self.reporter.put(['best', 'avg', 'min', 'test'], [pop.best().fitness, pop.average_fitness(),
                                                                   pop.worst().fitness, ep_best.fitness])

                if best is not None and pop.best().fitness - best.fitness < 0.0001:
                    no_change += 1
                else:
                    best = pop.best().copy()
                    no_change = 0
                i += 1

            if output_dir is not None:
                self.params['model'].save('{}ga_final_{}.nn'.format(output_dir, timestamp()), best.weights, best.biases)
            print('Evolution finished after {} iterations.'.format(i))
            self.best = best
