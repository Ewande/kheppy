from timeit import default_timer as timer
import numpy as np
import os

from kheppy.core import SimList
from kheppy.core import Simulation
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
        self.reporter = Reporter(['max', 'avg', 'min', 'start_pos'])
        self.best = None

    def evo_params(self, pop_size=100, p_mut=0.03, p_cross=0.75, epochs=100, param_init_limits=(-1, 1)):
        self.params['pop_size'] = pop_size
        self.params['p_mut'] = p_mut
        self.params['p_cross'] = p_cross
        self.params['epochs'] = epochs
        self.params['param_init'] = param_init_limits
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

    def eval_params(self, model, fitness_func, num_cycles=80, steps_per_cycle=7, aggregate_func=np.mean,
                    num_positions=1, position='static', move_step=1, move_noise=0):
        """Set parameters dedicated to evaluation process.
        
        :param model: 
        :param fitness_func: 
        :param num_cycles: 
        :param steps_per_cycle: 
        :param aggregate_func: 
        :param num_positions: 
        :param position: starting position policy during evolution
            static  - select random position before evolution starts, 
                    use that position in every epoch
            dynamic - select random position before each epoch,
            moving  - select random position before evolution starts, 
                    move position in random direction before each epoch,
                    see parameters move_step and move_noise.
        :param move_step: used only when position='moving'
        :param move_noise: used only when position='moving'
        
        :return: this GeneticAlgorithm object
        """
        self.params['model'] = model
        self.params['fit_func'] = fitness_func
        self.params['num_cycles'] = num_cycles
        self.params['steps'] = steps_per_cycle
        self.params['agg_func'] = aggregate_func
        self.params['num_sim'] = num_positions
        self.params['pos'] = position
        self.params['move_step'] = move_step
        self.params['move_noise'] = move_noise
        return self

    def sim_params(self, wd_path, robot_id, max_robot_speed=5):
        self.params['wd_path'] = wd_path
        self.params['robot_id'] = robot_id
        self.params['max_speed'] = max_robot_speed
        return self

    def early_stopping(self, epochs):
        self.params['stop'] = epochs
        return self

    def _prepare_positions(self, sim_list):
        if self.params['pos'] == 'dynamic':
            sim_list.shuffle_defaults()
        if self.params['pos'] == 'moving':
            sim_list.move_forward_defaults(self.params['move_step'], self.params['move_noise'])
        sim_list.reset_to_defaults()

    def run(self, output_dir=None, num_proc=1, seed=42, verbose=False):
        np.random.seed(seed)
        with SimList(self.params['wd_path'], 2 * self.params['pop_size'] + 1, self.params['num_sim'],
                     self.params['robot_id']) as sim_list:

            if verbose:
                print('Using {} simulation(s) per controller.'.format(self.params['num_sim']))
                print('Preparing population...')
            pop = Population(self.params['model'], self.params['pop_size'], self.params['param_init'])
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

    def test(self, seed=50, num_points=1000, num_cycles=160, controller=None, verbose=False):
        if controller is None:
            controller = self.best.copy()
        else:
            controller = controller.copy()

        if verbose:
            print('Testing using {} starting points. Single evaluation length = {} cycles.'
                  .format(num_points, num_cycles))

        with Simulation(self.params['wd_path']) as sim:
            sim.set_controlled_robot(self.params['robot_id'])
            sim.set_seed(seed)
            res = []
            for i in range(num_points):
                sim.move_robot_random()
                controller.reset_fitness()
                controller.evaluate(sim, self.params['model'], num_cycles, self.params['max_speed'],
                                    self.params['steps'], self.params['fit_func'], np.mean)
                res.append(controller.fitness)
                if verbose:
                    print('\rTesting progress: {:5.2f}%...'.format(100. * (i + 1) / num_points), end='', flush=True)
            if verbose:
                print('\nAverage fitness in test: {:.4f}.'.format(np.mean(res)))

            return np.mean(res)
