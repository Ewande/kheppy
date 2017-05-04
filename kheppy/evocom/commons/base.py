from abc import ABC
import numpy as np

from kheppy.core import Simulation
from kheppy.utils import Reporter


class BaseAlgorithm(ABC):

    def __init__(self):
        self.params = {}
        self.main_params()
        self.eval_params(model=None, fitness_func=None)
        self.sim_params(wd_path=None, robot_id=None)
        self.early_stopping(np.inf)
        self.reporter = Reporter(['max', 'avg', 'min', 'start_pos'])
        self.best = None

    def main_params(self, pop_size=100, epochs=100, param_init_limits=(-1, 1)):
        self.params['pop_size'] = pop_size
        self.params['epochs'] = epochs
        self.params['param_init'] = param_init_limits
        return self

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
