from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import numpy as np
import os
from timeit import default_timer as timer
from itertools import repeat

from kheppy.core import Simulation, SimList
from kheppy.utils import Reporter, timestamp


class BaseAlgorithm(ABC):

    def __init__(self):
        self.params = {}
        self.main_params()
        self.eval_params(model=None, fitness_func=None)
        self.sim_params(wd_path=None, robot_id=None)
        self.reporter = Reporter(['max', 'avg', 'min', 'ffe', 'start_pos'])
        self.best = None

    def main_params(self, pop_size=100, max_epochs=100, early_stop=None, max_ffe=None, param_init_limits=(-1, 1)):
        """Set evolution main parameters.
        
        Evolution finishes when at least one condition is met:
        - maximum number of epochs is reached,
        - maximum number of fitness function evaluations is reached (turned off by default),
        - number of epochs defined as early stopping is reached (turned off by default).
        
        :param pop_size: population size, int
        :param max_epochs: maximum number of epochs, int or None (turned off)
        :param early_stop: number of epochs without increase in best fitness value, int or None (turned off)
        :param max_ffe: maximum number of fitness function evaluations, int or None (turned off)
        :param param_init_limits: interval from which initial gene values are drawn, tuple of size 2
        
        :return: this object
        """
        if max_epochs is None and max_ffe is None:
            raise ValueError('Evolution will never end. Max_epochs and max_ffe cannot both be None.')

        self.params['pop_size'] = pop_size
        self.params['epochs'] = max_epochs if max_epochs is not None else np.inf
        self.params['stop'] = early_stop if early_stop is not None else self.params['epochs']
        self.params['ffe'] = max_ffe if max_ffe is not None else np.inf
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

        :return: this object
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

    def _prepare_positions(self, sim_list):
        if self.params['pos'] == 'dynamic':
            sim_list.shuffle_defaults()
        if self.params['pos'] == 'moving':
            sim_list.move_forward_defaults(self.params['move_step'], self.params['move_noise'])
        sim_list.reset_to_defaults()

    @abstractmethod
    def _get_init_pop(self):
        pass

    @abstractmethod
    def _get_next_pop(self, pop):
        pass

    def _evaluate_pop(self, pop):
        return pop.evaluate(self.params['sim_list'], self.params['num_cycles'], self.params['steps'],
                            self.params['max_speed'], self.params['fit_func'], self.params['agg_func'],
                            self.params['num_proc'])

    def run(self, output_dir=None, num_proc=1, seed=42, verbose=False):
        np.random.seed(seed)
        with SimList(self.params['wd_path'], 2 * self.params['pop_size'] + 1, self.params['num_sim'],
                     self.params['robot_id']) as sim_list:
            self.params['sim_list'] = sim_list
            self.params['num_proc'] = num_proc

            if verbose:
                print('Using {} simulation(s) per controller.'.format(self.params['num_sim']))
                print('Preparing population...')
            pop = self._get_init_pop()
            best, no_change, i, ffe = None, 0, 0, 0

            sim_list.shuffle_defaults(seed=seed)
            sim_list.reset_to_defaults()

            while i < self.params['epochs'] and no_change < self.params['stop'] and ffe <= self.params['ffe']:

                if verbose:
                    print('Epoch {:>3} '.format(i + 1), end='', flush=True)
                start = timer()
                pop, epoch_ffe, epoch_sim_time = self._get_next_pop(pop)
                ffe += epoch_ffe

                if verbose:
                    print('finished in {:>5.2f}s (simulation: {:>5.2f}s) | max fitness: {:.4f} | '
                          'average fitness: {:.4f} | min fitness: {:.4f}. Total FFE: {:>8}.'
                          .format(timer() - start, epoch_sim_time, pop.best().fitness, pop.average_fitness(),
                                  pop.worst().fitness, ffe))
                self.reporter.put(['max', 'avg', 'min', 'ffe', 'start_pos'],
                                  [pop.best().fitness, pop.average_fitness(), pop.worst().fitness, ffe,
                                  [sim.get_robot_position() for sim in sim_list.default_sims]])

                if best is not None and pop.best().fitness - best.fitness < 0.0001:
                    no_change += 1
                elif ffe <= self.params['ffe']:
                    best = pop.best().copy()
                    no_change = 0
                i += 1

                self._prepare_positions(sim_list)

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                self.params['model'].save('{}ga_final_{}.nn'.format(output_dir, timestamp()), best.weights, best.biases)
            if verbose:
                print('Evolution finished after {} iterations with total of {} FFE.'.format(i, ffe))
            self.best = best

    def _test(self, seed_offset, seed, num_points, num_cycles, controller):
        with Simulation(self.params['wd_path']) as sim:
            sim.set_controlled_robot(self.params['robot_id'])
            sim.set_seed(seed)
            for i in range(seed_offset):
                sim.move_robot_random()

            res = []
            for i in range(num_points):
                sim.move_robot_random()
                controller.reset_fitness()
                controller.evaluate(sim, self.params['model'], num_cycles, self.params['max_speed'],
                                    self.params['steps'], self.params['fit_func'], np.mean)
                res.append(controller.fitness)
            return res

    def test(self, seed=50, num_points=1000, num_cycles=160, controller=None, verbose=False):
        if controller is None:
            controller = self.best.copy()
        else:
            controller = controller.copy()

        if verbose:
            print('Testing using {} starting points. Single evaluation length = {} cycles.'
                  .format(num_points, num_cycles))

        res = []
        with Simulation(self.params['wd_path']) as sim:
            sim.set_controlled_robot(self.params['robot_id'])
            sim.set_seed(seed)
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

        return res
