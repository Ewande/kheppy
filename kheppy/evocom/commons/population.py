from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
import numpy as np

_PARALLEL_CONTEXT = None


def evaluate_controller(elem):
    time = 0
    controller, sims = _PARALLEL_CONTEXT[6][elem]
    controller.reset_fitness()
    for sim in sims:
        time += controller.evaluate(sim, _PARALLEL_CONTEXT[0], _PARALLEL_CONTEXT[1], _PARALLEL_CONTEXT[2],
                                    _PARALLEL_CONTEXT[3], _PARALLEL_CONTEXT[4], _PARALLEL_CONTEXT[5])
    controller.fitness /= len(sims)
    return time, elem, controller.fitness


class Population(ABC):

    def __init__(self, network, pop_list):
        self.network = network

        if isinstance(pop_list, int):
            self.pop = [None] * pop_list
        else:
            self.pop = list(pop_list)
        self.pop_size = len(self.pop)

    @abstractmethod
    def initialize(self, init_limits):
        pass

    def evaluate(self, sim_list, num_cycles, steps_per_cycle, max_speed, eval_func, aggregate_func, num_proc):
        total_sim_time = 0
        global _PARALLEL_CONTEXT
        _PARALLEL_CONTEXT = (self.network, num_cycles, steps_per_cycle, max_speed, eval_func, aggregate_func,
                             list(zip(self.pop, sim_list[:len(self.pop)])))

        if num_proc == 1:
            results = [evaluate_controller(i) for i in range(len(self.pop))]
        else:
            pool = Pool(processes=num_proc)
            results = pool.map(evaluate_controller, range(len(self.pop)),
                               chunksize=max(1, int(len(self.pop) / num_proc)))
            pool.close()
            pool.join()

        for sim_time, ind, fitness in results:
            self.pop[ind].fitness = fitness
            total_sim_time += sim_time

        return total_sim_time / num_proc

    def best(self):
        max_ind = np.argmax([controller.fitness for controller in self.pop])
        return self.pop[max_ind]

    def worst(self):
        min_ind = np.argmin([controller.fitness for controller in self.pop])
        return self.pop[min_ind]

    def average_fitness(self):
        return np.mean([controller.fitness for controller in self.pop])
