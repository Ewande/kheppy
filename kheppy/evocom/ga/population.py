from multiprocessing.pool import Pool

from kheppy.evocom.ga.individual import Controller
from numpy.random import randint, uniform, shuffle
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


class Population:

    def __init__(self, network, pop_size=None, init_limits=None, pop_list=None):
        self.network = network

        if pop_list is None:
            self.pop = []
            for _ in range(pop_size):
                weights = self.network.random_weights_list(init_limits)
                biases = self.network.random_biases_list(init_limits)
                self.pop.append(Controller(weights, biases))
        else:
            self.pop = pop_list

        self.pop_size = len(self.pop)

    def cross(self, prob):
        shuffle(self.pop)
        for i in range(0, len(self.pop), 2):
            if uniform() < prob:
                c1_new, c2_new = Controller.cross(self.pop[i], self.pop[i + 1])
                self.pop.append(c1_new)
                self.pop.append(c2_new)

    def mutate(self, prob):
        for controller in self.pop:
            controller.mutate(prob)

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

    def select(self, sel_type):
        if isinstance(sel_type, int):
            new_pop = []
            for _ in range(self.pop_size):
                group = randint(0, len(self.pop), sel_type).tolist()
                max_ind = np.argmax([self.pop[i].fitness for i in group])
                best = self.pop[group[max_ind]]
                new_pop.append(best.copy())
        else:
            cum_fit = np.cumsum([elem.fitness for elem in self.pop])
            draws = np.random.uniform(0, cum_fit[-1], self.pop_size)
            indices = np.searchsorted(cum_fit, draws)
            new_pop = [self.pop[ind].copy() for ind in indices]

        return Population(self.network, pop_list=new_pop)

    def best(self):
        max_ind = np.argmax([controller.fitness for controller in self.pop])
        return self.pop[max_ind]

    def worst(self):
        min_ind = np.argmin([controller.fitness for controller in self.pop])
        return self.pop[min_ind]

    def average_fitness(self):
        return np.mean([controller.fitness for controller in self.pop])
