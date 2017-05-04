from abc import ABC, abstractmethod
import numpy as np
from timeit import default_timer as timer


class Controller(ABC):

    def __init__(self, weights=None, biases=None, fitness=0):
        self.weights = [np.array(layer_weights) for layer_weights in weights] if weights is not None else []
        self.biases = [np.array(layer_biases) for layer_biases in biases] if biases is not None else []

        self.fitness = fitness

    @abstractmethod
    def copy(self):
        pass

    def reset_fitness(self):
        self.fitness = 0

    def evaluate(self, simulation, model, num_cycles, steps_per_cycle, max_speed, eval_func, aggregate_func):
        fitness = []
        time = 0
        for i in range(num_cycles):
            left, right = model.predict(simulation.get_sensor_states(), self.weights, self.biases)
            simulation.set_robot_speed(left * max_speed, right * max_speed)
            start = timer()
            simulation.simulate(steps_per_cycle)
            time += timer() - start
            fitness.append(eval_func(simulation.get_sensor_states(), left, right))

        fitness = aggregate_func(fitness)
        self.fitness += fitness
        return time
