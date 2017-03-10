from numpy.random import uniform, randint
import numpy as np
from timeit import default_timer as timer


class Controller:

    def __init__(self, weights=None, biases=None):
        self.weights = [np.array(layer_weights) for layer_weights in weights] if weights is not None else []
        self.biases = [np.array(layer_biases) for layer_biases in biases] if biases is not None else []

        self.fitness = 0

    def copy(self):
        controller = Controller(self.weights, self.biases)
        controller.fitness = self.fitness
        return controller

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

    def mutate(self, prob):
        self.weights = [(w + uniform(-0.05, 0.05, w.shape) * (uniform(0, 1, w.shape) > prob)) for w in self.weights]
        self.biases = [(b + uniform(-0.05, 0.05, b.shape) * (uniform(0, 1, b.shape) < prob)) for b in self.biases]

    @staticmethod
    def cross(c1, c2):
        c1_new = Controller()
        c2_new = Controller()
        for w1, w2, b1, b2 in zip(c1.weights, c2.weights, c1.biases, c2.biases):
            w1f, w2f, b1f, b2f = w1.flatten(), w2.flatten(), b1.flatten(), b2.flatten()

            half_w = randint(0, len(w1f))  # int(len(w1f)/2)
            half_b = randint(0, len(b1f))  # int(len(b1f)/2)
            w1_new = np.reshape(np.append(w1f[:half_w], w2f[half_w:]), w1.shape)
            w2_new = np.reshape(np.append(w2f[:half_w], w1f[half_w:]), w1.shape)
            b1_new = np.reshape(np.append(b1f[:half_b], b2f[half_b:]), b1.shape)
            b2_new = np.reshape(np.append(b2f[:half_b], b1f[half_b:]), b1.shape)
            c1_new.weights.append(w1_new)
            c2_new.weights.append(w2_new)
            c1_new.biases.append(b1_new)
            c2_new.biases.append(b2_new)

        return c1, c2
