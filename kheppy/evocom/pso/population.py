import numpy as np
import math

from kheppy.evocom.commons.population import Population
from kheppy.evocom.pso.individual import ControllerPSO


class PopulationPSO(Population):

    def __init__(self, network, pop_list):
        super().__init__(network, pop_list)
        self.global_best = None
        self.limits = (0, 0)

    def initialize(self, init_limits):
        self.limits = init_limits
        self.pop = []
        for _ in range(self.pop_size):
            weights = self.network.random_weights_list(init_limits)
            biases = self.network.random_biases_list(init_limits)
            velocities = [None, None]
            velocities[0] = self.network.random_weights_list(init_limits)
            velocities[1] = self.network.random_biases_list(init_limits)
            self.pop.append(ControllerPSO(weights, biases, velocities))
        return self

    def update_global_best(self):
        max_ind = np.argmax([controller.local_best.fitness for controller in self.pop])
        self.global_best = self.pop[max_ind].local_best.copy()

    def update_local_best(self):
        for controller in self.pop:
            controller.update_local_best()

    def move_particles(self, inertia, cognitive_param, social_param):
        for controller in self.pop:
            rnd_lws = self.network.random_weights_list((0, 1))
            rnd_gws = self.network.random_weights_list((0, 1))
            for i in range(len(controller.weights)):
                cwv = controller.velocities[0][i]
                cw = controller.weights[i]
                lw = controller.local_best.weights[i]
                gw = self.global_best.weights[i]
                nwv = inertia * cwv + rnd_lws[i] * cognitive_param * (lw - cw) + rnd_gws[i] * social_param * (gw - cw)
                controller.velocities[0][i] = nwv
                controller.velocities[0][i] = np.clip(controller.velocities[0][i], 2 * self.limits[0], 2 * self.limits[1])
                controller.weights[i] += controller.velocities[0][i]
                controller.weights[i] = np.clip(controller.weights[i], self.limits[0], self.limits[1])

            rnd_lbs = self.network.random_biases_list((0, 1))
            rnd_gbs = self.network.random_biases_list((0, 1))
            for i in range(len(controller.biases)):
                cbv = controller.velocities[1][i]
                cb = controller.biases[i]
                lb = controller.local_best.biases[i]
                gb = self.global_best.biases[i]
                nbv = inertia * cbv + rnd_lbs[i] * cognitive_param * (lb - cb) + rnd_gbs[i] * social_param * (gb - cb)
                controller.velocities[1][i] = nbv
                controller.velocities[1][i] = np.clip(controller.velocities[1][i], 2 * self.limits[0], 2 * self.limits[1])
                controller.biases[i] += controller.velocities[1][i]
                controller.biases[i] = np.clip(controller.biases[i], self.limits[0], self.limits[1])

    def local_bests(self):
        return [controller.local_best for controller in self.pop]

    def best(self):
        return self.global_best
