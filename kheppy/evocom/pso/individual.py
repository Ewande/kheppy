from kheppy.evocom.commons import Controller


class ControllerPSO(Controller):

    def __init__(self, weights=None, biases=None, velocities=None, fitness=0):
        super().__init__(weights, biases, fitness)
        self.velocities = velocities
        self.local_best = None

    def update_local_best(self):
        if self.local_best is None or self.local_best.fitness < self.fitness:
            self.local_best = self.copy()

    def copy(self):
        return ControllerPSO(self.weights, self.biases, fitness=self.fitness)
