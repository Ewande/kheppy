from kheppy.evocom.commons.nn import NeuralNet
from kheppy.evocom.ga import GeneticAlgorithm
import numpy as np


def avoid_collision(sensors, left_motor, right_motor):
    speed_factor = (abs(left_motor) + abs(right_motor)) / 2
    movement_factor = 1 - np.sqrt(abs(left_motor - right_motor) / 2)
    proximity_factor = 1 - np.sqrt(max(sensors))
    return speed_factor * movement_factor * proximity_factor


if __name__ == '__main__':
    model = NeuralNet(8).add_layer(30, 'relu').add_layer(2, 'tanh')
    ga = GeneticAlgorithm().eval_params(model, avoid_collision).sim_params('worlds/circles.wd', 1, 5).evo_params(epochs=5)
    ga.run('/home/user/kheppy_results/', verbose=True)
