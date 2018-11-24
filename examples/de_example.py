from pathlib import Path

from kheppy.evocom.commons import NeuralNet
from kheppy.evocom.de import DiffEvolution
from kheppy.utils.fitfunc import avoid_collision


if __name__ == '__main__':
    world_file = str(Path(__file__).parent / 'worlds/circle.wd')

    model = NeuralNet(8).add_layer(30, 'relu').add_layer(2, 'tanh')
    de = DiffEvolution()
    de.eval_params(model, avoid_collision).sim_params(world_file, 1, 5).main_params(max_epochs=5)
    de.de_params()  # set DE-specific parameters here
    de.run('/home/user/kheppy_results/', verbose=True)
    de.test(num_points=100, verbose=True)
