from kheppy.evocom.commons.nn import NeuralNet
from kheppy.evocom.ga import GeneticAlgorithm
from kheppy.utils.fitfunc import avoid_collision


if __name__ == '__main__':
    model = NeuralNet(8).add_layer(30, 'relu').add_layer(2, 'tanh')
    ga = GeneticAlgorithm()
    ga.eval_params(model, avoid_collision).sim_params('worlds/circle.wd', 1, 5).evo_params(epochs=5)
    ga.run('/home/user/kheppy_results/', verbose=True)
    ga.test(num_points=100, verbose=True)

