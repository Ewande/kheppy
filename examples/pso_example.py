from kheppy.evocom.commons.nn import NeuralNet
from kheppy.evocom.pso.pso import PartSwarmOpt
from kheppy.utils.fitfunc import avoid_collision


if __name__ == '__main__':
    model = NeuralNet(8).add_layer(30, 'relu').add_layer(2, 'tanh')
    pso = PartSwarmOpt()
    pso.eval_params(model, avoid_collision).sim_params('worlds/circle.wd', 1, 5).main_params(epochs=5)
    pso.run('/home/user/kheppy_results/', verbose=True)
    pso.test(num_points=100, verbose=True)
