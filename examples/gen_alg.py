import numpy as np

from kheppy.evocom.commons.nn import NeuralNet
from kheppy.evocom.ga import GeneticAlgorithm
from kheppy.utils.fitfunc import avoid_collision


if __name__ == '__main__':
    model = NeuralNet(8).add_layer(30, 'relu').add_layer(2, 'tanh')
    # params = [4]
    # for p in params:
    res = []
    rep = GeneticAlgorithm().reporter
    for i in range(10, 30):
        ga = GeneticAlgorithm()
        ga.reporter = rep
        ga.sim_params('worlds/circle.wd', 1, 5)
        ga.eval_params(model, avoid_collision, num_positions=1)
        ga.evo_params(pop_size=150, p_mut=0.02, p_cross=0.75, epochs=1, param_init_limits=(-3, 3))
        ga.sel_params(sel_type=6)
        ga.run(num_proc=2, seed=i, verbose=True)
        test_res = ga.test(num_points=1)
        res.append(test_res)
        if i != 29:
            rep.move_to_new_series()
    rep.save('/Users/augoff/Dropbox/QA/private-research/master_thesis/results/1/static/all_seeds.rep')
    print('[' + ', '.join(['{:.4f}'.format(x) for x in res]) + ']')
    print('avg test fitness = {:.4f}'.format(np.mean(res)))
