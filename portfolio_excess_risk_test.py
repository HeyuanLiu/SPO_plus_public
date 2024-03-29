import pandas as pd
import numpy as np
from tools import test_tools

model_params = {
    'n_features': 5,
    'n_samples': 6400,
    'dim_cost': 50,
}

data_gen_model = 'shortest_path'

if data_gen_model == 'shortest_path':
    data_params = {
        'deg': [6],
        # 'eps_bar': [0.5],
        'eps_bar': [0., 0.5],
    }
else:
    raise Exception('Data generation error.')

test_params = {
    'test_size': 10000,
    'n_trails': 20,
}

loss_list = ['spo', 'spop', 'l2', 'l1']
# pred_model_list = ['linear']
pred_model_list = ['two_layers']
# loss_list = ['spop']
suffix = '036'
test_results = test_tools.portfolio_model_excess_risk_test(
    model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=True, data_gen_model=data_gen_model)
test_results.to_csv('results/portfolio_excess_risk_results_' + suffix + '_' + str(np.random.randint(1e6)) + '.csv')
