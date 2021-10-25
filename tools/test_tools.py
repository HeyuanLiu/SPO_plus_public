import itertools

import numpy as np
import torch
from torch import nn
import pandas as pd
from itertools import permutations

from tools import loss_func_tools
from tools import data_generation_tools
from tools import spo_framework
from tools import prediction_tools
from tools import optimization_oracle_tools
from tools import optim_tools


def portfolio_model_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                         data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples = model_params['n_samples']
    dim_cost = model_params['dim_cost']

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
    }

    optimization_params = {'r': np.log(dim_cost) - np.log(dim_cost - 0.9)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'i', 'n_samples', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    for param_value in itertools.product(*data_param_value, range(n_trails)):
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value):
            param[name] = value

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param)
        actions_hindsight, _ = optimization_oracle_tools.entropy_oracle(y_test, optimization_params, False)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param,
            model_coef=model_coef)

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                if pred_model == 'two_layers' and loss_func == 'l2':
                    lr = 0.01
                elif pred_model == 'linear':
                    if loss_func == 'spo':
                        lr = 1.
                    else:
                        lr = 0.1
                else:
                    lr = 1.
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': pred_model_map[pred_model],
                    'model_params': _clone_params(initial_params),
                    'predict_model_back': pred_model_back_map[pred_model],
                    'optimization_oracle': optimization_oracle_tools.entropy_oracle,
                    'optimization_params': optimization_params,
                    'optimization_oracle_back': optimization_oracle_tools.entropy_oracle_back,
                    'loss_func': loss_map[loss_func],
                    'optimizer': optim_tools.adam,
                    # 'optimizer': optim_tools.sgd_momentum,
                    # Notes:
                    # SPO, teo layers: lr = 1.0
                    # 'optimizer_config': {'learning_rate': lr, 'momentum': 0.9, 'lr_decay': 0.995},
                    'require_grad': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=20000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo,
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo,
                    ]

    return test_results


def portfolio_model_excess_risk_test(model_params, data_params, test_params, loss_list, pred_model_list,
                                     if_test_ini=False, data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples = model_params['n_samples']
    dim_cost = model_params['dim_cost']

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
    }

    optimization_params = {'r': np.log(dim_cost) - np.log(dim_cost - 0.9)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'i', 'n_samples', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline', 'normal_mean_spo_loss'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    for param_value in itertools.product(*data_param_value, range(n_trails)):
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value):
            param[name] = value

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param)
        actions_hindsight, _ = optimization_oracle_tools.entropy_oracle(y_test, optimization_params, False)
        y_mean = model_coef['c_mean'].detach().clone()
        acction_y_mean, _ = optimization_oracle_tools.entropy_oracle(y_mean, optimization_params, False)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param,
            model_coef=model_coef)
        flag_mean_spo_loss = True

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                if pred_model == 'two_layers' and loss_func == 'l2':
                    lr = 0.01
                elif pred_model == 'linear':
                    if loss_func == 'spo':
                        lr = 1.
                    else:
                        lr = 0.1
                else:
                    lr = 1.
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': pred_model_map[pred_model],
                    'model_params': _clone_params(initial_params),
                    'predict_model_back': pred_model_back_map[pred_model],
                    'optimization_oracle': optimization_oracle_tools.entropy_oracle,
                    'optimization_params': optimization_params,
                    'optimization_oracle_back': optimization_oracle_tools.entropy_oracle_back,
                    'loss_func': loss_map[loss_func],
                    'optimizer': optim_tools.adam,
                    # 'optimizer': optim_tools.sgd_momentum,
                    # Notes:
                    # SPO, teo layers: lr = 1.0
                    # 'optimizer_config': {'learning_rate': lr, 'momentum': 0.9, 'lr_decay': 0.995},
                    'require_grad': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=20000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                              'cost_mean': y_mean, 'action_cost_mean': acction_y_mean, },
                    if_test_ini=if_test_ini and (j == 0), if_mean_spo_loss=flag_mean_spo_loss,
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if flag_mean_spo_loss:
                    loss_mean = loss['loss_mean']
                    normal_spo_loss_mean = loss_mean / hindsight
                    flag_mean_spo_loss = False

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, normal_spo_loss_mean,
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, normal_spo_loss_mean,
                    ]

    return test_results


def portfolio_argmax_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                          data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    hidden_dim = model_params.get('hidden_dim', 128)
    minmax = model_params.get('min/max', 'max')

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map = {
        'spop': loss_func_tools.spop_argmax_loss_func,
        # 'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    def pred_model_map(pred_model):
        if pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
    }

    baseline_action = torch.ones(dim_cost) / dim_cost
    optimization_params = {'const': None}

    test_results = pd.DataFrame(columns=data_param_name + [
        'i', 'n_samples', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline'])

    for param_value in itertools.product(*data_param_value, range(n_trails)):
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value):
            param[name] = value

        ################################
        # Something new here about neg #
        ################################
        neg = minmax == 'max'
        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.softmax_oracle(y_test, optimization_params, False)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param,
            model_coef=model_coef, neg=neg)

        for pred_model in pred_model_list:
            for j, loss_func in enumerate(loss_list):
                predict_model = pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.softmax_oracle,
                    'optimization_params': optimization_params,
                    'loss_func': loss_map[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': 64,
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=10000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight)
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo,
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo,
                    ]

    return test_results


def barrier_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                 data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples = model_params['n_samples']
    dim_cost = model_params['dim_cost']

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
        'multi_class': data_generation_tools.multi_class_data,
    }

    optimization_params = {'r': 2 * dim_cost * np.log(dim_cost)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost, dtype=torch.float64) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'i', 'n_samples', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    for param_value in itertools.product(*data_param_value, range(n_trails)):
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value):
            param[name] = value

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=False)
        actions_hindsight, _ = optimization_oracle_tools.barrier_oracle(y_test, optimization_params, False)
        argmin_hindsight = y_test.argmin(dim=1, keepdim=True)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param, model_coef=model_coef, neg=False)

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float64')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float64'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float64')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float64')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float64')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float64')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                if pred_model == 'two_layers' and loss_func == 'l2':
                    lr = 0.01
                elif pred_model == 'linear':
                    if loss_func == 'spo':
                        lr = 1.
                    else:
                        lr = 0.1
                else:
                    lr = 1.
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': pred_model_map[pred_model],
                    'model_params': _clone_params(initial_params),
                    'predict_model_back': pred_model_back_map[pred_model],
                    'optimization_oracle': optimization_oracle_tools.barrier_oracle,
                    'optimization_params': optimization_params,
                    'test_optimization_oracle': optimization_oracle_tools.argmin_test,
                    'test_optimization_params': {'arg': 'min'},
                    'optimization_oracle_back': optimization_oracle_tools.barrier_oracle_back,
                    'loss_func': loss_map[loss_func],
                    'optimizer': optim_tools.adam,
                    # 'optimizer': optim_tools.sgd_momentum,
                    # Notes:
                    # SPO, teo layers: lr = 1.0
                    'optimizer_config': {'learning_rate': 0.1, 'lr_decay': 0.99},
                    'require_grad': True,
                    'if_argmax': True,
                    'minibatch_size': 8,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=3000, if_quiet=False,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                              'argmin_hindsight': argmin_hindsight,
                              },
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, 'test spo loss:', loss_test, 'best cost in hindsight', hindsight)
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo,
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo,
                    ]

    return test_results


def shortest_path_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                       data_gen_model='shortest_path'):
    n_features = model_params['n_features']
    n_samples = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    hidden_dim = model_params.get('hidden_dim', 128)
    grid_dim = model_params.get('grid_dim', 4)
    assert dim_cost == 2 * grid_dim * (grid_dim - 1), 'cost dim doesnot match grid dim!'
    min_max = model_params.get('min_max', 'min')

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map = {
        'spop': loss_func_tools.spop_loss_func,
        # 'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    def pred_model_map(_pred_model):
        if _pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif _pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
    }

    baseline_action = torch.zeros(dim_cost)
    # optimization_params = {'const': None}

    test_results = pd.DataFrame(columns=data_param_name + [
        'i', 'n_samples', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline'])

    def _path_decoding(_grid_dim, path_encoded):
        loc_x, loc_y = 0, 0
        num_edges = _grid_dim * (_grid_dim - 1)
        path_decoded = np.zeros(2 * num_edges)
        for direction in path_encoded:
            if direction:
                path_decoded[1 * loc_x + (_grid_dim - 1) * loc_y + num_edges] = 1
                loc_x += 1
            else:
                path_decoded[(_grid_dim - 1) * loc_x + 1 * loc_y] = 1
                loc_y += 1
        return path_decoded

    def _construct_grid_path(_grid_dim):
        assert _grid_dim >= 2, 'Grid dim at least 2!'
        path_0 = [0] * (_grid_dim - 1) + [1] * (_grid_dim - 1)
        paths_encoded = list(set(permutations(path_0)))

        paths = []
        for path_encoded in paths_encoded:
            paths.append(_path_decoding(_grid_dim, path_encoded))

        paths = np.array(paths, dtype='float32')

        return torch.from_numpy(paths)

    optimization_params = {
        'paths': _construct_grid_path(grid_dim),
        'min_max': min_max,
    }

    for param_value in itertools.product(*data_param_value, range(n_trails)):
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value):
            param[name] = value

        ################################
        # Something new here about neg #
        ################################
        neg = min_max == 'max'
        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.shortest_path_oracle(y_test, optimization_params, False)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param,
            model_coef=model_coef, neg=neg)

        for pred_model in pred_model_list:
            print(pred_model)
            for j, loss_func in enumerate(loss_list):
                predict_model = pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.shortest_path_oracle,
                    'optimization_params': optimization_params,
                    'loss_func': loss_map[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': 64,
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=10000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, 'test spo loss:', loss_test, 'best cost in hindsight', hindsight)
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo,
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        n_samples, loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo,
                    ]

    return test_results


def barrier_vs_argmin_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                           data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples_list = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    neg = model_params.get('neg', False)

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map_barrier = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    loss_map_argmin = {
        'spop': loss_func_tools.spop_argmax_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
        'multi_class': data_generation_tools.multi_class_data,
    }

    optimization_params = {'r': 2 * dim_cost * np.log(dim_cost)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'n_samples', 'i', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline', 'type'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    def _pred_model_map(_pred_model):
        if _pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif _pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    for param_value_tuple in itertools.product(*data_param_value, n_samples_list, range(n_trails)):
        param_value = list(param_value_tuple)
        n_samples = param_value[-2]
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value[:-2]):
            param[name] = value
        print(param, param_value)

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.barrier_oracle(y_test, optimization_params, False)
        argmin_hindsight = y_test.argmin(dim=1, keepdim=True)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param, model_coef=model_coef, neg=neg)

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': pred_model_map[pred_model],
                    'model_params': _clone_params(initial_params),
                    'predict_model_back': pred_model_back_map[pred_model],
                    'optimization_oracle': optimization_oracle_tools.barrier_oracle,
                    'optimization_params': optimization_params,
                    'test_optimization_oracle': optimization_oracle_tools.argmin_test,
                    'test_optimization_params': {'arg': 'min'},
                    'optimization_oracle_back': optimization_oracle_tools.barrier_oracle_back,
                    'loss_func': loss_map_barrier[loss_func],
                    'optimizer': optim_tools.adam,
                    # 'optimizer': optim_tools.sgd_momentum,
                    # Notes:
                    # SPO, teo layers: lr = 1.0
                    'optimizer_config': {'learning_rate': 0.1, 'lr_decay': 0.999},
                    'require_grad': True,
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=5000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                              'argmin_hindsight': argmin_hindsight,
                              },
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'barrier')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, 'barrier',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, 'barrier',
                    ]

                print('argmin start.')
                predict_model = _pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.softmax_oracle,
                    'optimization_params': {'const': None},
                    'loss_func': loss_map_argmin[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': min(64, n_samples),
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=10000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': argmin_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'argmin')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, 'argmin',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, 'argmin',
                    ]

    return test_results


def barrier_vs_argmin_excess_risk_test(model_params, data_params, test_params, loss_list, pred_model_list,
                                       if_test_ini=False, data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples_list = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    neg = model_params.get('neg', False)

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map_barrier = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    loss_map_argmin = {
        'spop': loss_func_tools.spop_argmax_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
        'multi_class': data_generation_tools.multi_class_data,
    }

    optimization_params = {'r': 2 * dim_cost * np.log(dim_cost)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'n_samples', 'i', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline', 'normal_mean_spo_loss', 'type'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    def _pred_model_map(_pred_model):
        if _pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif _pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    for param_value_tuple in itertools.product(*data_param_value, n_samples_list, range(n_trails)):
        param_value = list(param_value_tuple)
        n_samples = param_value[-2]
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value[:-2]):
            param[name] = value
        print(param, param_value)

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.barrier_oracle(y_test, optimization_params, False)
        argmin_hindsight = y_test.argmin(dim=1, keepdim=True)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param, model_coef=model_coef, neg=neg)

        y_mean = model_coef['c_mean'].detach().clone()
        action_y_mean, _ = optimization_oracle_tools.barrier_oracle(y_mean, optimization_params, False)
        argmin_hindsight_ymean = y_mean.argmin(dim=1, keepdim=True)
        flag_mean_spo_loss = True

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                # spo_model = spo_framework.SpoTest({
                #     'n_features': n_features,
                #     'dim_cost': dim_cost,
                #     'baseline_action': baseline_action,
                #     'predict_model': pred_model_map[pred_model],
                #     'model_params': _clone_params(initial_params),
                #     'predict_model_back': pred_model_back_map[pred_model],
                #     'optimization_oracle': optimization_oracle_tools.barrier_oracle,
                #     'optimization_params': optimization_params,
                #     'test_optimization_oracle': optimization_oracle_tools.argmin_test,
                #     'test_optimization_params': {'arg': 'min'},
                #     'optimization_oracle_back': optimization_oracle_tools.barrier_oracle_back,
                #     'loss_func': loss_map_barrier[loss_func],
                #     'optimizer': optim_tools.adam,
                #     # 'optimizer': optim_tools.sgd_momentum,
                #     # Notes:
                #     # SPO, teo layers: lr = 1.0
                #     'optimizer_config': {'learning_rate': 0.1, 'lr_decay': 0.999},
                #     'require_grad': True,
                #     'if_argmax': True,
                # })
                #
                # loss = spo_model.update(
                #     x_input, y_input, num_iter=20000, if_quiet=True,
                #     test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                #               'argmin_hindsight': argmin_hindsight, 'cost_mean': y_mean,
                #               'action_cost_mean': action_y_mean, 'argmin_hindsight_ymean': argmin_hindsight_ymean,
                #               },
                #     if_test_ini=if_test_ini and (j == 0),
                # )
                #
                # loss_test = loss['loss_spo_test']
                # hindsight = loss['hindsight']
                # print(loss_func, pred_model, loss_test, hindsight, 'barrier')
                # normal_spo = loss_test / hindsight
                # train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                # if loss['loss_spo_baseline'] is not None:
                #     baseline_spo = loss['loss_spo_baseline'] / hindsight
                # else:
                #     baseline_spo = None
                #
                # if if_test_ini:
                #     if j == 0:
                #         loss_ini = loss['loss_spo_test_ini']
                #         hind_ini = loss['hindsight_ini']
                #         spo_ini = loss_ini / hind_ini
                #     test_results.loc[len(test_results.index)] = list(param_value) + [
                #         loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                #         baseline_spo, 'barrier',
                #     ]
                # else:
                #     test_results.loc[len(test_results.index)] = list(param_value) + [
                #         loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                #         baseline_spo, 'barrier',
                #     ]

                print('argmin start.')
                predict_model = _pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.softmax_oracle,
                    'optimization_params': {'const': None},
                    'loss_func': loss_map_argmin[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': min(64, n_samples),
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=20000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': argmin_hindsight,
                              'cost_mean': y_mean, 'action_cost_mean': argmin_hindsight_ymean, },
                    if_test_ini=if_test_ini and (j == 0), if_mean_spo_loss=flag_mean_spo_loss,
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'argmin')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                ########## New ###############
                if flag_mean_spo_loss:
                    loss_mean = loss['loss_mean']
                    normal_spo_loss_mean = loss_mean / hindsight
                    flag_mean_spo_loss = False
                ########## New ###############

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, normal_spo_loss_mean, 'argmin',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, normal_spo_loss_mean, 'argmin',
                    ]

    return test_results


def entropy_vs_argmin_test(model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
                           data_gen_model='portfolio'):
    n_features = model_params['n_features']
    n_samples_list = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    neg = model_params.get('neg', False)

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map_barrier = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    loss_map_argmin = {
        'spop': loss_func_tools.spop_argmax_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
        'multi_class': data_generation_tools.multi_class_data,
    }

    optimization_params = {'r': 2 * dim_cost * np.log(dim_cost)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'n_samples', 'i', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline', 'type'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    def _pred_model_map(_pred_model):
        if _pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif _pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    for param_value_tuple in itertools.product(*data_param_value, n_samples_list, range(n_trails)):
        param_value = list(param_value_tuple)
        n_samples = param_value[-2]
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value[:-2]):
            param[name] = value
        print(param, param_value)

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.barrier_oracle(y_test, optimization_params, False)
        argmin_hindsight = y_test.argmin(dim=1, keepdim=True)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param, model_coef=model_coef, neg=neg)

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': pred_model_map[pred_model],
                    'model_params': _clone_params(initial_params),
                    'predict_model_back': pred_model_back_map[pred_model],
                    'optimization_oracle': optimization_oracle_tools.barrier_oracle,
                    'optimization_params': optimization_params,
                    'test_optimization_oracle': optimization_oracle_tools.argmin_test,
                    'test_optimization_params': {'arg': 'min'},
                    'optimization_oracle_back': optimization_oracle_tools.barrier_oracle_back,
                    'loss_func': loss_map_barrier[loss_func],
                    'optimizer': optim_tools.adam,
                    # 'optimizer': optim_tools.sgd_momentum,
                    # Notes:
                    # SPO, teo layers: lr = 1.0
                    'optimizer_config': {'learning_rate': 0.1, 'lr_decay': 0.999},
                    'require_grad': True,
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=5000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                              'argmin_hindsight': argmin_hindsight,
                              },
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'barrier')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, 'barrier',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, 'barrier',
                    ]

                print('argmin start.')
                predict_model = _pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.softmax_oracle,
                    'optimization_params': {'const': None},
                    'loss_func': loss_map_argmin[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': min(64, n_samples),
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=10000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': argmin_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'argmin')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, 'argmin',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, 'argmin',
                    ]

    return test_results


def multi_class_barrier_vs_argmin_test(
        model_params, data_params, test_params, loss_list, pred_model_list, if_test_ini=False,
        data_gen_model='portfolio',
):
    n_features = model_params['n_features']
    n_samples_list = model_params['n_samples']
    dim_cost = model_params['dim_cost']
    neg = model_params.get('neg', False)

    # deg_list = data_params['deg']
    # tau_list = data_params['tau']
    # n_factors_list = data_params['n_factors']

    data_param_name, data_param_value = [], []
    for param_name in data_params:
        data_param_name.append(param_name)
        data_param_value.append(data_params[param_name])

    test_set_size = test_params['test_size']
    n_trails = test_params['n_trails']

    loss_map_barrier = {
        'spop': loss_func_tools.spop_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    loss_map_argmin = {
        'spop': loss_func_tools.spop_argmax_loss_func,
        'spo': loss_func_tools.spo_loss_func,
        'l2': loss_func_tools.mse_loss_func,
        'l1': loss_func_tools.abs_loss_func,
    }

    pred_model_map = {
        'linear': prediction_tools.linear_prediction_model,
        'two_layers': prediction_tools.two_layers_model,
    }

    pred_model_back_map = {
        'linear': prediction_tools.linear_prediction_model_back,
        'two_layers': prediction_tools.two_layers_model_back,
    }

    data_gen_map = {
        'portfolio': data_generation_tools.portfolio_data,
        'shortest_path': data_generation_tools.shortest_path_data,
        'multi_class': data_generation_tools.multi_class_data,
    }

    optimization_params = {'r': 2 * dim_cost * np.log(dim_cost)}
    # optimization_params = {'r': np.log(dim_cost) / 2}
    baseline_action = torch.ones(dim_cost) / dim_cost

    test_results = pd.DataFrame(columns=data_param_name + [
        'n_samples', 'i', 'surrogate_loss_func', 'pred_model', 'normalized_spo_loss', 'hindsight',
        'train_normal_spo', 'normalized_spo_ini', 'normal_spo_baseline', 'type'])

    def _clone_params(num_params):
        num_params_copy = {}
        for num_param in num_params:
            num_params_copy[num_param] = num_params[num_param].detach().clone()
        return num_params_copy

    def _pred_model_map(_pred_model):
        if _pred_model == 'linear':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=dim_cost),
            )
        elif _pred_model == 'two_layers':
            return nn.Sequential(
                nn.Linear(in_features=n_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=dim_cost),
            )
        else:
            raise Exception('Prediction Model Type Error!')

    for param_value_tuple in itertools.product(*data_param_value, n_samples_list, range(n_trails)):
        param_value = list(param_value_tuple)
        n_samples = param_value[-2]
        if param_value[-1] == 0:
            print(param_value)
        param = {}
        for name, value in zip(data_param_name, param_value[:-2]):
            param[name] = value
        print(param, param_value)

        x_test, y_test, model_coef = data_gen_map[data_gen_model](
            n_features, test_set_size, dim_cost, param, neg=neg)
        actions_hindsight, _ = optimization_oracle_tools.barrier_oracle(y_test, optimization_params, False)
        argmin_hindsight = y_test.argmin(dim=1, keepdim=True)
        x_input, y_input, _ = data_gen_map[data_gen_model](
            n_features, n_samples, dim_cost, param, model_coef=model_coef, neg=neg)

        for pred_model in pred_model_list:
            if pred_model == 'linear':
                initial_params = {
                    'W': torch.from_numpy(np.random.normal(size=(n_features, dim_cost)).astype('float32')),
                    'b': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32'))
                }
            elif pred_model == 'two_layers':
                hidden_dim = model_params.get('hidden_dim', 256)
                initial_params = {
                    'W1': torch.from_numpy(
                        (np.random.normal(size=(n_features, hidden_dim)) / np.sqrt(hidden_dim)).astype('float32')),
                    'W2': torch.from_numpy(
                        (np.random.normal(size=(hidden_dim, dim_cost)) / np.sqrt(dim_cost)).astype('float32')),
                    'b1': torch.from_numpy(np.random.normal(size=hidden_dim).astype('float32')),
                    'b2': torch.from_numpy(np.random.normal(size=dim_cost).astype('float32')),
                }
            else:
                raise Exception(
                    'Prediction model can only be "linear" or "two_layers". The input is: ' + pred_model)

            for j, loss_func in enumerate(loss_list):
                if loss_func == 'spop':
                    spo_model = spo_framework.SpoTest({
                        'n_features': n_features,
                        'dim_cost': dim_cost,
                        'baseline_action': baseline_action,
                        'predict_model': pred_model_map[pred_model],
                        'model_params': _clone_params(initial_params),
                        'predict_model_back': pred_model_back_map[pred_model],
                        'optimization_oracle': optimization_oracle_tools.barrier_oracle,
                        'optimization_params': optimization_params,
                        'test_optimization_oracle': optimization_oracle_tools.argmin_test,
                        'test_optimization_params': {'arg': 'min'},
                        'optimization_oracle_back': optimization_oracle_tools.barrier_oracle_back,
                        'loss_func': loss_map_barrier[loss_func],
                        'optimizer': optim_tools.adam,
                        # 'optimizer': optim_tools.sgd_momentum,
                        # Notes:
                        # SPO, teo layers: lr = 1.0
                        'optimizer_config': {'learning_rate': 0.1, 'lr_decay': 0.999},
                        'require_grad': True,
                        'if_argmax': True,
                    })

                    loss = spo_model.update(
                        x_input, y_input, num_iter=5000, if_quiet=True,
                        test_set={
                            'features': x_test, 'cost_real': y_test, 'action_hindsight': actions_hindsight,
                            'argmin_hindsight': argmin_hindsight,
                        },
                        if_test_ini=if_test_ini and (j == 0),
                    )

                    loss_test = loss['loss_spo_test']
                    hindsight = loss['hindsight']
                    print(loss_func, pred_model, loss_test, hindsight, 'barrier')
                    normal_spo = loss_test / hindsight
                    train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                    if loss['loss_spo_baseline'] is not None:
                        baseline_spo = loss['loss_spo_baseline'] / hindsight
                    else:
                        baseline_spo = None

                    if if_test_ini:
                        if j == 0:
                            loss_ini = loss['loss_spo_test_ini']
                            hind_ini = loss['hindsight_ini']
                            spo_ini = loss_ini / hind_ini
                        test_results.loc[len(test_results.index)] = list(param_value) + [
                            loss_func + 'barrier', pred_model, normal_spo, hindsight, train_spo, spo_ini,
                            baseline_spo, 'barrier',
                        ]
                    else:
                        test_results.loc[len(test_results.index)] = list(param_value) + [
                            loss_func + 'barrier', pred_model, normal_spo, hindsight, train_spo, None,
                            baseline_spo, 'barrier',
                        ]

                print('argmin start.')
                predict_model = _pred_model_map(pred_model)
                spo_model = spo_framework.SpoTest({
                    'n_features': n_features,
                    'dim_cost': dim_cost,
                    'baseline_action': baseline_action,
                    'predict_model': predict_model,
                    'optimization_oracle': optimization_oracle_tools.softmax_oracle,
                    'optimization_params': {'const': None},
                    'loss_func': loss_map_argmin[loss_func],
                    'optimizer': torch.optim.Adam(predict_model.parameters()),
                    'require_grad': False,
                    'minibatch_size': min(64, n_samples),
                    'if_argmax': True,
                })

                loss = spo_model.update(
                    x_input, y_input, num_iter=10000, if_quiet=True,
                    test_set={'features': x_test, 'cost_real': y_test, 'action_hindsight': argmin_hindsight},
                    if_test_ini=if_test_ini and (j == 0),
                )

                loss_test = loss['loss_spo_test']
                hindsight = loss['hindsight']
                print(loss_func, pred_model, loss_test, hindsight, 'argmin')
                normal_spo = loss_test / hindsight
                train_spo = np.array(loss['loss_spo'][-100:-1]).mean() / hindsight
                if loss['loss_spo_baseline'] is not None:
                    baseline_spo = loss['loss_spo_baseline'] / hindsight
                else:
                    baseline_spo = None

                if if_test_ini:
                    if j == 0:
                        loss_ini = loss['loss_spo_test_ini']
                        hind_ini = loss['hindsight_ini']
                        spo_ini = loss_ini / hind_ini
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, spo_ini,
                        baseline_spo, 'argmin',
                    ]
                else:
                    test_results.loc[len(test_results.index)] = list(param_value) + [
                        loss_func, pred_model, normal_spo, hindsight, train_spo, None,
                        baseline_spo, 'argmin',
                    ]

    return test_results
