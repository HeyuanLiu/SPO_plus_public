import torch
from torch import nn
import numpy as np


# Main SPO framework
class SpoTest(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.n_features = params['n_features']
        self.dim_cost = params['dim_cost']
        self.require_grad = params['require_grad']

        # Add baseline decision
        self.baseline_action = params.get('baseline_action', None)

        # Add prediction model f(.): x -> c
        self.predict_model = params['predict_model']

        # If no closed-form solution of the optimize part, we will need manually back-propagate the gradient
        if self.require_grad:
            def _clone_params(num_params):
                num_params_copy = {}
                for num_param in num_params:
                    num_params_copy[num_param] = num_params[num_param].detach().clone()
                return num_params_copy
            self.model_params = _clone_params(params['model_params'])
            self.model_params_initial = _clone_params(params['model_params'])
            self.predict_model_back = params['predict_model_back']
        else:
            self.model_params = None
            for m in self.predict_model:
                if type(m) == nn.Linear:
                    nn.init.xavier_uniform_(m.weight.data)

        # Add optimization oracle w^*(.): c -> w
        self.optimization_oracle = params['optimization_oracle']
        self.optimization_params = params.get('optimization_params', None)
        if self.require_grad:
            self.optimization_oracle_back = params['optimization_oracle_back']

        # Add optimization oracle for test
        self.test_optimization_oracle = params.get(
            'test_optimization_oracle',
            self.optimization_oracle,
        )
        self.test_optimization_params = params.get(
            'test_optimization_params',
            self.optimization_params,
        )

        # Add loss function l(., .): c' * c -> R
        self.loss_func = params['loss_func']

        # Add optimizer for model training
        self.optimizer = params.get('optimizer', None)
        self.optimizer_config = params.get('optimizer_config', None)

        self.minibatch_size = params.get('minibatch_size', 4)
        self.if_argmax = params.get('if_argmax', False)

    # Reset the model
    def reset(self):
        def _clone_params(params):
            params_copy = {}
            for param in params:
                params_copy[param] = params[param].detach().clone()
            return params_copy
        self.model_params = _clone_params(self.model_params_initial)

    # Make prediction based on the input features and the prediction model
    def predict(self, features):
        if self.require_grad:
            cost_pred, predict_cache = self.predict_model(features, self.model_params)
        else:
            cost_pred = self.predict_model(features)
            predict_cache = None
        return cost_pred, predict_cache

    # Compute the optimal decision based on the predicted/realized cost
    def optimize(self, cost, if_test=False):
        if if_test:
            oracle = self.test_optimization_oracle
            params = self.test_optimization_params
        else:
            oracle = self.optimization_oracle
            params = self.optimization_params

        if params is not None:
            actions, action_cache = oracle(cost, params, self.require_grad)
        else:
            actions, action_cache = oracle(cost, self.require_grad)
        return actions, action_cache

    # Forward Pass: predict then optimize
    def forward(self, features, if_test=False):
        cost_pred, predict_cache = self.predict(features)
        action_pred, oracle_cache = self.optimize(cost_pred, if_test=if_test)
        return cost_pred, action_pred, {'predict_cache': predict_cache, 'oracle_cache': oracle_cache}

    def cost(self, cost, action):
        batch_size = cost.shape[0]
        if action.shape[1] == 1:
            return cost.gather(1, action).mean()
        else:
            return torch.matmul(
                cost.view(batch_size, 1, self.dim_cost),
                action.view(batch_size, self.dim_cost, 1)
            ).mean()

    # Compute the SPO and surrogate loss, and the gradient of surrogate loss
    def loss(self, cost_pred, action_pred, cost_real, forward_cache):
        action_hindsight, _ = self.optimize(cost_real)
        action_spop, _ = self.optimize(2 * cost_pred - cost_real)
        loss_params = {
            'cost_pred': cost_pred,
            'action_pred': action_pred,
            'cost_real': cost_real,
            'action_hindsight': action_hindsight,
            'action_spop': action_spop,
        }

        if self.if_argmax:
            # print(self.optimization_params)
            # print(cost_real.shape)
            # print(action_pred.shape)
            # print(action_hindsight.shape)
            # print(action_spop.shape)
            loss_spo = self.cost(cost_real, action_pred) - self.cost(cost_real, action_hindsight)
        else:
            batch_size = cost_pred.shape[0]
            loss_spo = torch.matmul(
                cost_real.view(batch_size, 1, self.dim_cost),
                (action_pred - action_hindsight).view(batch_size, self.dim_cost, 1)
            ).mean()

        loss_surrogate, grad = self.loss_func(loss_params, self.require_grad)

        # if np.isnan(loss_surrogate):
        #     for param in self.model_params:
        #         print(self.model_params[param].detach().data)
        #     raise Exception('nan in loss, please check the above model parameters.')

        return loss_spo, loss_surrogate, grad, forward_cache

    def loss_fast(self, features, cost_real):
        cost_pred, action_pred, forward_cache = self.forward(features)
        action_hindsight, _ = self.optimize(cost_real)
        action_spop, _ = self.optimize(2 * cost_pred - cost_real)
        loss_params = {
            'cost_pred': cost_pred,
            'action_pred': action_pred,
            'cost_real': cost_real,
            'action_hindsight': action_hindsight,
            'action_spop': action_spop,
        }

        batch_size = features.shape[0]
        if self.if_argmax:
            loss_spo = self.cost(cost_real, action_pred) - self.cost(cost_real, action_hindsight)
        else:
            loss_spo = torch.matmul(
                cost_real.view(batch_size, 1, self.dim_cost),
                (action_pred - action_hindsight).view(batch_size, self.dim_cost, 1)
            ).mean()

        loss_surrogate, grad = self.loss_func(loss_params, self.require_grad)
        return loss_spo, loss_surrogate, grad, forward_cache

    # Loss for test set
    def loss_test(self, features, cost_real, action_hindsight, argmin_hindsight=None, cost_pred=None, action_pred=None):
        ##################################
        # Old Code
        ##################################
        # cost_pred, action_pred, forward_cache = self.forward(features, if_test=True)

        ##################################
        # Testing Part
        ##################################
        # flag = False
        # if cost_pred is not None and action_pred is not None:
        #     flag = True

        if cost_pred is None:
            cost_pred, action_pred, forward_cache = self.forward(features, if_test=True)
        elif action_pred is None:
            action_pred = self.optimize(cost_pred, if_test=True)
        ##################################
        # Testing Part Ends
        ##################################

        if action_hindsight is None:
            action_hindsight, _ = self.optimize(cost_real)
        action_spop, _ = self.optimize(2 * cost_pred - cost_real)
        loss_params = {
            'cost_pred': cost_pred,
            'action_pred': action_pred,
            'cost_real': cost_real,
            'action_hindsight': action_hindsight,
            'action_spop': action_spop,
        }

        test_size = features.shape[0]

        if argmin_hindsight is not None:
            best_hindsight = self.cost(cost_real, argmin_hindsight)
            loss_spo = self.cost(cost_real, action_pred) - best_hindsight
            print('argmin_hindsight')
        elif self.if_argmax:
            best_hindsight = self.cost(cost_real, action_hindsight)
            loss_spo = self.cost(cost_real, action_pred) - best_hindsight
        else:
            loss_spo = torch.matmul(
                cost_real.view(test_size, 1, self.dim_cost),
                (action_pred - action_hindsight).view(test_size, self.dim_cost, 1)
            ).mean()

            best_hindsight = torch.matmul(
                cost_real.view(test_size, 1, self.dim_cost),
                action_hindsight.view(test_size, self.dim_cost, 1)
            ).mean()

            # if flag:
            #     loss_spo = torch.matmul(
            #         cost_real.view(test_size, 1, self.dim_cost),
            #         (action_pred - action_hindsight).view(test_size, self.dim_cost, 1)
            #     )
            #     best_hindsight = torch.matmul(
            #         cost_real.view(test_size, 1, self.dim_cost),
            #         action_hindsight.view(test_size, self.dim_cost, 1)
            #     )
            #     print(loss_spo[:10])
            #     print(best_hindsight[:10])
            #     raise Exception('Here!')

        loss_surrogate, grad = self.loss_func(loss_params, False)

        if self.baseline_action is not None:
            if self.if_argmax:
                loss_spo_baseline = torch.matmul(
                    self.baseline_action,
                    cost_real.view(test_size, self.dim_cost, 1),
                ).mean() - best_hindsight
            else:
                loss_spo_baseline = torch.matmul(
                    self.baseline_action,
                    cost_real.view(test_size, self.dim_cost, 1),
                ).mean() - best_hindsight
        else:
            loss_spo_baseline = None

        return loss_spo, loss_surrogate, best_hindsight, loss_spo_baseline

    # Backward pass: Compute the gradient with respect to prediction model parameters
    def backward(self, grad, forward_cache):
        dc = self.optimization_oracle_back(grad, forward_cache['oracle_cache'])
        dparam = self.predict_model_back(dc, forward_cache['predict_cache'])
        return dparam

    # Train the prediction model
    def update(self, features, cost_real, optimizer=None, num_iter=1000, optimizer_config=None, if_quiet=False,
               if_print_loss=True, test_set=None, if_test_ini=False, if_mean_spo_loss=False):
        if optimizer is None:
            optimizer = self.optimizer
            if self.optimizer_config is not None:
                optimizer_config = self.optimizer_config.copy()

        loss_spo_list = []
        loss_surr_list = []

        # Compute the initial test set loss
        if if_test_ini and (test_set is not None):
            loss_spo_test, _, best_hindsight, _ = self.loss_test(
                test_set['features'], test_set['cost_real'], test_set.get('action_hindsight'),
                argmin_hindsight=test_set.get('argmin_hindsight'),
            )
            loss_spo_test_ini = loss_spo_test.data.item()
            best_hindsight_ini = best_hindsight.data.item()
        else:
            loss_spo_test_ini, best_hindsight_ini = None, None

        # Train the prediction model
        for i in range(num_iter):
            # Pick mini-batch
            feature_batch, cost_real_batch = batch_loader(features, cost_real, batch_size=self.minibatch_size)
            cost_pred, action_pred, forward_cache = self.forward(feature_batch)
            # Compute loss and gradient
            loss_spo, loss_surrogate, grad, forward_cache = self.loss(
                cost_pred, action_pred, cost_real_batch, forward_cache)
            # Update the coefficients
            if self.require_grad:
                dparam = self.backward(grad, forward_cache)
                self.model_params, optimizer_config = optimizer(self.model_params, dparam, optimizer_config)
            else:
                optimizer.zero_grad()
                loss_surrogate.backward()
                optimizer.step()

            loss_spo_list.append(loss_spo.data.item())
            loss_surr_list.append(loss_surrogate.data.item())

            if i % (num_iter // 10) == 0:
                if not if_quiet:
                    print('i: ', i)
                    if if_print_loss:
                        print('loss_spo: ', loss_spo.data.item())
                        print('loss_surr: ', loss_surrogate.data.item())

        # Compute the test set loss
        if test_set is not None:
            loss_spo_test, _, best_hindsight, loss_spo_baseline = self.loss_test(
                test_set['features'], test_set['cost_real'], test_set.get('action_hindsight'),
                argmin_hindsight=test_set.get('argmin_hindsight'),
            )
            loss_spo_test = loss_spo_test.data.item()
            best_hindsight = best_hindsight.data.item()
            if loss_spo_baseline is not None:
                loss_spo_baseline = loss_spo_baseline.data.item()
        else:
            loss_spo_test, best_hindsight, loss_spo_baseline = None, None, None

        if if_mean_spo_loss:
            assert test_set.get('cost_mean') is not None, 'cost_mean does not in test set! '
            assert test_set.get('action_cost_mean') is not None, 'action_cost_mean does not in test set! '
            loss_mean, _, _, _ = self.loss_test(
                test_set['features'], test_set['cost_real'], test_set.get('action_hindsight'),
                argmin_hindsight=test_set.get('argmin_hindsight'), cost_pred=test_set['cost_mean'],
                action_pred=test_set['action_cost_mean'],
            )
            loss_mean = loss_mean.data.item()
        else:
            loss_mean = None

        return {'loss_spo': loss_spo_list, 'loss_surr': loss_surr_list, 'loss_spo_test': loss_spo_test,
                'hindsight': best_hindsight, 'loss_spo_test_ini': loss_spo_test_ini,
                'hindsight_ini': best_hindsight_ini, 'loss_spo_baseline': loss_spo_baseline,
                'loss_mean': loss_mean,
                }


# Uniformly random mini-batch
def batch_loader(x_input, y_input, batch_size=4):
    n_samples = x_input.shape[0]
    rows = np.random.choice(range(n_samples), size=batch_size, replace=False)
    return x_input[rows], y_input[rows]
