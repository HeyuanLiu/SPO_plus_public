from torch import nn
import torch


def mse_loss_func(loss_params, require_grad):
    cost_pred = loss_params['cost_pred']
    cost_real = loss_params['cost_real']
    loss = nn.functional.mse_loss(cost_pred, cost_real)
    batch_size, dim_cost = cost_pred.shape[0], cost_pred.shape[1]
    if require_grad:
        grad = {
            'dc': 2 * (cost_pred - cost_real) / batch_size / dim_cost,
            'dw': torch.zeros_like(cost_pred)
        }
    else:
        grad = None
    return loss, grad


def abs_loss_func(loss_params, require_grad):
    cost_pred = loss_params['cost_pred']
    cost_real = loss_params['cost_real']
    loss = nn.functional.l1_loss(cost_pred, cost_real)
    batch_size, dim_cost = cost_pred.shape[0], cost_pred.shape[1]
    if require_grad:
        grad = {
            'dc': torch.sign(cost_pred - cost_real) / batch_size / dim_cost,
            'dw': torch.zeros_like(cost_pred)
        }
    else:
        grad = None
    return loss, grad


def action_diff_loss_func(loss_params, require_grad):
    action_pred = loss_params['action_pred']
    action_hind = loss_params['action_hindsight']
    loss = nn.functional.mse_loss(action_pred, action_hind)
    batch_size, dim_action = action_pred.shape[0], action_pred.shape[1]
    if require_grad:
        grad = {
            'dc': torch.zeros_like(action_pred),
            'dw': 2 * (action_pred - action_hind) / batch_size / dim_action
        }
    else:
        grad = None
    return loss, grad


def spop_loss_func(loss_params, require_grad):
    cost_pred = loss_params['cost_pred']
    cost_real = loss_params['cost_real']
    action_pred = loss_params['action_pred']
    action_hind = loss_params['action_hindsight']
    action_spop = loss_params['action_spop']

    batch_size, dim_cost = cost_pred.shape[0], cost_pred.shape[1]

    cost_spop = cost_real - 2 * cost_pred
    loss = cost_spop.view(batch_size, 1, dim_cost).matmul(action_spop.view(batch_size, dim_cost, 1)).mean()
    loss += 2 * cost_pred.view(batch_size, 1, dim_cost).matmul(action_hind.view(batch_size, dim_cost, 1)).mean()
    loss -= cost_real.view(batch_size, 1, dim_cost).matmul(action_hind.view(batch_size, dim_cost, 1)).mean()

    if require_grad:
        grad = {
            'dc': (- 2 * action_spop + 2 * action_hind) / batch_size,
            'dw': torch.zeros_like(action_pred),
        }
    else:
        grad = None

    return loss, grad


def spo_loss_func(loss_params, require_grad):
    cost_real = loss_params['cost_real']
    action_pred = loss_params['action_pred']
    action_hind = loss_params['action_hindsight']

    batch_size, dim_cost = cost_real.shape[0], cost_real.shape[1]

    loss = cost_real.view(batch_size, 1, dim_cost).matmul(
        (action_pred - action_hind).view(batch_size, dim_cost, 1)).mean()

    if require_grad:
        grad = {
            'dc': torch.zeros_like(cost_real),
            'dw': cost_real / batch_size,
        }
    else:
        grad = None

    return loss, grad


def spop_argmax_loss_func(loss_params, require_grad=False):
    if require_grad:
        raise Exception('Not done yet.')
    cost_real = loss_params['cost_real']
    cost_pred = loss_params['cost_pred']
    action_hind = loss_params['action_hindsight']
    # print(cost_pred.shape)
    # print(action_hind.shape)
    loss = (cost_real - 2 * cost_pred).amax(dim=1).sum()
    loss += 2 * cost_pred.gather(1, action_hind).sum()
    loss -= cost_real.amin(dim=1).sum()
    return loss, None
