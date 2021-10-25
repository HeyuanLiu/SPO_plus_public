import torch
from torch import nn
from torch.nn import functional
import numpy as np


def softmax_oracle(cost_pred, params, require_grad):
    const = params['const']
    if const is not None:
        action_pred = functional.softmax(-const * cost_pred, dim=1)
    else:
        action_pred = torch.argmax(-cost_pred, dim=1, keepdim=True)
    # print(cost_pred.shape)
    # print(action_pred.shape)

    if require_grad:
        return action_pred, {'const': const, 'action_pred': action_pred}
    else:
        return action_pred, None


def softmax_oracle_back(grad, oracle_cache):
    dw, dc = grad['dw'], grad['dc']
    w, const = oracle_cache['action_pred'], oracle_cache['const']
    batch_size, dim_cost = w.shape[0], w.shape[1]
    dc += const * (w * dw - w.view(batch_size, 1, dim_cost).matmul(
        dw.view(batch_size, dim_cost, 1)).view(batch_size, 1) * w)
    return dc


def entropy_oracle(cost_pred, params, require_grad, err_tol=1e-6):
    def ent_prob(w_test):
        d = w_test.shape[0]
        return w_test.view(1, d).matmul(torch.log(w_test).view(d, 1))[0][0]

    def ent_cost(cost, a):
        b = cost * (-a)
        b -= torch.amax(b)
        ent = b.dot(torch.exp(b)) / torch.exp(b).sum() - torch.log(torch.exp(b).sum())

        return ent

        # w_test = torch.exp(cost * -a)
        # w_test = w_test / w_test.sum()

        # return ent_prob(w_test)

    def ent_bisection(cost, r):

        a_last = 1
        if ent_cost(cost, a_last) < r:
            a_new = 2 * a_last
            while ent_cost(cost, a_new) < r and a_new < 1e4:
                a_last = a_new
                a_new = 2 * a_new
        else:
            a_new = a_last / 2
            while ent_cost(cost, a_new) > r:
                a_last = a_new
                a_new = a_new / 2

        a_mid = (a_last + a_new) / 2
        a_up, a_low = max(a_last, a_new), min(a_last, a_new)
        while a_up - a_low > err_tol:
            if ent_cost(cost, a_mid) > r:
                a_up = a_mid
            else:
                a_low = a_mid
            a_mid = (a_up + a_low) / 2
        return a_mid

    n_batch, dim_cost = cost_pred.shape[0], cost_pred.shape[1]
    r = params['r']
    assert r > 0, 'need r > 0'
    assert r < np.log(dim_cost) - np.log(dim_cost - 1), 'need r < log(n) - log(n-1)'
    r = params['r'] - np.log(dim_cost)

    a_batch = torch.Tensor([-ent_bisection(cost_pred[i], r) for i in range(cost_pred.shape[0])]).view(n_batch, 1)

    w = cost_pred * a_batch
    w -= w.amax(dim=1).view(n_batch, 1)
    w = torch.exp(w)
    # print('w: ', w)
    # print(w.sum(1))
    w = w / w.sum(1).view(n_batch, 1)
    # print('w: ', w)

    # print('r: ', r)
    # for i in range(cost_pred.shape[0]):
    #    print('ent: ', ent_prob(w[i]).item())

    oracle_cache = None
    if require_grad:
        oracle_cache = {
            'r': r + np.log(dim_cost),
            'a': a_batch,
            'w': w,
            'cost': cost_pred,
        }

    # print('a: ', a_batch)
    # print('w: ', w)

    return w, oracle_cache


def entropy_oracle_back(grad, oracle_cache):
    dw = grad['dw']
    r, a, w, c = oracle_cache['r'], oracle_cache['a'], oracle_cache['w'], oracle_cache['cost']
    batch_size, dim_cost = w.shape[0], w.shape[1]
    # print('dc 1st: ', grad['dc'])
    dc = grad['dc'] + w * dw * a
    # print('dc 2nd: ', dc)
    # print('w.dw.a: ', w * dw * a)
    dc -= w.view(batch_size, 1, dim_cost).matmul(
        dw.view(batch_size, dim_cost, 1)).view(batch_size, 1) * w * a
    # print('dc 3rd: ', dc)
    # print('w.wT.dw: ', w.view(batch_size, 1, dim_cost).matmul(
    #    dw.view(batch_size, dim_cost, 1)).view(batch_size, 1) * w)
    c_dot_w = w.view(batch_size, 1, dim_cost).matmul(c.view(batch_size, dim_cost, 1)).view(batch_size, 1)
    # print('c.w: ', c_dot_w)
    dw_da = w * (c - c_dot_w)
    # print('dw/da: ', dw_da)
    dent_dc = w * (c - c_dot_w) * a * a
    # print('dent/dc: ', dent_dc)
    dent_da = ((1 + torch.log(w)) * w * (c - c_dot_w)).sum(1).view(batch_size, 1)
    # print('dent/da: ', dent_da)
    # dc -= dent_dc.view(batch_size, 1, dim_cost).matmul(dw.view(batch_size, dim_cost, 1)).view(
    #     batch_size, 1) * dw_da / dent_da
    dc -= dw_da.view(batch_size, 1, dim_cost).matmul(dw.view(batch_size, dim_cost, 1)).view(
        batch_size, 1) * dent_dc / dent_da
    # print('dc 4th: ', dc)

    return dc


def barrier_oracle(cost_pred, params, require_grad, err_tol=1e-6, max_iter=50):

    def barrier_alt(c, u):
        d = c.shape[0]
        return d * torch.log((1 / (c + u)).sum()) + torch.log(c + u).sum()

    def barrier_bisection(cost, r_tar):
        c = cost - cost.min()
        u_low, u_try = 0., 1.

        count = 0
        while barrier_alt(c, u_try) > r_tar and count < max_iter:
            u_try = 2 * u_try - u_low
            count += 1
        u_up = u_try
        # if count == max_iter:
        #     print('--', u_try, c)

        count = 0
        barrier_curr = barrier_alt(c, u_try)
        # while abs(barrier_curr - r) > err_tol * r and count < max_iter:
        while u_up - u_low > err_tol * u_up and count < max_iter:
            # print(u_up - u_low)
            u_mid = (u_up + u_low) / 2
            barrier_curr = barrier_alt(c, u_mid)
            if barrier_curr > r_tar:
                u_low = u_mid
            else:
                u_up = u_mid
            count += 1
        # if count == max_iter:
        #     print(r, barrier_curr, u_low, u_up, c)
        # print(u_up - u_low)

        return (u_up + u_low) / 2 - cost.min()

    r = params['r']
    n_batch, dim_cost = cost_pred.shape[0], cost_pred.shape[1]
    u_batch = np.array([barrier_bisection(cost_pred[i], r) for i in range(cost_pred.shape[0])], dtype='float32')
    u_batch = torch.from_numpy(u_batch).view(n_batch, 1)
    # print('----- u:', u_batch)
    w_scale = 1 / (cost_pred + u_batch)
    w = w_scale / w_scale.sum(1).view(n_batch, 1)

    oracle_cache = None
    if require_grad:
        oracle_cache = {
            'cost': cost_pred,
            'u': u_batch,
            'w': w,
            'w_scale': w_scale,
        }

    return w, oracle_cache


def barrier_oracle_back(grad, oracle_cache, debug=False):
    dl_dw, dl_dc = grad['dw'] * 1., grad['dc'] * 1.
    # assert dw.sum() == 0, 'dw/dc not done yet!'
    # print('dl_dc:', dl_dc)

    cost_pred, u, w_scale, w = oracle_cache['cost'], oracle_cache['u'], oracle_cache['w_scale'], oracle_cache['w']
    n_batch, dim_cost = w_scale.shape[0], w_scale.shape[1]

    # dl_dws = (dl_dw - torch.matmul(dl_dw.view(n_batch, 1, dim_cost), w.view(n_batch, dim_cost, 1)).view(
    #     n_batch, 1)) / w_scale.sum(dim=1).view(n_batch, 1)
    ws_sum = w_scale.sum(dim=1)
    dl_dws = (dl_dw - (dl_dw * w).sum(dim=1).view(n_batch, 1)) / ws_sum.view(n_batch, 1)
    # print('dl_dws =', dl_dws) ###
    dws_du = - torch.pow(w_scale, 2)
    # print('dws_du =', dws_du) ###
    dws_dc = - torch.pow(w_scale, 2)
    # print('dws_dc =', dws_dc) ###
    du_dc_tmp = w_scale - dim_cost * torch.pow(w_scale, 2) / ws_sum.view(n_batch, 1)
    du_dc = - du_dc_tmp / du_dc_tmp.sum(dim=1).view(n_batch, 1)
    # print('tmp:', du_dc_tmp)
    # print('du_dc:', du_dc)
    # print('du_dc =', du_dc) #????
    dl_du = (dl_dws * dws_du).sum(dim=1).view(n_batch, 1)
    # print('dl_du =', dl_du) ###
    # dl_du = torch.matmul(dl_dws.view(n_batch, 1, dim_cost), dws_du.view(n_batch, dim_cost, 1)).view(n_batch, 1)
    # print('dl_du_dc =', du_dc * dl_du)
    # print(du_dc.shape, dl_du.shape)
    dl_dc += du_dc * dl_du
    # print('dl_dws_dc =', dl_dws * dws_dc)
    # print(dl_dws.shape, dws_dc.shape)
    dl_dc += dl_dws * dws_dc

    if debug:
        return dl_dc, (dl_dws, dws_du, dws_dc, du_dc, dl_du, du_dc * dl_du, dl_dws * dws_dc)
    else:
        return dl_dc


def argmin_test(cost_pred, params, require_grad=False):
    print('argmin')

    if params['arg'] == 'min':
        return cost_pred.argmin(dim=1, keepdim=True), None
    elif params['arg'] == 'max':
        return cost_pred.argmax(dim=1, keepdim=True), None
    else:
        raise Exception('only min/max is allowed')


def shortest_path_oracle(cost_pred, params, requied_grad):
    if requied_grad:
        print('Warning! no grad will be provided.')

    paths = params['paths']
    min_max = params['min_max']
    cost_paths = torch.matmul(cost_pred, paths.transpose(1, 0))

    if min_max == 'min':
        arg_paths = cost_paths.argmin(dim=1, keepdim=False)
    elif min_max == 'max':
        arg_paths = cost_paths.argmax(dim=1, keepdim=False)
    else:
        raise Exception('min_max can only be min or max!')

    return paths[arg_paths], None
