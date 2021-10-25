import numpy as np
import torch


def linear_model_data(n_features, n_samples, dim_cost, std=1., model_coef=None):
    if model_coef is None:
        true_coef = np.random.randint(2, size=(n_features, dim_cost)) * 2
    else:
        true_coef = model_coef['true_coef']
    x_input = np.random.rand(n_samples, n_features) + 1
    y_input = x_input.dot(true_coef) + np.random.rand(n_samples, dim_cost) * std + 1
    # print(y_input)

    x_input = torch.from_numpy(x_input.astype('float32'))
    y_input = torch.from_numpy(y_input.astype('float32')).reshape(n_samples, dim_cost)

    return x_input, y_input, {'true_coef': true_coef}


def portfolio_data(n_features, n_samples, dim_cost, data_params, model_coef=None):
    deg = data_params['deg']
    tau = data_params['tau']
    n_factors = data_params['n_factors']

    if model_coef is None:
        b_mat = np.random.randint(2, size=(n_features, dim_cost))
        l_mat = np.random.uniform(low=-1., high=1., size=(dim_cost, n_factors)) / 400. * tau
        f = np.random.normal(size=n_factors)
    else:
        b_mat = model_coef['b_mat']
        l_mat = model_coef['l_mat']
        f = model_coef['f']

    x = np.random.normal(size=(n_samples, n_features))
    r_mean = np.power(np.matmul(x, b_mat) / 10 / np.sqrt(n_features) + np.power(0.1, 1. / deg), deg)
    # r = r_mean + l_mat.dot(f) + np.random.normal(size=(n_samples, dim_cost)) / 100. * tau
    r = r_mean + np.random.normal(size=(n_samples, dim_cost)) / 100. * tau

    x = torch.from_numpy(x.astype('float32'))
    r = torch.from_numpy(r.astype('float32'))

    return x, r, {'b_mat': b_mat, 'l_mat': l_mat, 'f': f}


def shortest_path_data(n_features, n_samples, dim_cost, data_params, model_coef=None, neg=True):
    deg = data_params['deg']
    eps_bar = data_params['eps_bar']

    if model_coef is None:
        b_mat = np.random.randint(2, size=(n_features, dim_cost))
    else:
        b_mat = model_coef['b_mat']

    x = np.random.normal(size=(n_samples, n_features))
    c_mean = np.power(1 + np.matmul(x, b_mat) / np.sqrt(n_features), deg) + 1
    eps = 1 + np.random.uniform(low=1. - eps_bar, high=1. + eps_bar, size=(n_samples, dim_cost))
    c = c_mean * eps

    x = torch.from_numpy(x.astype('float32'))
    if neg:
        c = torch.from_numpy(-c.astype('float32'))
        c_mean = torch.from_numpy(-c_mean.astype('float32'))
    else:
        c = torch.from_numpy(c.astype('float32'))
        c_mean = torch.from_numpy(c_mean.astype('float32'))

    return x, c, {'b_mat': b_mat, 'c_mean': c_mean}


def multi_class_data(n_features, n_samples, dim_cost, data_params, model_coef=None, neg=False):
    print('----Generating Multi-Class Data----')

    deg = data_params['deg']
    eps_bar = data_params['eps_bar']
    d_type = data_params.get('dtype', 'float32')
    n_labs = dim_cost

    if model_coef is None:
        b_vec = np.random.randint(2, size=n_features)
    else:
        b_vec = model_coef['b_vec']

    x = np.random.normal(size=(n_samples, n_features))
    eps = np.random.uniform(low=1 - eps_bar, high=1 + eps_bar, size=n_samples)
    y_mean = x.dot(b_vec)

    c_mean = np.zeros((n_samples, dim_cost))
    for i in range(n_samples):
        if y_mean[i] >= 0:
            sig = np.exp(-y_mean[i]) / (1 + np.exp(-y_mean[i]))
        else:
            sig = 1 / (1 + np.exp(y_mean[i]))
        lab = int(n_labs * sig)
        if lab == n_labs:
            lab -= 1
        for j in range(dim_cost):
            c_mean[i, j] = abs(lab - j) + 1

    y = np.power(np.abs(y_mean), deg) * np.sign(y_mean) * eps

    c = np.zeros((n_samples, dim_cost))
    for i in range(n_samples):
        if y[i] >= 0:
            sig = np.exp(-y[i]) / (1 + np.exp(-y[i]))
        else:
            sig = 1 / (1 + np.exp(y[i]))
        lab = int(n_labs * sig)
        if lab == n_labs:
            lab -= 1
        for j in range(dim_cost):
            c[i, j] = abs(lab - j) + 1

    x = torch.from_numpy(x.astype(d_type))
    c = torch.from_numpy(c.astype(d_type))
    c_mean = torch.from_numpy(c_mean.astype(d_type))

    return x, c, {'b_vec': b_vec, 'c_mean': c_mean}
