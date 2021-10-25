import torch
from torch import nn
from tools import layers_tools


def linear_prediction_model(features, model_params):
    out = torch.matmul(features, model_params['W']) + model_params['b']
    return out, {'features': features}


def linear_prediction_model_back(dc, predict_cache):
    return {
        'W': predict_cache['features'].transpose(0, 1).matmul(dc),
        'b': dc.sum(dim=0),
    }


def two_layers_model(features, model_params):
    hidden_layer, cache1 = layers_tools.affine_relu_forward(features, model_params['W1'], model_params['b1'])
    cost_pred, cache2 = layers_tools.affine_forward(hidden_layer, model_params['W2'], model_params['b2'])
    return cost_pred, {'cache_layer1': cache1, 'cache_layer2': cache2}


def two_layers_model_back(dc, predict_cache):
    cache_layer1, cache_layer2 = predict_cache['cache_layer1'], predict_cache['cache_layer2']
    grad = {}
    dlayer, dw, db = layers_tools.affine_backward(dc, cache_layer2)
    grad['W2'] = dw
    grad['b2'] = db
    _, dw, db = layers_tools.affine_relu_backword(dlayer, cache_layer1)
    grad['W1'] = dw
    grad['b1'] = db
    return grad


def torch_linear():
    torch_linear = nn.Linear
    return
