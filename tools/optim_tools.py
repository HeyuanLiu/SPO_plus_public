import torch
import numpy as np


def sgd_momentum_old(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)

    v = config.get('velocity', torch.zeros_like(w))

    v = config['momentum'] * v + dw
    next_w = w - config['learning_rate'] * v
    config['velocity'] = v

    return next_w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config['lr_current'] = config['learning_rate']
    config.setdefault('momentum', 0.9)
    config.setdefault('lr_decay', 0.995)

    next_w = {}
    for param in w:
        v = config.get('velocity' + param, torch.zeros_like(w[param]))
        v = config['momentum'] * v + dw[param]
        next_w[param] = w[param] - config['lr_current'] * v
        config['velocity' + param] = v
        config['lr_current'] *= config['lr_decay']

    return next_w, config


def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config['lr_current'] = config['learning_rate']
    config.setdefault('lr_decay', 0.995)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-6)
    config.setdefault('t', 0)

    for param in x:
        config.setdefault('m' + param, np.zeros_like(x))
        config.setdefault('v' + param, np.zeros_like(x))

    next_x = {}
    #############################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in   #
    # the next_x variable. Don't forget to update the m, v, and t variables     #
    # stored in config and to use the epsilon scalar to avoid dividing by zero. #
    #############################################################################
    t = config.get('t') + 1
    for param in x:
        m, v = config.get('m' + param), config.get('v' + param)
        beta1, beta2 = config.get('beta1'), config.get('beta2')
        m = (1 - beta1) * dx[param] + beta1 * m
        v = (1 - beta2) * np.power(dx[param], 2) + beta2 * v
        mhat = m / (1 - np.power(beta1, t))
        vhat = v / (1 - np.power(beta2, t))
        # print(param, (torch.divide(mhat, np.sqrt(vhat) + config.get('epsilon'))).norm(p=2))
        next_x[param] = x[param] - config.get('learning_rate') * torch.divide(
            mhat, np.sqrt(vhat) + config.get('epsilon'))
        config['m' + param], config['v' + param] = m, v
    config['t'] = t
    config['lr_current'] *= config['lr_decay']
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return next_x, config
