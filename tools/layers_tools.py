import torch


def affine_forward(x, w, b):
    out = torch.matmul(x, w) + b
    return out, {'x': x, 'w': w, 'b': b}


def affine_backward(dout, cache):
    x, w, b = cache['x'], cache['w'], cache['b']
    dx = dout.matmul(w.transpose(0, 1))
    dw = x.transpose(0, 1).matmul(dout)
    db = dout.sum(dim=0)

    return dx, dw, db


def affine_relu_forward(x, w, b):
    out = torch.matmul(x, w) + b
    relu_out = torch.nn.functional.relu(out)
    return relu_out, {'x': x, 'w': w, 'b': b, 'out': out}


def affine_relu_backword(dout, cache):
    x, w, b, out = cache['x'], cache['w'], cache['b'], cache['out']
    dout *= (out > 0)
    dx = dout.matmul(w.transpose(0, 1))
    dw = x.transpose(0, 1).matmul(dout)
    db = dout.sum(dim=0)
    return dx, dw, db
