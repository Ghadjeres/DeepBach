"""
@author: Gaetan Hadjeres
"""

import torch
from torch.autograd import Variable


def cuda_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor.cuda(), volatile=volatile)
    else:
        return Variable(tensor, volatile=volatile)


def to_numpy(variable: Variable):
    if torch.cuda.is_available():
        return variable.data.cpu().numpy()
    else:
        return variable.data.numpy()


def init_hidden(num_layers, batch_size, lstm_hidden_size,
                volatile=False):
    hidden = (
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size),
            volatile=volatile),
        cuda_variable(
            torch.randn(num_layers, batch_size, lstm_hidden_size),
            volatile=volatile)
    )
    return hidden
