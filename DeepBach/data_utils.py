#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gaetan Hadjeres
"""

import torch
from DeepBach.helpers import cuda_variable


def mask_entry(tensor, entry_index, dim):
    """
    Masks entry entry_index on dim dim
    similar to
    torch.cat((	tensor[ :entry_index],	tensor[ entry_index + 1 :], 0)
    but on another dimension
    :param tensor:
    :param entry_index:
    :param dim:
    :return:
    """
    idx = [i for i in range(tensor.size(dim)) if not i == entry_index]
    idx = cuda_variable(torch.LongTensor(idx))
    tensor = tensor.index_select(dim, idx)
    return tensor


def reverse_tensor(tensor, dim):
    """
    Do tensor[:, ... ,  -1::-1, :] along dim dim
    :param tensor:
    :param dim:
    :return:
    """
    idx = [i for i in range(tensor.size(dim) - 1, -1, -1)]
    idx = cuda_variable(torch.LongTensor(idx))
    tensor = tensor.index_select(dim, idx)
    return tensor
