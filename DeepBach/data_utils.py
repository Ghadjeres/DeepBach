"""
@author: Gaetan Hadjeres
"""

import torch


def mask_entry(tensor, entry_index, dim):
    """
    Remove entry entry_index along dimension dim.

    GPU SPEEDUP: index tensor is created directly on the same device as the
    input, avoiding the CPU->GPU round-trip that the original
    cuda_variable(torch.LongTensor(idx)) caused.
    """
    idx = [i for i in range(tensor.size(dim)) if i != entry_index]
    idx = torch.tensor(idx, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)


def reverse_tensor(tensor, dim):
    """
    Reverse tensor along dimension dim.

    GPU SPEEDUP: same device-aware index creation as mask_entry.
    """
    idx = torch.arange(tensor.size(dim) - 1, -1, -1,
                       dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim, idx)