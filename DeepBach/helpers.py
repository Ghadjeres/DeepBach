"""
@author: Gaetan Hadjeres (Modified for PyTorch 1.10+)
"""

import torch


def cuda_variable(tensor):
    """Move tensor to CUDA if available."""
    if torch.cuda.is_available():
        # non_blocking=True allows the CPU->GPU transfer to overlap with GPU
        # compute when the DataLoader used pin_memory=True, improving throughput.
        return tensor.cuda(non_blocking=True)
    return tensor


def to_numpy(variable):
    """Detach and move tensor to CPU numpy array."""
    if torch.cuda.is_available():
        return variable.detach().cpu().numpy()
    return variable.detach().numpy()


def init_hidden(num_layers, batch_size, lstm_hidden_size):
    """
    Initialise LSTM hidden state (h_0, c_0) to zeros.

    BUG FIX: the original used torch.randn, which initialises with random
    normal values.  Standard LSTM convention is zero initialisation — random
    init can cause unstable gradients at the start of each sequence and makes
    training harder to reproduce.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h = torch.zeros(num_layers, batch_size, lstm_hidden_size, device=device)
    c = torch.zeros(num_layers, batch_size, lstm_hidden_size, device=device)
    return h, c