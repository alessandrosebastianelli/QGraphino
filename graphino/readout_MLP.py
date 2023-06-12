"""
Author: Salva Rühling Cachay
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from utilities.utils import get_activation_function

from .TorchCircuit import TorchCircuit
from .config import *


class ONI_MLP(nn.Module):
    """
    Fully connected MLP on top of node embeddings
    L - number of hidden layers, each of those having half the number of neurons than the previous one.
    """

    def __init__(self, input_dim, output_dim, dropout=0, L=2, batch_norm=True, act_func='elu', device='cuda'):
        super().__init__()

        FC_layers = []
        for l in range(L):
            out_dim_l = input_dim // 2 ** (l + 1)
            FC_layers.append(
                nn.Linear(input_dim // 2 ** l, out_dim_l, bias=True)
            )
            if batch_norm:
                FC_layers.append(nn.BatchNorm1d(out_dim_l))
            FC_layers.append(get_activation_function(act_func, device=device))
            if dropout > 0:
                FC_layers.append(nn.Dropout(dropout))

        self.FC_layers = nn.ModuleList(FC_layers)
        self.out_dim_last_L = input_dim // 2 ** L

        self.fc1 = nn.Linear(self.out_dim_last_L, NUM_LAYERS*NUM_QUBITS)
        self.qc  = TorchCircuit.apply
        self.out_layer = nn.Linear(NUM_QC_OUTPUTS, output_dim, bias=True)
        self.L = L


    def forward(self, x):
        for module in self.FC_layers:
            x = module(x)

        x = F.elu(self.fc1(x))
        x = np.pi * torch.tanh(x)
        x = self.qc(x[0])
        x = F.elu(x)

        y = self.out_layer(x.float())

        return y
