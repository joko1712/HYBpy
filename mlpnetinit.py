import numpy as np
import torch
from customMLP import TanhLayer, ReLULayer, LSTMLayer


def mlpnetinitw(ann):

    w = []

    for layer in ann.layers:
        if isinstance(layer, TanhLayer) or isinstance(layer, ReLULayer):
            layer.w.data = torch.randn_like(
                layer.w) * np.sqrt(2 / (layer.w.size(0) + layer.w.size(1)))
            layer.b.data = torch.zeros_like(layer.b)

            w.extend(layer.w.flatten().detach().numpy())
            w.extend(layer.b.flatten().detach().numpy())

        elif isinstance(layer, LSTMLayer):
            for param in layer.lstm_cell.parameters():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
                w.extend(param.detach().cpu().numpy().flatten())

    
    w = np.array(w).reshape(-1, 1)

    return w, ann
