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

    # TODO: Add LSTM layer initialization
    '''        
    elif isinstance(layer, LSTMLayer):
            # LSTM layer weight initialization could be more complex
            # Initialize the parameters specific to your LSTM structure
            # Example for one weight matrix
            layer.wf.data = torch.randn_like(layer.wf) * np.sqrt(2 / (layer.wf.size(0) + layer.wf.size(1)))
            # ... do this for all LSTM parameters

            # Add LSTM layer parameters to the list
            w.extend(layer.wf.flatten().detach().numpy())
            # ... do this for all LSTM parameters
    '''
    w = np.array(w).reshape(-1, 1)
    return w, ann
