import torch
import torch.nn as nn

from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, ann):
        super().__init__()
        self.ann = ann

    def forward(self, t, x):
        if x.ndim == 1:
            x = x.unsqueeze(1)  
        elif x.shape[0] < x.shape[1]: 
            x = x.transpose(0, 1)
        return self.ann(x).transpose(0, 1) 

def hybodesolver_neuralode(ann, x0, t):
    odefunc = ODEFunc(ann)
    trajectory = odeint(odefunc, x0, t, method='dopri5')
    return trajectory
