import numpy as np
import torch

from sympy import symbols, diff, Matrix

ires = 11

teste = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

state_tensor = torch.tensor(teste, dtype=torch.float64)
state_adjusted = state_tensor[0 :ires]

print("state_adjusted", state_adjusted)