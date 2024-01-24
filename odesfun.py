from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
import numpy as np

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols):


    current_state_dict = ann.state_dict()
    print("Current State Dict:", current_state_dict)

    new_state_dict = {}
    for param_tensor in ann.state_dict():
        if "w" in param_tensor:
            new_state_dict[param_tensor] = torch.randn(*current_state_dict[param_tensor].shape)
        elif "b" in param_tensor:
            new_state_dict[param_tensor] = torch.zeros_like(current_state_dict[param_tensor])


    ann.load_state_dict(new_state_dict)

    if jac is None and hess is None:
        fstate = fstate_func(projhyb)
        anninp, rann, _ = anninp_rann_func(projhyb)

        return fstate, None, None

    else:

        if projhyb['mode'] == 1:


            DanninpDstate = derivativeXY(anninp, state_symbols)
            DanninpDstate = np.array(DanninpDstate)
            DanninpDstate = DanninpDstate.reshape(len(anninp)+1, len(anninp))
            DanninpDstate = DanninpDstate.astype(np.float32)
            DanninpDstate = torch.from_numpy(DanninpDstate)

                   
            y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

            DrannDs = torch.mm(DrannDanninp, DanninpDstate.t())



            DfDrannDrannDw = torch.mm(DfDrann, DrannDw)

            

            # fjac [12x119]
            # DfDrann [12x7]
            # DrannDw [7x119]
            # DrannDs [7x12]
            # DfDs [12x12]
            # jac [12x119]

            DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann,DrannDs)

            
            fjac = torch.mm(DfDsDfDrannDrannDs,jac) + DfDrannDrannDw


            return fstate, fjac

        elif projhyb['mode'] == 3:
            anninp, rann, _ = anninp_rann_func(projhyb)

            fstate = fstate_func(projhyb)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None

     
def derivativeXY(X,Y):
    z = []
    for i in range(0, len(Y)):
        for j in range(0, len(X)):
            cal = diff(X[j], Y[i])

            z = z + [cal]
    return z
