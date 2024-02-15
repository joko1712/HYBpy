from __future__ import division

from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
import numpy as np
from derivativeXY import numerical_derivativeXY


def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values):

    current_state_dict = ann.state_dict()

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


            DanninpDstate = numerical_derivativeXY(anninp, state_symbols, values)
            DanninpDstate = np.array(DanninpDstate)
            DanninpDstate = DanninpDstate.reshape(len(anninp)+1, len(anninp))
            DanninpDstate = DanninpDstate.astype(np.float64)
            DanninpDstate = torch.from_numpy(DanninpDstate)
   
            y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

            DrannDs = torch.mm(DrannDanninp, DanninpDstate.t())

            DfDrannDrannDw = torch.mm(DfDrann, DrannDw)

            DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann,DrannDs)

            fjac = torch.mm(DfDsDfDrannDrannDs,jac) + DfDrannDrannDw

            fstate = [expr.evalf(subs=values) for expr in fstate]

            return fstate, fjac

        elif projhyb['mode'] == 3:
            anninp, rann, _ = anninp_rann_func(projhyb)

            fstate = fstate_func(projhyb)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None

     
