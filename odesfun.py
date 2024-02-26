from __future__ import division

from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
import numpy as np
from derivativeXY import numerical_derivativeXY
from derivativeXY import numerical_derivativeXY_optimized
from derivativeXY import numerical_derivativeXY_optimized_torch


def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, fstate, anninp, anninp_tensor, state_symbols, values):

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

            NValues = {}
            NValues.update(values)
            NValues.update(state)

            #DfDs_sym = [[expr.diff(symbol) for symbol in state_symbols] for expr in fstate]
            #DfDs = [[expr.subs(NValues) for expr in row] for row in DfDs_sym]

            #DfDs = numerical_derivativeXY(fstate, state_symbols, NValues)

            DfDs = numerical_derivativeXY_optimized(fstate, state_symbols, NValues)

            rann_symbol = []
            for i in range(1, projhyb["mlm"]["ny"]+1):
                rann_symbol.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["id"]))
            
            #DfDrann_sys = [[expr.diff(symbol) for symbol in rann_symbol] for expr in fstate]
            #DfDrann = [[expr.subs(NValues) for expr in row] for row in DfDrann_sys]
     
            DfDrann = numerical_derivativeXY_optimized(fstate, rann_symbol, NValues)

            DfDrann = np.array(DfDrann)
            DfDrann = DfDrann.reshape(len(fstate), projhyb["mlm"]["ny"])
            DfDrann = DfDrann.astype(np.float64)
            DfDrann = torch.from_numpy(DfDrann)

            DfDs = np.array(DfDs)
            DfDs = DfDs.reshape(len(fstate), len(state_symbols))
            DfDs = DfDs.astype(np.float64)
            DfDs = torch.from_numpy(DfDs)

            #DanninpDstate = numerical_derivativeXY(anninp, state_symbols, NValues)
            DanninpDstate = numerical_derivativeXY_optimized(anninp, state_symbols, NValues)
            DanninpDstate = np.array(DanninpDstate)
            DanninpDstate = DanninpDstate.reshape(len(anninp)+1, len(anninp))
            DanninpDstate = DanninpDstate.astype(np.float64)
            DanninpDstate = torch.from_numpy(DanninpDstate)
   
            y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

            DrannDs = torch.mm(DrannDanninp, DanninpDstate.t())

            DfDrannDrannDw = torch.mm(DfDrann, DrannDw)

            DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann,DrannDs)

            fjac = torch.mm(DfDsDfDrannDrannDs,jac) + DfDrannDrannDw

            fstate = [expr.evalf(subs=NValues) for expr in fstate]

            print("fstate",fstate)
            return fstate, fjac

        elif projhyb['mode'] == 3:
            anninp, rann, _ = anninp_rann_func(projhyb)

            fstate = fstate_func(projhyb)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None

     
