from __future__ import division

from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
import numpy as np
from derivativeXY import numerical_derivativeXY
from derivativeXY import numerical_derivativeXY_optimized
from derivativeXY import numerical_derivativeXY_optimized_torch
from derivativeXY import numerical_diferentiation
from derivativeXY import numerical_diferentiation_torch

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def computeDFDS(projhyb, fstate, state_symbols, NValues):
    if projhyb['mlm']['DFDS'] == None:
        DfDs = numerical_diferentiation_torch(fstate, state_symbols, NValues)
        projhyb['mlm']['DFDS'] = DfDs
    else:
        DfDs = projhyb['mlm']['DFDS']

    DfDs = DfDs.subs(NValues)
    DfDs = np.array(DfDs)
    DfDs = DfDs.reshape(len(fstate), len(state_symbols))
    if np.iscomplexobj(DfDs):
        DfDs = DfDs.real
    DfDs = DfDs.astype(np.float64)
    DfDs = torch.from_numpy(DfDs)

    return DfDs

def computeDFDRANN(projhyb, fstate, rann_symbol, NValues):
    if projhyb['mlm']['DFDRANN'] == None:
        DfDrann = numerical_diferentiation_torch(fstate, rann_symbol, NValues)
        projhyb['mlm']['DFDRANN'] = DfDrann
    else:
        DfDrann = projhyb['mlm']['DFDRANN']

    DfDrann = DfDrann.subs(NValues)

    DfDrann = np.array(DfDrann)
    DfDrann = DfDrann.reshape(len(fstate), projhyb["mlm"]["ny"])
    if np.iscomplexobj(DfDrann):
        print("DfDrann is complex")
        DfDrann = DfDrann.real
    DfDrann = DfDrann.astype(np.float64)
    DfDrann = torch.from_numpy(DfDrann)

    return DfDrann

def computeDANNINPDSTATE(projhyb, anninp, state_symbols, NValues):
    if projhyb['mlm']['DANNINPDSTATE'] == None:
        DanninpDstate = numerical_diferentiation_torch(anninp, state_symbols, NValues)
        projhyb['mlm']['DANNINPDSTATE'] = DanninpDstate
    else:
        DanninpDstate = projhyb['mlm']['DANNINPDSTATE']

    DanninpDstate = DanninpDstate.subs(NValues)

    DanninpDstate = np.array(DanninpDstate)
    DanninpDstate = DanninpDstate.reshape(len(anninp)+1, len(anninp))
    if np.iscomplexobj(DanninpDstate):
        DanninpDstate = DanninpDstate.real
    DanninpDstate = DanninpDstate.astype(np.float64)
    DanninpDstate = torch.from_numpy(DanninpDstate)

    return DanninpDstate

def computeBackpropagation(ann, anninp_tensor):
    y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

    return y, DrannDanninp, DrannDw

def computeDRANNDS(DrannDanninp, DanninpDstate):
    DrannDs = torch.mm(DrannDanninp, DanninpDstate.t())

    return DrannDs

def computeDfDrannDrannDw(DfDrann, DrannDw):
    DfDrannDrannDw = torch.mm(DfDrann, DrannDw)

    return DfDrannDrannDw

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, fstate, anninp, anninp_tensor, state_symbols, values):

    if jac is None and hess is None:
                
        NValues = {}
        NValues.update(values)
        NValues.update(state)

        fstate = [expr.subs(NValues) for expr in fstate]


        return fstate

    else:

        if projhyb['mode'] == 1:
            
            NValues = {}
            NValues.update(values)
            NValues.update(state)

            if projhyb['mlm']["FSTATE"] == None:
                fstate = sp.sympify(fstate)
                projhyb['mlm']["FSTATE"] = fstate
            else:
                fstate = projhyb['mlm']["FSTATE"]

            if projhyb['mlm']['STATE_SYMBOLS'] == None:
                state_symbols = sp.sympify(state_symbols)
                projhyb['mlm']['STATE_SYMBOLS'] = state_symbols
            else:
                state_symbols = projhyb['mlm']['STATE_SYMBOLS']

            if projhyb['mlm']['ANNINP'] == None:
                anninp = sp.sympify(anninp)
                projhyb['mlm']['ANNINP'] = anninp
            else:
                anninp = projhyb['mlm']['ANNINP']
            

            #DfDs_sym = [[expr.diff(symbol) for symbol in state_symbols] for expr in fstate]
            #DfDs = [[expr.subs(NValues) for expr in row] for row in DfDs_sym]

            #DfDs = numerical_derivativeXY(fstate, state_symbols, NValues)
            #DfDs = numerical_derivativeXY_optimized(fstate, state_symbols, NValues)
            #DfDs = numerical_derivativeXY_optimized_torch(fstate, state_symbols, NValues)

            rann_symbol = []
            for i in range(1, projhyb["mlm"]["ny"]+1):
                rann_symbol.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["id"]))
            
            #DfDrann_sys = [[expr.diff(symbol) for symbol in rann_symbol] for expr in fstate]
            #DfDrann = [[expr.subs(NValues) for expr in row] for row in DfDrann_sys]
            #DanninpDstate = numerical_derivativeXY(anninp, state_symbols, NValues)



            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_function = {
                    executor.submit(computeDFDS, projhyb, fstate, state_symbols, NValues): 'DfDs',
                    executor.submit(computeDFDRANN, projhyb, fstate, rann_symbol, NValues): 'DfDrann',
                    executor.submit(computeDANNINPDSTATE, projhyb, anninp, state_symbols, NValues): 'DanninpDstate',
                    executor.submit(computeBackpropagation, ann, anninp_tensor): 'backpropagation'
                }

                results = {}
                for future in as_completed(future_to_function):
                    key = future_to_function[future]
                    results[key] = future.result()

            DfDs = results['DfDs']
            DfDrann = results['DfDrann']
            DanninpDstate = results['DanninpDstate']
            y, DrannDanninp, DrannDw = results['backpropagation']


            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_function = {
                    executor.submit(computeDRANNDS, DrannDanninp, DanninpDstate): 'DrannDs',
                    executor.submit(computeDfDrannDrannDw, DfDrann, DrannDw): 'DfDrannDrannDw'
                }

                results = {}
                for future in as_completed(future_to_function):
                    key = future_to_function[future]
                    results[key] = future.result()


            DrannDs = results['DrannDs']
            DfDrannDrannDw = results['DfDrannDrannDw']
            

            DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann,DrannDs)


            fjac = torch.mm(DfDsDfDrannDrannDs,jac) + DfDrannDrannDw

            fstate = [expr.subs(NValues) for expr in fstate]
            print(fstate)
    
            return fstate, fjac

        elif projhyb['mode'] == 3:
            anninp, rann, _ = anninp_rann_func(projhyb)

            fstate = fstate_func(projhyb)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None