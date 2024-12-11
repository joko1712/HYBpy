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

# Caching for expensive computations
projhyb_cache = {}

def computeDFDS(projhyb, fstate, state_symbols, NValues):
    if projhyb['mlm']['DFDS'] is None:
        DfDs = numerical_diferentiation_torch(fstate, state_symbols, NValues)
        projhyb['mlm']['DFDS'] = DfDs
    else:
        DfDs = projhyb['mlm']['DFDS']

    
    DfDs = DfDs.subs(NValues)
    
        
    DfDs = np.array(DfDs).reshape(len(fstate), len(state_symbols))


    if np.iscomplexobj(DfDs):
        DfDs = DfDs.real

    DfDs = torch.from_numpy(DfDs.astype(np.float64))
    return DfDs

def computeDFDRANN(projhyb, fstate, rann_symbol, NValues):

    if projhyb['mlm']['DFDRANN'] is None:
        DfDrann = numerical_diferentiation_torch(fstate, rann_symbol, NValues)
        projhyb['mlm']['DFDRANN'] = DfDrann
    else:
        DfDrann = projhyb['mlm']['DFDRANN']

        
    DfDrann = DfDrann.subs(NValues)
    DfDrann = np.array(DfDrann).reshape(len(fstate), projhyb["mlm"]["ny"])


    if np.iscomplexobj(DfDrann):
        DfDrann = DfDrann.real
    DfDrann = torch.from_numpy(DfDrann.astype(np.float64))
    return DfDrann

def computeDANNINPDSTATE(projhyb, anninp, state_symbols, NValues):
    if projhyb['mlm']['DANNINPDSTATE'] is None:

        DanninpDstate = numerical_diferentiation_torch(anninp, state_symbols, NValues)

        projhyb['mlm']['DANNINPDSTATE'] = DanninpDstate
    else:
        DanninpDstate = projhyb['mlm']['DANNINPDSTATE']

    DanninpDstate = DanninpDstate.subs(NValues)

    DanninpDstate = np.array(DanninpDstate)
    if len(anninp) > 1:
        
        DanninpDstate = DanninpDstate.reshape(len(anninp), len(state_symbols))

    if np.iscomplexobj(DanninpDstate):
        DanninpDstate = DanninpDstate.real
    DanninpDstate = torch.from_numpy(DanninpDstate.astype(np.float64))
    return DanninpDstate

def computeBackpropagation(ann, anninp_tensor, projhyb):
    y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor, projhyb['mlm']['ny'])

    return y, DrannDanninp, DrannDw

def computeDRANNDS(DrannDanninp, DanninpDstate):

    return torch.mm(DrannDanninp, DanninpDstate)

def computeDfDrannDrannDw(DfDrann, DrannDw):


    return torch.mm(DfDrann, DrannDw)

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann):

    if jac is None and hess is None:
        NValues = {**values, **state}
        fstate = [expr.subs(NValues) for expr in fstate]
        return fstate



    if projhyb['mode'] == 1:
        NValues = {**values, **state}
        fstate = projhyb_cache.get('FSTATE', sp.sympify(fstate))
        state_symbols = projhyb_cache.get('STATE_SYMBOLS', sp.sympify(state_symbols))
        anninp = projhyb_cache.get('ANNINP', sp.sympify(anninp))

        projhyb_cache.update({
            'FSTATE': fstate,
            'STATE_SYMBOLS': state_symbols,
            'ANNINP': anninp
        })

        rann_symbol = rann

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_function = {
                executor.submit(computeDFDS, projhyb, fstate, state_symbols, NValues): 'DfDs',
                executor.submit(computeDFDRANN, projhyb, fstate, rann_symbol, NValues): 'DfDrann',
                executor.submit(computeDANNINPDSTATE, projhyb, anninp, state_symbols, NValues): 'DanninpDstate',
                executor.submit(computeBackpropagation, ann, anninp_tensor, projhyb): 'backpropagation'
            }

            results = {}
            for future in as_completed(future_to_function):
                key = future_to_function[future]
                results[key] = future.result()


        DfDs = results['DfDs']
        DfDrann = results['DfDrann']
        DanninpDstate = results['DanninpDstate']
        y, DrannDanninp, DrannDw = results['backpropagation']

        with ThreadPoolExecutor(max_workers=4) as executor:
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

        DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann, DrannDs)

        fjac = torch.mm(DfDsDfDrannDrannDs, jac) + DfDrannDrannDw


        fstate = [expr.subs(NValues) for expr in fstate]
        return fstate, fjac

    elif projhyb['mode'] == 3:
        anninp, rann, _ = anninp_rann_func(projhyb)
        fstate = fstate_func(projhyb)
        DfDs = Matrix([fstate]).jacobian(Matrix([state]))
        DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))
        fjac = DfDs * jac + DfDrann
        return fstate, fjac, None

    return None, None, None