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

def convert_and_transfer_matrix(sym_matrix, device):
    if isinstance(sym_matrix, sp.Matrix):
        np_array = np.array(sym_matrix).astype(np.float64)
    else:
        np_array = sym_matrix.astype(np.float64)
    tensor = torch.from_numpy(np_array)
    tensor = tensor.to(device)
    return tensor

def computeDFDS(projhyb, fstate, state_symbols, NValues, device):
    if projhyb['mlm']['DFDS'] is None:
        DfDs = numerical_diferentiation_torch(fstate, state_symbols, NValues)
        projhyb['mlm']['DFDS'] = DfDs
    else:
        DfDs = projhyb['mlm']['DFDS']
    
    if isinstance(DfDs, sp.Matrix):
        DfDs = DfDs.subs(NValues)
        DfDs = convert_and_transfer_matrix(DfDs, device)
    else:
        DfDs = DfDs.to(device)
    
    DfDs = DfDs.reshape(len(fstate), len(state_symbols))
    
    if DfDs.is_complex():
        DfDs = DfDs.real()
    
    return DfDs

def computeDFDRANN(projhyb, fstate, rann_symbol, NValues, device):
    if projhyb['mlm']['DFDRANN'] == None:
        DfDrann = numerical_diferentiation_torch(fstate, rann_symbol, NValues)
        projhyb['mlm']['DFDRANN'] = DfDrann
    else:
        DfDrann = projhyb['mlm']['DFDRANN']

    if isinstance(DfDrann, sp.Matrix):
        DfDrann = DfDrann.subs(NValues)
        DfDrann = convert_and_transfer_matrix(DfDrann, device)
    else:
        DfDrann = DfDrann.to(device)

    DfDrann = DfDrann.reshape(len(fstate), projhyb["mlm"]["ny"])
    
    if DfDrann.is_complex():
        DfDrann = DfDrann.real()

    return DfDrann

def computeDANNINPDSTATE(projhyb, anninp, state_symbols, NValues, device):
    if projhyb['mlm']['DANNINPDSTATE'] == None:
        DanninpDstate = numerical_diferentiation_torch(anninp, state_symbols, NValues)
        projhyb['mlm']['DANNINPDSTATE'] = DanninpDstate
    else:
        DanninpDstate = projhyb['mlm']['DANNINPDSTATE']

    if isinstance(DanninpDstate, sp.Matrix):
        DanninpDstate = DanninpDstate.subs(NValues)
        DanninpDstate = convert_and_transfer_matrix(DanninpDstate, device)
    else:
        DanninpDstate = DanninpDstate.to(device)
    
    DanninpDstate = DanninpDstate.reshape(len(anninp)+1, len(anninp))
    
    if DanninpDstate.is_complex():
        DanninpDstate = DanninpDstate.real()

    return DanninpDstate

def computeBackpropagation(ann, anninp_tensor, device):
    y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

    y = y.to(device)
    DrannDanninp = DrannDanninp.to(device)
    DrannDw = DrannDw.to(device)

    return y, DrannDanninp, DrannDw

def computeDRANNDS(DrannDanninp, DanninpDstate, device):
    DrannDs = torch.mm(DrannDanninp, DanninpDstate.t())

    DrannDs = DrannDs.to(device)

    return DrannDs

def computeDfDrannDrannDw(DfDrann, DrannDw, device):
    DfDrannDrannDw = torch.mm(DfDrann, DrannDw)

    DfDrannDrannDw = DfDrannDrannDw.to(device)

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


            device = torch.device("cpu")

            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_function = {
                    executor.submit(computeDFDS, projhyb, fstate, state_symbols, NValues, device): 'DfDs',
                    executor.submit(computeDFDRANN, projhyb, fstate, rann_symbol, NValues, device): 'DfDrann',
                    executor.submit(computeDANNINPDSTATE, projhyb, anninp, state_symbols, NValues, device): 'DanninpDstate',
                    executor.submit(computeBackpropagation, ann, anninp_tensor, device): 'backpropagation'
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
                    executor.submit(computeDRANNDS, DrannDanninp, DanninpDstate, device): 'DrannDs',
                    executor.submit(computeDfDrannDrannDw, DfDrann, DrannDw, device): 'DfDrannDrannDw'
                }

                results = {}
                for future in as_completed(future_to_function):
                    key = future_to_function[future]
                    results[key] = future.result()


            DrannDs = results['DrannDs']
            DfDrannDrannDw = results['DfDrannDrannDw']
            

            DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann,DrannDs)

            DfDsDfDrannDrannDs = DfDsDfDrannDrannDs.to(device)
            
            jac = jac.to(device)

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
