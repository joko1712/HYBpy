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

projhyb_cache = {}

def computeDFDS(projhyb, fstate, state_symbols, NValues):
    if 'DFDS_FUNC' not in projhyb['mlm']:
        x_matrix = sp.Matrix(fstate)
        jacobian_matrix = x_matrix.jacobian(state_symbols)

        all_symbols = sorted(jacobian_matrix.free_symbols, key=lambda s: s.name)
        projhyb['mlm']['DFDS_SYMBOLS'] = all_symbols

        jacobian_func = sp.lambdify(all_symbols, jacobian_matrix, modules='numpy')
        projhyb['mlm']['DFDS_FUNC'] = jacobian_func
    else:
        jacobian_func = projhyb['mlm']['DFDS_FUNC']
        all_symbols = projhyb['mlm']['DFDS_SYMBOLS']

    all_values = [NValues[str(sym)] for sym in all_symbols]
    jacobian_evaluated = jacobian_func(*all_values)

    jacobian_evaluated = np.array(jacobian_evaluated, dtype=np.float32)

    if np.iscomplexobj(jacobian_evaluated):
        jacobian_evaluated = jacobian_evaluated.real

    jacobian_evaluated = jacobian_evaluated.reshape(len(fstate), len(state_symbols))

    return torch.from_numpy(jacobian_evaluated)


def computeDFDRANN(projhyb, fstate, rann_symbol, NValues):
    if 'DFDRANN_FUNC' not in projhyb['mlm']:
        x_matrix = sp.Matrix(fstate)
        jacobian_matrix = x_matrix.jacobian(rann_symbol)

        all_symbols = sorted(jacobian_matrix.free_symbols, key=lambda s: s.name)
        projhyb['mlm']['DFDRANN_SYMBOLS'] = all_symbols

        jacobian_func = sp.lambdify(all_symbols, jacobian_matrix, modules='numpy')
        projhyb['mlm']['DFDRANN_FUNC'] = jacobian_func
    else:
        jacobian_func = projhyb['mlm']['DFDRANN_FUNC']
        all_symbols = projhyb['mlm']['DFDRANN_SYMBOLS']

    all_values = [NValues[str(sym)] for sym in all_symbols]
    jacobian_evaluated = jacobian_func(*all_values)

    jacobian_evaluated = np.array(jacobian_evaluated, dtype=np.float32)

    if np.iscomplexobj(jacobian_evaluated):
        jacobian_evaluated = jacobian_evaluated.real

    jacobian_evaluated = jacobian_evaluated.reshape(len(fstate), projhyb["mlm"]["ny"])

    return torch.from_numpy(jacobian_evaluated)


def computeDANNINPDSTATE(projhyb, anninp, state_symbols, NValues):
    if 'DANNINPDSTATE_FUNC' not in projhyb['mlm']:
        x_matrix = sp.Matrix(anninp)
        jacobian_matrix = x_matrix.jacobian(state_symbols)

        all_symbols = sorted(jacobian_matrix.free_symbols, key=lambda s: s.name)
        projhyb['mlm']['DANNINPDSTATE_SYMBOLS'] = all_symbols

        jacobian_func = sp.lambdify(all_symbols, jacobian_matrix, modules='numpy')
        projhyb['mlm']['DANNINPDSTATE_FUNC'] = jacobian_func
    else:
        jacobian_func = projhyb['mlm']['DANNINPDSTATE_FUNC']
        all_symbols = projhyb['mlm']['DANNINPDSTATE_SYMBOLS']

    all_values = [NValues[str(sym)] for sym in all_symbols]
    jacobian_evaluated = jacobian_func(*all_values)

    jacobian_evaluated = np.array(jacobian_evaluated, dtype=np.float32)

    if np.iscomplexobj(jacobian_evaluated):
        jacobian_evaluated = jacobian_evaluated.real

    if len(anninp) > 1:
        jacobian_evaluated = jacobian_evaluated.reshape(len(anninp), len(state_symbols))

    return torch.from_numpy(jacobian_evaluated)


def computeBackpropagation(ann, anninp_tensor, projhyb):
    loss_fn = torch.nn.MSELoss()
    ny = projhyb['mlm']['ny']
    target_tensor = torch.zeros((ny, 1), dtype=torch.float32)

    y, DrannDanninp, DrannDw = ann.backpropagate_lstm(
        anninp_tensor, target_tensor, loss_fn
    )
    return y, DrannDanninp, DrannDw

def computeDRANNDS(DrannDanninp, DanninpDstate):
    # Ensure both tensors are float32
    DrannDanninp = DrannDanninp.to(dtype=torch.float32)
    DanninpDstate = DanninpDstate.to(dtype=torch.float32)
    return torch.mm(DrannDanninp, DanninpDstate)

def computeDfDrannDrannDw(DfDrann, DrannDw):
    # Ensure both tensors are float32
    DfDrann = DfDrann.to(dtype=torch.float32)
    DrannDw = DrannDw.to(dtype=torch.float32)
    return torch.mm(DfDrann, DrannDw)

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann):
    
    global projhyb_cache

    run_id = projhyb.get("run_id", "default")
    if run_id not in projhyb_cache:
        projhyb_cache[run_id] = {}
    cache = projhyb_cache[run_id]

    if jac is None and hess is None:
        NValues = {**values, **state}
        fstate = [expr.subs(NValues) for expr in fstate]
        return fstate

    if projhyb['mode'] == 1:
        NValues = {**values, **state}

        fstate = cache.get('FSTATE', sp.sympify(fstate))
        state_symbols = cache.get('STATE_SYMBOLS', sp.sympify(state_symbols))
        anninp = cache.get('ANNINP', sp.sympify(anninp))

        cache.update({
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

        # Ensure DfDs and torch.mm(DfDrann, DrannDs) are float32
        DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann, DrannDs).to(dtype=torch.float32)

        # Ensure fjac is float32
        fjac = torch.mm(DfDsDfDrannDrannDs, jac.to(dtype=torch.float32)) + DfDrannDrannDw.to(dtype=torch.float32)

        if 'FSTATE_FUNC' not in projhyb['mlm']:
            fstate_expr_list = fstate_func(projhyb, NValues)
            all_symbols = sorted(
                list(set().union(*(expr.free_symbols for expr in fstate_expr_list))),
                key=lambda s: s.name
            )
            fstate_func_lambdified = sp.lambdify(all_symbols, fstate_expr_list, modules='numpy')
            projhyb['mlm']['FSTATE_FUNC'] = fstate_func_lambdified
            projhyb['mlm']['FSTATE_SYMBOLS'] = all_symbols
        else:
            fstate_func_lambdified = projhyb['mlm']['FSTATE_FUNC']
            all_symbols = projhyb['mlm']['FSTATE_SYMBOLS']

        all_values = [NValues[str(sym)] for sym in all_symbols]
        fstate = fstate_func_lambdified(*all_values)

        if projhyb["thread"] and not getattr(projhyb["thread"], "do_run", True):
            return projhyb, {}, {}, None

        #fstate = [expr.subs(NValues) for expr in fstate]
        return fstate, fjac

    elif projhyb['mode'] == 3:
        anninp, rann, _ = anninp_rann_func(projhyb)
        fstate = fstate_func(projhyb)
        DfDs = Matrix([fstate]).jacobian(Matrix([state]))
        DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))
        fjac = DfDs * jac + DfDrann
        return fstate, fjac, None

    return None, None, None
