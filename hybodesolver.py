from __future__ import division

import numpy as np
import torch
from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
from derivativeXY import numerical_derivativeXY


def hybodesolver(ann, odesfun, controlfun, eventfun, t0, tf, state, jac, hess, w, batch, projhyb):
    t = t0
    hopt = []

    state_symbols = []

    anninp, rann, anninp_mat = anninp_rann_func(projhyb, state)

    anninp_tensor = torch.tensor(anninp_mat, dtype=torch.float64)
    anninp_tensor = anninp_tensor.view(-1, 1)       

    activations = [anninp_tensor]

    y = activations[-1]

    print("y", y)

    rann_results = ann.forward(y)

    rann_results = rann_results.detach().numpy()

    values = extract_species_values(projhyb,state)
    values["compartment"] = int(projhyb["compartment"]["1"]["val"])
    for range_y in range(0, len(rann_results)):
        values["rann"+str(range_y+1)] = rann_results[range_y].item()
        print("rann", values["rann"+str(range_y+1)])

    for i in range(1, projhyb["mlm"]["ny"]+1):
        values[projhyb["mlm"]["y"][str(i)]["id"]] = values[projhyb["mlm"]["y"][str(i)]["val"]]

    for i in range(1, projhyb["nparameters"]+1):
        values[projhyb["parameters"][str(i)]["id"]] = projhyb["parameters"][str(i)]["val"]

    for i in range(1, projhyb["nspecies"]+1):
        state_symbols.append(sp.sympify(projhyb["species"][str(i)]["id"]))

    for i in range(1, projhyb["ncompartments"]+1):
        state_symbols.append(sp.sympify(projhyb["compartment"][str(i)]["id"]))

    jac = torch.tensor(jac, dtype=torch.float64)
    fstate = fstate_func(projhyb,values)
    print("state_symbols", state_symbols)
    print("values", values)


    DfDs_sym = [[expr.diff(symbol) for symbol in state_symbols] for expr in fstate]
    DfDs = [[expr.subs(values) for expr in row] for row in DfDs_sym]

    #DfDs = numerical_derivativeXY(fstate, state_symbols, values)
    rann_symbol = []
    for i in range(1, projhyb["mlm"]["ny"]+1):
        rann_symbol.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["id"]))
    
    DfDrann_sys = [[expr.diff(symbol) for symbol in rann_symbol] for expr in fstate]
    DfDrann = [[expr.subs(values) for expr in row] for row in DfDrann_sys]
    #DfDrann = numerical_derivativeXY(fstate, rann_symbol, values)

    DfDrann = np.array(DfDrann)
    DfDrann = DfDrann.reshape(len(fstate), len(rann))
    DfDrann = DfDrann.astype(np.float64)
    DfDrann = torch.from_numpy(DfDrann)

    DfDs = np.array(DfDs)
    DfDs = DfDs.reshape(len(fstate), len(state_symbols))
    DfDs = DfDs.astype(np.float64)
    DfDs = torch.from_numpy(DfDs)

    while t < tf:
        h = min(projhyb['time']['TAU'], tf - t)
        batch['h'] = h

        if eventfun and callable(eventfun):
            if jac is not 0:
                batch, state, dstatedstate = eventfun(t, batch, state)
                jac = dstatedstate * jac
            else:
                batch, state = eventfun(t, batch, state)

        if controlfun and callable(controlfun):
            ucontrol1 = controlfun(t, batch)
            
        else:
            ucontrol1 = []
        
        if jac is not 0:
            k1_state, k1_jac = odesfun(ann, t, state, jac, None, w, ucontrol1, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
        else:
            k1_state = odesfun(ann,t, state, None, None, w, ucontrol1, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)

        # FIX THIS 
        
        control = None
        #ucontrol2 = controlfun(t + h / 2, batch) if controlfun is not None else []
        ucontrol2 = controlfun() if control is not None else []

        
        h2 = h / 2
        h2 = torch.tensor(h2, dtype=torch.float64)
        k1_state = np.array(k1_state)

        
        k1_state = k1_state.astype(np.float64)
        k1_state = torch.from_numpy(k1_state)

        state = np.array(state)
        state = state.astype(np.float64)
        state = torch.from_numpy(state)

        h2k1_jac = torch.mul(h2, k1_jac)

        
        jach2 = jac + h2k1_jac

        if jac is not 0:
            k2_state, k2_jac = odesfun(ann,t + h2, state + h2 * k1_state, jac + h2 * k1_jac, None, w, ucontrol2, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
           
            k2_state = np.array(k2_state)
            k2_state = k2_state.astype(np.float64)
            k2_state = torch.from_numpy(k2_state)

            k3_state, k3_jac = odesfun(ann,t + h2, state + h2 * k2_state, jac + h2 * k2_jac, None, w, ucontrol2, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
            k3_state = np.array(k3_state)
            k3_state = k3_state.astype(np.float64)
            k3_state = torch.from_numpy(k3_state)

        else:
            k2_state = odesfun(ann,t + h2, state + h2 * k1_state, None, None, w, ucontrol2, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
            k3_state = odesfun(ann,t + h2, state + h2 * k2_state, None, None, w, ucontrol2, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)

        hl = h - h / 1e10
        ucontrol4 = controlfun(t + hl, batch) if controlfun is not None else []

        if jac is not None:
            k4_state, k4_jac = odesfun(ann,t + hl, state + hl * k3_state, jac + hl * k3_jac, None, w, ucontrol4, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
            k4_state = np.array(k4_state)
            k4_state = k4_state.astype(np.float64)
            k4_state = torch.from_numpy(k4_state)
        else:
            k4_state = odesfun(ann,t + hl, state + hl * k3_state, None, None, w, ucontrol4, projhyb, DfDs, DfDrann, fstate, anninp, anninp_tensor, state_symbols, values)
        state = state + h * (k1_state / 6 + k2_state / 3 + k3_state / 3 + k4_state / 6)

        if jac is not None:
            jac = jac + h * (k1_jac / 6 + k2_jac / 3 + k3_jac / 3 + k4_jac / 6)

        t = t + h

    return t, state, jac, hess



def anninp_rann_func(projhyb,state):
    species_values = extract_species_values(projhyb, state)

    totalsyms = ["t", "dummyarg1", "dummyarg2", "w"]

    for i in range(1, projhyb["nspecies"]+1):
        totalsyms.append(projhyb["species"][str(i)]["id"])  # Species

    for i in range(1, projhyb["ncompartments"]+1):
        totalsyms.append(projhyb["compartment"][str(i)]["id"])  # Compartments

    for i in range(1, projhyb["nparameters"]+1):
        totalsyms.append(projhyb["parameters"][str(i)]["id"])  # Parameters

    for i in range(1, projhyb["nruleAss"]+1):
        totalsyms.append(projhyb["ruleAss"][str(i)]["id"])  # RuleAss

    for i in range(1, projhyb["nreaction"]+1):
        totalsyms.append(projhyb["reaction"][str(i)]["id"])  # Reaction

    for i in range(1, projhyb["ncontrol"]+1):
        totalsyms.append(projhyb["control"][str(i)]["id"])  # Control

    anninp = []
    anninp_mat = []    
    rann = []

    for i in range(1,  projhyb["mlm"]["nx"]+1):
        totalsyms.append(projhyb["mlm"]["x"][str(i)]["id"])

        val_expr = sp.sympify(projhyb["mlm"]["x"][str(i)]["val"])

        max_expr = sp.sympify(projhyb["mlm"]["x"][str(i)]["max"])

        anninp.append(val_expr/max_expr)


        val_mat = val_expr.evalf(subs=species_values)
        max_mat = max_expr.evalf(subs=species_values)


        anninp_mat.append(val_mat/max_mat)

    for i in range(1, projhyb["mlm"]["ny"]+1):
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["id"])  # Ann outputs
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["val"])

        rann.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["val"]))

    totalsyms = symbols(totalsyms)


    anninp_symbol = sp.sympify(anninp)

    return anninp_symbol, rann, anninp_mat


def extract_species_values(projhyb, state):
    # TODO: ADD CONTROL VALUES
    species_values = {}
    for key, species in projhyb['species'].items():
        species_id = species['id']
        species_val = state[int(key)]
        species_values[species_id] = species_val

    return species_values

