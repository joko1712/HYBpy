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

    rann_results = ann.forward(y)

    rann_results = rann_results.detach().numpy()


    state = extract_species_values(projhyb, state)
    values = {}
    for range_y in range(0, len(rann_results)):
        values["rann"+str(range_y+1)] = rann_results[range_y].item()

    for i in range(1, projhyb["ncompartment"]+1):
        values[str(projhyb["compartment"][str(i)]["id"])] = int(projhyb["compartment"][str(i)]["val"])


    for i in range(1, projhyb["mlm"]["ny"]+1):
        values[projhyb["mlm"]["y"][str(i)]["id"]] = values[projhyb["mlm"]["y"][str(i)]["val"]]

    for i in range(1, projhyb["nparameters"]+1):
        values[projhyb["parameters"][str(i)]["id"]] = projhyb["parameters"][str(i)]["val"]

    for i in range(1, projhyb["nspecies"]+1):
        state_symbols.append(sp.Symbol(projhyb["species"][str(i)]["id"]))

    for i in range(1, projhyb["ncompartment"]+1):
        state_symbols.append(sp.Symbol(projhyb["compartment"][str(i)]["id"]))
    
    ### CHANGE THIS!!
    '''
    feed = 0.1250 * t
    values["D"] = feed / values["V"]
    values["Sin"] = 500
    '''

    if jac is not None:
        jac = torch.tensor(jac, dtype=torch.float64)
    fstate = fstate_func(projhyb, values)

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
        

        if jac != None:
            k1_state, k1_jac = odesfun(ann, t, state, jac, None, w, ucontrol1, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
        else:
            k1_state = odesfun(ann, t, state, None, None, w, ucontrol1, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
        
        control = None
        ucontrol2 = controlfun(t + h / 2, batch) if controlfun is not None else []

        h2 = h / 2
        h2 = torch.tensor(h2, dtype=torch.float64)
        print("k1_state", k1_state)
        k1_state = np.array(k1_state)
        k1_state = k1_state.astype(np.float64)
        k1_state = torch.from_numpy(k1_state)
        
        if jac != None:

            state1 = update_state(state, h2, k1_state)
            k2_state, k2_jac = odesfun(ann, t + h2, state1, jac + h2 * k1_jac, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
           
            k2_state = np.array(k2_state)
            k2_state = k2_state.astype(np.float64)
            k2_state = torch.from_numpy(k2_state)

            state2 = update_state(state, h2, k2_state)

            k3_state, k3_jac = odesfun(ann, t + h2, state2, jac + h2 * k2_jac, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
            
            k3_state = np.array(k3_state)
            k3_state = k3_state.astype(np.float64)
            k3_state = torch.from_numpy(k3_state)

        else:
            state1 = update_state(state, h2, k1_state)
            k2_state = odesfun(ann, t + h2, state1, None, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
            state2 = update_state(state, h2, k2_state)
            k3_state = odesfun(ann, t + h2, state2, None, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)

        hl= h - h / 1e10
        ucontrol4 = controlfun(t + hl, batch) if controlfun is not None else []

        if jac != None:

            state3 = update_state(state, hl, k3_state)
            k4_state, k4_jac = odesfun(ann, t + hl, state3, jac + hl * k3_jac, None, w, ucontrol4, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
            k4_state = np.array(k4_state)
            k4_state = k4_state.astype(np.float64)
            k4_state = torch.from_numpy(k4_state)
        else:
            state3 = update_state(state, hl, k3_state)
            k4_state = odesfun(ann, t + hl, state3, None, None, w, ucontrol4, projhyb, fstate, anninp, anninp_tensor, state_symbols, values)
        

        if jac != None:
            stateFinal = calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state)
            state = extract_species_values(projhyb, stateFinal)

        else :
            stateFinal = calculate_state_final_nojac(state, h, k1_state, k2_state, k3_state, k4_state)
            state = extract_species_values(projhyb, stateFinal)

        if jac != None:
            jac = jac + h * (k1_jac / 6 + k2_jac / 3 + k3_jac / 3 + k4_jac / 6)

        t = t + h
    
    stateFinal.append(int(projhyb["compartment"]["1"]["val"]))

    return t, stateFinal, jac, hess

def anninp_rann_func(projhyb, state):

    species_values = extract_species_values(projhyb, state)

    totalsyms = ["t", "dummyarg1", "dummyarg2", "w"]

    for i in range(1, projhyb["nspecies"]+1):
        totalsyms.append(projhyb["species"][str(i)]["id"])  # Species

    for i in range(1, projhyb["ncompartment"]+1):
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

        val_str = projhyb["mlm"]["x"][str(i)]["val"]
        max_str = projhyb["mlm"]["x"][str(i)]["max"]

        # Debugging statements


        try:
            val_expr = sp.sympify(val_str)
            max_expr = sp.sympify(max_str)
        except Exception as e:
            raise ValueError(f"Error sympifying val or max: {e}")


        if not isinstance(val_expr, sp.Symbol):
            val_expr = sp.Symbol(val_str)


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
    species_values = {}
    for key, species in projhyb['species'].items():
        species_id = species['id']
        species_val = state[int(key)-1]
        species_values[species_id] = species_val

    return species_values


def update_state(state, h2, k1_state):
    new_state = {}
    
    for index, (species_id, value) in enumerate(state.items()):

        if index < len(k1_state):
            new_value = value + h2 * k1_state[index]
            new_state[species_id] = new_value
        else:
            new_state[species_id] = value
    
    return new_state


def calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state):
    stateFinal = []
    
    for i, value in enumerate(state.values()):
        new_value = value + h * (k1_state[i] / 6 + k2_state[i] / 3 + k3_state[i] / 3 + k4_state[i] / 6)
        stateFinal.append(new_value.item())

    
    return stateFinal



def calculate_state_final_nojac(state, h, k1_state, k2_state, k3_state, k4_state):
    stateFinal = []
    
    for i, value in enumerate(state.values()):
        new_value = value + h * (k1_state[i] / 6 + k2_state[i] / 3 + k3_state[i] / 3 + k4_state[i] / 6)
        stateFinal.append(new_value)

    
    return stateFinal