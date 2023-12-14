from fStateFunc import fstate_func 
import torch
from sympy import *
import sympy as sp
import numpy as np

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb):

    state_symbols = []

    for i in range(1, projhyb["nspecies"]+1):
        state_symbols.append(sp.sympify(projhyb["species"][str(i)]["id"]))

    for i in range(1, projhyb["ncompartments"]+1):
        state_symbols.append(sp.sympify(projhyb["compartment"][str(i)]["id"]))

    print("state_symbols:", state_symbols)

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
            fstate = fstate_func(projhyb)

            anninp, rann, anninp_mat = anninp_rann_func(projhyb)

            DanninpDstate = derivativeXY(anninp, state_symbols)
            #TODO: substituicao dos val das variaveis
            #TODO: Calcular as derivadas antes e qui so substituir 

            print("DanninpDstate:", DanninpDstate)

            anninp_tensor = torch.tensor(anninp_mat, dtype=torch.float32)
            anninp_tensor = anninp_tensor.view(-1, 1)            
            print("anninp_tensor:", anninp_tensor)
            _, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

            #TODO: FIX THIS ANNINP NOT BEEING WELL CALCULATED


            DfDs = derivativeXY(fstate, state_symbols)
            
            '''
            val = []

            adp, asa, asp, aspp, atp, compartment, hs, hsp, nadp, nadph, phos, rann1, rann2, rann3, rann4, rann5, rann6, rann7, thr = symbols("adp asa asp aspp atp compartment hs hsp nadp nadph phos rann1 rann2 rann3 rann4 rann5 rann6 rann7 thr")
            for i in range(0, len(DfDs)):
                calc = DfDs[i].subs([(adp,1), (asa, 1), (asp, 1), (aspp, 1), (atp,1), (compartment, 1), (hs,1 ), (hsp,1 ), (nadp, 1), (nadph, 1), (phos, 1), (rann1, 1), (rann2, 1), (rann3, 1), (rann4, 1), (rann5, 1), (rann6, 1), (rann7, 1) , (thr, 1)])
                val = val + [calc]
            print("val:", val)
            '''

            DfDrann = derivativeXY(fstate, rann)

            print("DfDrann:", DfDrann)

            DrannDs = DrannDanninp * DanninpDstate

            fjac = (DfDs + DfDrann * DrannDs) * jac + DfDrann * DrannDw

            return fstate, fjac, None

        elif projhyb['mode'] == 3:
            anninp, rann, _ = anninp_rann_func(projhyb)

            fstate = fstate_func(projhyb)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None

     
def anninp_rann_func(projhyb):
    species_values = extract_species_values(projhyb)

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


        anninp.append(val_expr/max_expr)

        anninp_mat.append(val_mat/max_mat)

    for i in range(1, projhyb["mlm"]["ny"]+1):
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["id"])  # Ann outputs
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["val"])

        rann.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["val"]))

    totalsyms = symbols(totalsyms)

    print("anninp:", anninp)

    anninp_symbol = sp.sympify(anninp)

    return anninp_symbol, rann, anninp_mat


def extract_species_values(projhyb):
    species_values = {}
    for key, species in projhyb['species'].items():
        species_id = species['id']
        species_val = species['val']
        species_values[species_id] = species_val
        
    return species_values


def derivativeXY(X,Y):
    z = []
    print("X:", X)
    print("Y:", Y)
    print("len(X):", len(X))
    print("len(Y):", len(Y))

    for i in range(0, len(Y)):
        for j in range(0, len(X)):

            print("X[i]:", X[j])
            print("Y[j]:", Y[i])

            cal = diff(X[j], Y[i])
            print("cal:", cal)

            z = z + [cal]

    print("z:", z)
    print("len(z):", len(z))

    return z
