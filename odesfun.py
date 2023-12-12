from fStateFunc import fstate_func 
import torch
import sympy as sp
from sympy import symbols, sympify, simplify, Matrix, Eq
import numpy as np

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb):

    current_state_dict = ann.state_dict()
    print("Current State Dict:", current_state_dict)

    new_state_dict = {}
    for param_tensor in ann.state_dict():
        if "w" in param_tensor:
            new_state_dict[param_tensor] = torch.randn(*current_state_dict[param_tensor].shape)
        elif "b" in param_tensor:
            new_state_dict[param_tensor] = torch.zeros_like(current_state_dict[param_tensor])

    state_symbols = sp.symbols(['state_{}'.format(i) for i in range(len(state))])

    ann.load_state_dict(new_state_dict)

    if jac is None and hess is None:
        fstate = fstate_func(projhyb)
        anninp, rann = anninp_rann_func(projhyb)

        return fstate, None, None

    else:

        if projhyb['mode'] == 1:
            fstate = fstate_func(projhyb)

            anninp, rann = anninp_rann_func(projhyb)
            
            DanninpDstate = anninp.jacobian(state_symbols)
            print("DanninpDstate:", DanninpDstate)

            anninp_numerical = [expr.evalf() if isinstance(expr, sp.Expr) else float(expr) for expr in anninp]
            anninp_tensor = torch.tensor(anninp_numerical, dtype=torch.float32)

            anninp_tensor = anninp_tensor.view(-1, 1)            
            _, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor)

            #TODO: FIX THIS ANNINP NOT BEEING WELL CALCULATED


            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            DrannDs = DrannDanninp * DanninpDstate

            fjac = (DfDs + DfDrann * DrannDs) * jac + DfDrann * DrannDw

            return fstate, fjac, None

        elif projhyb['mode'] == 3:
            anninp, rann = anninp_rann_func(projhyb)

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
    rann = []

    for i in range(1,  projhyb["mlm"]["nx"]+1):
        totalsyms.append(projhyb["mlm"]["x"][str(i)]["id"])

        val_expr = sp.sympify(projhyb["mlm"]["x"][str(i)]["val"])
        max_expr = sp.sympify(projhyb["mlm"]["x"][str(i)]["max"])

        val = val_expr.evalf(subs=species_values)
        max_val = max_expr.evalf(subs=species_values)


        anninp.append(val/max_val)

    for i in range(1, projhyb["mlm"]["ny"]+1):
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["id"])  # Ann outputs
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["val"])

        rann.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["val"]))

    totalsyms = symbols(totalsyms)


    for i in range(1, len(anninp)+1):
        anninp.append(anninp[i-1]/projhyb["mlm"]["x"][str(i)]["max"])

    print("anninp:", anninp)
    anninp_matrix = sp.Matrix(anninp)

    return anninp_matrix, rann


def extract_species_values(projhyb):
    species_values = {}
    for key, species in projhyb['species'].items():
        species_id = species['id']
        species_val = species['val']
        species_values[species_id] = species_val
    return species_values