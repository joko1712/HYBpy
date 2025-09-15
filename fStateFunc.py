from __future__ import division

import json
import sympy as sp
from sympy import symbols, sympify, simplify, Matrix, Eq, Symbol
import numpy as np
from sympy.parsing.sympy_parser import parse_expr

def fstate_func(projhyb, values):
    Species = []
    species_dict = {}
    parameters_dict = {}

    if 'SYMBOL_CACHE' not in projhyb:
        projhyb['SYMBOL_CACHE'] = {}
    symbol_cache = projhyb['SYMBOL_CACHE']

    def get_symbol(name):
        if name not in symbol_cache:
            symbol_cache[name] = sp.Symbol(name)
        return symbol_cache[name]

    for i in range(1, projhyb["nspecies"]+1):
        species_name = projhyb["species"][str(i)]["id"]
        Species.append(get_symbol(species_name))
        species_dict[species_name] = get_symbol(species_name)
        projhyb["species"][str(i)]["dcomp"] = 0
        for m in range(1, projhyb["nraterules"]+1):
            if projhyb["raterules"][str(m)]["id"] == projhyb["species"][str(i)]["compartment"]:
                projhyb["species"][str(i)]["dcomp"] = sp.sympify(projhyb["raterules"][str(m)]["val"])

    Compartments = [get_symbol(projhyb["compartment"][str(i)]["id"]) for i in range(1, projhyb["ncompartment"]+1)]

    variables = {get_symbol(projhyb["mlm"]["x"][str(i)]["id"]): sp.sympify(projhyb["mlm"]["x"][str(i)]["val"])
                 for i in range(1, projhyb["mlm"]["nx"]+1)}

    output = {get_symbol(projhyb["mlm"]["y"][str(i)]["id"]): sp.sympify(projhyb["mlm"]["y"][str(i)]["val"])
              for i in range(1, projhyb["mlm"]["ny"]+1)}

    parametersvariables = {}
    for i in range(1, projhyb["nparameters"]+1):
        param_name = projhyb["parameters"][str(i)]["id"]
        parameters_dict[param_name] = get_symbol(param_name)
        parametersvariables[get_symbol(param_name)] = sp.sympify(projhyb["parameters"][str(i)]["val"])

    ruleassvariables = {}
    for i in range(1, projhyb["nruleAss"] + 1):
        rule_id = projhyb["ruleAss"][str(i)]["id"]
        rule_val = projhyb["ruleAss"][str(i)]["val"]
        parsed_expr = parse_rule_val(rule_val)
        ruleassvariables[get_symbol(rule_id)] = parsed_expr

    combined_dict = {**species_dict, **parameters_dict}

    Raterules = [get_symbol(projhyb["raterules"][str(i)]["id"]) for i in range(1, projhyb["nraterules"]+1)]
    fRaterules = [sp.sympify(projhyb["raterules"][str(i)]["val"]) for i in range(1, projhyb["nraterules"]+1)]

    ucontrol = [get_symbol(projhyb["control"][str(i)]["id"]) for i in range(1, projhyb["ncontrol"]+1)]

    rates = []
    for i in range(1, projhyb["nspecies"]+1):
        for j in range(1, projhyb["nreaction"]+1):
            if projhyb["reaction"][str(j)]["id"] in projhyb["outputs"]:
                nvalues = get_symbol(projhyb["reaction"][str(j)]["id"]) * projhyb["reaction"][str(j)]["Y"][str(i)]
            else:
                nvalues = sp.sympify(projhyb["reaction"][str(j)]["rate"], locals=combined_dict) * projhyb["reaction"][str(j)]["Y"][str(i)]
            rates.append(nvalues)

    fSpecies = []
    for i in range(1, projhyb["nspecies"]+1):
        species_id = projhyb["species"][str(i)]["id"]
        if species_id.startswith("chybrid"):
            continue
        rates_sum = sum(rates[(i-1)*projhyb["nreaction"]:i*projhyb["nreaction"]])
        fSpecies.append(
            rates_sum - (projhyb["species"][str(i)]["dcomp"] /
                         sp.sympify(projhyb["species"][str(i)]["compartment"])) *
            get_symbol(species_id)
        )

    State = Species + Raterules
    fState = fSpecies + fRaterules

    subout = {get_symbol(projhyb["mlm"]["y"][str(i)]["id"]): sp.sympify(projhyb["mlm"]["y"][str(i)]["val"])
              for i in range(1, projhyb["mlm"]["ny"]+1)}

    fState = sp.Matrix(fState).subs(subout)
    fState = fState.subs(ruleassvariables)
    fState = list(fState)

    return fState

'''
def fstate_func(projhyb,values):
    Species = []

    species_dict = {}
    parameters_dict = {}

    for i in range(1, projhyb["nspecies"]+1):
        Species.append(sp.sympify(projhyb["species"][str(i)]["id"]))
        species_name = projhyb["species"][str(i)]["id"]
        species_dict[species_name] = sp.Symbol(species_name)

        projhyb["species"][str(i)]["dcomp"] = 0

        for m in range(1, projhyb["nraterules"]+1):
            if projhyb["raterules"][str(m)]["id"] == projhyb["species"][str(i)]["compartment"]:
                projhyb["species"][str(i)]["dcomp"] = sp.sympify(
                    projhyb["raterules"][str(m)]["val"])

    Compartments = []

    for i in range(1, projhyb["ncompartment"]+1):
        Compartments.append(sp.sympify(projhyb["compartment"][str(i)]["id"]))

    variables = {}
    for i in range(1, projhyb["mlm"]["nx"]+1):
        variables[symbols(projhyb["mlm"]["x"][str(i)]["id"])] = sympify(
            projhyb["mlm"]["x"][str(i)]["val"])

    output = {}
    no = projhyb["mlm"]["ny"]
    for i in range(1, projhyb["mlm"]["ny"]+1):
        output[symbols(projhyb["mlm"]["y"][str(i)]["id"])] = sympify(
            projhyb["mlm"]["y"][str(i)]["val"])

    parametersvariables = {}
    for i in range(1, projhyb["nparameters"]+1):
        param_name = projhyb["parameters"][str(i)]["id"]
        parameters_dict[param_name] = sp.Symbol(param_name)
        parametersvariables[symbols(projhyb["parameters"][str(i)]["id"])] = sympify(
            projhyb["parameters"][str(i)]["val"])

    ruleassvariables = {}
    for i in range(1, projhyb["nruleAss"] + 1):
        rule_id = projhyb["ruleAss"][str(i)]["id"]
        rule_val = projhyb["ruleAss"][str(i)]["val"]
        
        parsed_expr = parse_rule_val(rule_val)
        
        ruleassvariables[sp.Symbol(rule_id)] = parsed_expr
    
        
    combined_dict = {**species_dict, **parameters_dict}

    Raterules = []
    fRaterules = []

    for i in range(1, projhyb["nraterules"]+1):
        Raterules.append(symbols(projhyb["raterules"][str(i)]["id"]))
        fRaterules.append(sympify(projhyb["raterules"][str(i)]["val"]))

    ucontrol = []
    for i in range(1, projhyb["ncontrol"]+1):
        ucontrol.append(symbols(projhyb["control"][str(i)]["id"]))

    rates = []


    for i in range(1, projhyb["nspecies"]+1):
        for j in range(1, projhyb["nreaction"]+1):

            if projhyb["reaction"][str(j)]["id"] in projhyb["outputs"]:
                nvalues = sympify(projhyb["reaction"][str(j)]["id"]) * projhyb["reaction"][str(j)]["Y"][str(i)]

            else: 
                nvalues = sympify(projhyb["reaction"][str(j)]["rate"], locals=combined_dict) * projhyb["reaction"][str(j)]["Y"][str(i)]

            rates.append(sympify(nvalues))


    fSpecies = []
    for i in range(1, projhyb["nspecies"]+1):
        species_id = projhyb["species"][str(i)]["id"]
        if species_id.startswith("chybrid"):
            continue
        rates_sum = sum(rates[(i-1)*projhyb["nreaction"]:i*projhyb["nreaction"]])
        fSpecies.append(
            rates_sum - (projhyb["species"][str(i)]["dcomp"]/sympify(projhyb["species"]
                        [str(i)]["compartment"])) * sp.Symbol(projhyb["species"][str(i)]["id"])
        )



    nyparameters= {}
    for i in range(1, projhyb["mlm"]["ny"]+1):
        nyparameters[symbols(projhyb["mlm"]["y"][str(i)]["id"])] = values[projhyb["mlm"]["y"][str(i)]["val"]]

    State = Species + Raterules

    fState = fSpecies + fRaterules

    
    subout = {}
    for i in range(1, projhyb["mlm"]["ny"]+1):
        subout[symbols(projhyb["mlm"]["y"][str(i)]["id"])] = sympify(
            projhyb["mlm"]["y"][str(i)]["val"])
    
    fState = [expr.subs(subout) for expr in fState]

    fState = [expr.subs(ruleassvariables) for expr in fState]

'''


'''
    nspecies = data["nspecies"]
    nraterules = data["nraterules"]
    nstate = nspecies + nraterules
    w = []


    print("State", State)
    print("fState", fState)
    print("rates", rates)
    print("anninp", anninp)
    print("rann", rann)
    print("ucontrol", ucontrol)
    print("w", w)


    for i in range(1, len(anninp)+1):
        anninp.append(anninp[i-1]/data["mlm"]["x"][str(i)]["max"])

    # CALLED FROM METGHODS
    print("anninp", anninp)
    DanninpDstate = Matrix([anninp]).jacobian(Matrix([State]))
    print("DanninpDstate", DanninpDstate)

    DanninpDucontrol = Matrix([anninp]).jacobian(Matrix([ucontrol]))
    print("DanninpDucontrol", DanninpDucontrol)

    DrDs = Matrix([rates]).jacobian(Matrix([State]))
    print("DrDs", DrDs)


    DfDs = Matrix([fState]).jacobian(Matrix([State]))
    print("DfDs", DfDs)

    DfDrann = Matrix([fState]).jacobian(Matrix([rann]))
    print("DfDrann", DfDrann)

    rann = ["vm11", "vm2f", "katpase", "vm3f", "vm4f", "knadph", "vm5"]

    DrDrann = Matrix([rates]).jacobian(Matrix([rann]))
    print("DrDrann", DrDrann)

    
    return fState

'''

def parse_rule_val(rule_val):
    # Extract all unique variables from the string
    symbols_in_expr = {}
    for term in rule_val.replace('*', ' ').replace('/', ' ').split():
        symbols_in_expr[term] = sp.Symbol(term)
    
    # Parse the expression using these symbols
    parsed_expr = parse_expr(rule_val, local_dict=symbols_in_expr)
    return parsed_expr