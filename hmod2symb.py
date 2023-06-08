import json
from sympy import symbols, sympify, simplify

with open("sample.json", "r") as read_file:
    data = json.load(read_file)

# Create the symbolic variables
totalsyms = ["t", "dummyarg1", "dummyarg2", "w"]

for i in range(1, data["nspecies"]+1):
    totalsyms.append(data["species"][str(i)]["id"])  # Species

for i in range(1, data["ncompartments"]+1):
    totalsyms.append(data["compartment"][str(i)]["id"])  # Compartments

for i in range(1, data["nparameters"]+1):
    totalsyms.append(data["parameters"][str(i)]["id"])  # Parameters

for i in range(1, data["nruleAss"]+1):
    totalsyms.append(data["ruleAss"][str(i)]["id"])  # RuleAss

for i in range(1, data["nreaction"]+1):
    totalsyms.append(data["reaction"][str(i)]["id"])  # Reaction

for i in range(1, data["ncontrol"]+1):
    totalsyms.append(data["control"][str(i)]["id"])  # Control

anninp = []
rann = []

for i in range(1,  data["mlm"]["nx"]+1):
    totalsyms.append(data["mlm"]["x"][str(i)]["id"])  # Ann inputs

    # Create a symbolic expression
    expr = symbols(data["mlm"]["x"][str(i)]["val"])
    sym_expr = expr / data["mlm"]["x"][str(i)]["max"]  # Perform division
    anninp.append(sym_expr)

for i in range(1, data["mlm"]["ny"]+1):
    totalsyms.append(data["mlm"]["y"][str(i)]["id"])  # Ann outputs
    totalsyms.append(data["mlm"]["y"][str(i)]["val"])

    expr = symbols(data["mlm"]["y"][str(i)]["val"])
    rann.append(expr)

# This converts all string identifiers to symbolic ones
totalsyms = symbols(totalsyms)

# Species data
Species = []

for i in range(1, data["nspecies"]+1):
    Species.append(symbols(data["species"][str(i)]["id"]))
    for m in range(1, data["nraterules"]+1):
        if data["raterules"][str(m)]["id"] == data["species"][str(i)]["compartment"]:
            data["species"][str(i)]["dcomp"] = sympify(
                data["raterules"][str(m)]["val"])

ncompartments = data["ncompartments"]
Compartments = []

for i in range(1, ncompartments+1):
    Compartments.append(symbols(data["compartment"][str(i)]["id"]))

variables = {}
for i in range(1, data["mlm"]["nx"]+1):
    variables[symbols(data["mlm"]["x"][str(i)]["id"])] = sympify(
        data["mlm"]["x"][str(i)]["val"])

output = {}
no = data["mlm"]["ny"]
for i in range(1, data["mlm"]["ny"]+1):
    output[symbols(data["mlm"]["y"][str(i)]["id"])] = sympify(
        data["mlm"]["y"][str(i)]["val"])

parametersvariables = {}
for i in range(1, data["nparameters"]+1):
    parametersvariables[symbols(data["parameters"][str(i)]["id"])] = sympify(
        data["parameters"][str(i)]["val"])

ruleassvariables = {}
for i in range(1, data["nruleAss"]+1):
    ruleassvariables[symbols(data["ruleAss"][str(i)]["id"])] = sympify(
        data["ruleAss"][str(i)]["val"])

nraterules = data["nraterules"]
Raterules = []
fRaterules = []

for i in range(1, nraterules+1):
    Raterules.append(symbols(data["raterules"][str(i)]["id"]))
    fRaterules.append(sympify(data["raterules"][str(i)]["val"]))

ncontrol = data["ncontrol"]
ucontrol = []
for i in range(1, ncontrol+1):
    ucontrol.append(symbols(data["control"][str(i)]["id"]))

stoichm = []

reaction = {}

for i in range(1, data["nreaction"]+1):
    reaction[str(i)] = symbols(data["reaction"][str(i)]["rate"])

ratefuns = []
ratevars = []

for i in range(1, data["nraterules"]+1):
    ratefuns.append(sympify(data['raterules'][str(i)]['val']))
    ratevars.append(symbols(data['raterules'][str(i)]['id']))

rates = {}

for i in range(1, data['nspecies']+1):
    rates_sum = 0
    for j in range(1, data["nreaction"]+1):
        rates_sum += reaction[str(j)] * \
            int(sympify(data["reaction"][str(j)]["Y"][str(i)]))
    rates[str(i)] = rates_sum

fSpecies = []

for i in range(1, data['nspecies']+1):
    species_value = rates[str(i)] - (float(sympify(data['species'][str(i)]['dcomp'])) / symbols(
        data['species'][str(i)]['compartment']) * symbols(data['species'][str(i)]['id']))
    fSpecies.append(species_value)

State = Species + Raterules
fState = fSpecies + fRaterules

# create unified dictionary
unified_dict = {**variables, **output, **
                parametersvariables, **ruleassvariables}

# replace the variables in fState with actual values
fState = [f.subs(unified_dict) for f in fState]

nstate = data['nspecies'] + data['nraterules']

w = []

print("State", State)
print("fState", fState)
print("rates", rates)
print("anninp", anninp)
print("rann", rann)
print("ucontrol", ucontrol)
print("w", w)
print("reaction", reaction)
