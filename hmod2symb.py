import json
import sympy as sp
from sympy import symbols, sympify, simplify, Matrix, Eq
import numpy as np

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
    totalsyms.append(data["mlm"]["x"][str(i)]["id"])

    val = sp.sympify(data["mlm"]["x"][str(i)]["val"])
    max = sp.sympify(data["mlm"]["x"][str(i)]["max"])

    anninp.append(val/max)

for i in range(1, data["mlm"]["ny"]+1):
    totalsyms.append(data["mlm"]["y"][str(i)]["id"])  # Ann outputs
    totalsyms.append(data["mlm"]["y"][str(i)]["val"])

    expr = symbols(data["mlm"]["y"][str(i)]["val"])
    rann.append(expr)

# This converts all string identifiers to symbolic ones
totalsyms = symbols(totalsyms)

Species = []

for i in range(1, data["nspecies"]+1):
    Species.append(sp.sympify(data["species"][str(i)]["id"]))

    data["species"][str(i)]["dcomp"] = 0

    for m in range(1, data["nraterules"]+1):
        if data["raterules"][str(m)]["id"] == data["species"][str(i)]["compartment"]:
            data["species"][str(i)]["dcomp"] = sp.sympify(
                data["raterules"][str(m)]["val"])

Compartments = []

for i in range(1, data["ncompartments"]+1):
    Compartments.append(sp.sympify(data["compartment"][str(i)]["id"]))

for i in range(1, data["mlm"]["nx"]+1):
    var_id = data['mlm']['x'][str(i)]['id']
    globals()[var_id] = None
    exec(f"{var_id} = sp.sympify(data['mlm']['x'][str(i)]['val'])")


for i in range(1, data["mlm"]["ny"]+1):
    var_id = data['mlm']['y'][str(i)]['id']
    globals()[var_id] = None
    exec(f"{var_id} = sp.sympify(data['mlm']['y'][str(i)]['val'])")

for i in range(1, data["nparameters"]+1):
    if data["parameters"][str(i)]["reaction"] == "global":
        var_id = data['parameters'][str(i)]['id']
        globals()[var_id] = None
        exec(f"{var_id} = sp.sympify(data['parameters'][str(i)]['val'])")


for i in range(1, data["nruleAss"]+1):
    var_id = data['ruleAss'][str(i)]['id']
    globals()[var_id] = None
    exec(f"{var_id} = sp.sympify(data['ruleAss'][str(i)]['val'])")


Raterules = []
fRaterules = []

for i in range(1, data["nraterules"]+1):
    Raterules.append(symbols(data["raterules"][str(i)]["id"]))
    fRaterules.append(sympify(data["raterules"][str(i)]["val"]))

ucontrol = []
for i in range(1, data["ncontrol"]+1):
    ucontrol.append(symbols(data["control"][str(i)]["id"]))

stoichm = []
reaction = []

for i in range(1, data["nreaction"]+1):
    for j in range(1, data["nparameters"]+1):
        if data["parameters"][str(j)]["reaction"] == data["reaction"][str(i)]["id"]:
            var_id = data['parameters'][str(j)]['id']
            globals()[var_id] = None
            exec(f"{var_id} = sp.sympify(data['parameters'][str(j)]['val'])")

    reaction.append(sympify(data["reaction"][str(i)]["rate"]))
    nval = sp.sympify(data["reaction"][str(i)]["Y"])
    stoichm.append(nval)

stoichm = np.array(stoichm)

rates = []

for i in range(1, data["nspecies"]+1):
    rates.append(np.sum([r * s for r, s in zip(reaction, stoichm[:, i-1])]))

fSpecies = []

for i in range(1, data["nspecies"]+1):
    fSpecies.append(rates[i-1] - (data["species"][str(i)]["dcomp"] / sp.sympify(
        data["species"][str(i)]["compartment"]) * sp.sympify(data["species"][str(i)]["id"])))

State = Species + Raterules
fState = Matrix(fSpecies + fRaterules)
fState = fState.subs(list(zip(State, totalsyms)))

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
print("reaction", reaction)


A = Matrix([symbols("a")**2, symbols("b")**3])

B = Matrix(["a", "b"])

F = A.jacobian(B)

for i in range(1, len(anninp)+1):
    anninp.append(anninp[i-1]/data["mlm"]["x"][str(i)]["max"])


print("anninp", anninp)
DanninpDstate = Matrix([anninp]).jacobian(Matrix([State]))

print("DanninpDstate", DanninpDstate)

DanninpDucontrol = Matrix([anninp]).jacobian(Matrix([ucontrol]))

print("DanninpDucontrol", DanninpDucontrol)

DrDs = Matrix([rates]).jacobian(Matrix([State]))
print("DrDs", DrDs)

DrDrann = Matrix([rates]).jacobian(Matrix([rann]))
print("DrDrann", DrDrann)

DfDs = Matrix([fState]).jacobian(Matrix([State]))
print("DfDs", DfDs)

DfDrann = Matrix([fState]).jacobian(Matrix([rann]))
print("DfDrann", DfDrann)
