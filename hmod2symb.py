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

    rann.append(sp.sympify(data["mlm"]["y"][str(i)]["val"]))

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


Raterules = []
fRaterules = []

for i in range(1, data["nraterules"]+1):
    Raterules.append(symbols(data["raterules"][str(i)]["id"]))
    fRaterules.append(sympify(data["raterules"][str(i)]["val"]))

ucontrol = []
for i in range(1, data["ncontrol"]+1):
    ucontrol.append(symbols(data["control"][str(i)]["id"]))

rates = []


for i in range(1, data["nspecies"]+1):
    for j in range(1, data["nreaction"]+1):
        nvalues = sympify(data["reaction"][str(
            j)]["rate"]) * data["reaction"][str(j)]["Y"][str(i)]
        rates.append(sympify(nvalues))


fSpecies = []
for i in range(1, data["nspecies"]+1):
    rates_sum = sum(rates[(i-1)*data["nreaction"]:i*data["nreaction"]])
    fSpecies.append(
        rates_sum - (data["species"][str(i)]["dcomp"]/sympify(data["species"]
                     [str(i)]["compartment"])) * sympify(data["species"][str(i)]["id"])
    )


State = Species + Raterules
fState = fSpecies + fRaterules

# create unified dictionary
unified_dict = {**variables, **output, **
                parametersvariables, **ruleassvariables}

# replace the variables in fState with actual values
fState = [f.subs(unified_dict) for f in fState]

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


DfDs = Matrix([fState]).jacobian(Matrix([State]))
print("DfDs", DfDs)

DfDrann = Matrix([fState]).jacobian(Matrix([rann]))
print("DfDrann", DfDrann)

rann = ["vm11", "vm2f", "katpase", "vm3f", "vm4f", "knadph", "vm5"]

DrDrann = Matrix([rates]).jacobian(Matrix([rann]))
print("DrDrann", DrDrann)
