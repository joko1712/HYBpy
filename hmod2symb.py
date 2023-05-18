import json
from sympy import symbols

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

symbols(totalsyms)
print(anninp)
print(rann)
print(totalsyms)

# Species data
Species = []
