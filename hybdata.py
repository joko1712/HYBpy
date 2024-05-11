import json
import re
def hybdata(filename):

    with open(filename) as f:
        lines = f.readlines()
    f.close()

    species_dict = {}
    compartment_dict = {}
    parameters_dict = {}
    ruleAss_dict = {}
    var_dict = {}
    event_dict = {}
    reaction_dict = {}
    raterules_dict = {}
    time_dict = {}
    control_dict = {}
    nx_dict = {}
    ny_dict = {}
    id = ""
    val = 0
    vals = ""
    var = ""
    compartment = ""
    fixed = 0
    min = 0
    max = 0
    isres = 0
    reaction = ""
    cond = ""
    rate = ""
    tau = 0
    constant = 0
    symbolic = []
    options = ""
    layer = 0
    xfun = ""
    yfun = ""
    nx = 0
    ny = 0
    datasource = 0
    datafun = ""
    mode = 0
    method = 0
    jacobian = 0
    hessian = 0
    derivativecheck = ""
    niter = 0
    niteroptim = 0
    nstep = 0
    display = ""
    boostrap = 0
    nensemble = 0
    crossval = 0
    adalfa = 0
    admini = 0
    addrop = 0
    projId = ""

    correctreac = 0
    correctl = 1

    for line in lines:
        line = line.strip()

        # Eleminate the ; at the end of the line
        line = line.replace(";", "")
        line = line.replace(" ", "")

        # Check if line is last line
        if line == 'end':
            break

        # Check if line is a comment or if there is a commenqt in the line and erase it
        if "%" in line:
            line = line[0:line.find("%")]

        if ".id=" in line:
            match = re.search(r"(\w+)\.id='(\w+)'", line)
            if match:
                projId = match.group(2)

        # Check if is a tspan line and create the array for the json file
        if "tspan" in line:
            x = line[line.find("=")+1:line.find(":")]

            # Check if the y value is a single digit or a double digit ?? Maybe more if so loop
            if (line[line.find(":")+2].isnumeric() == True):
                y = line[line.find(":")+1:line.find(":")+3]
            else:
                y = line[line.find(":")+1]

            # Check if the z value is a single digit or a double digit ?? Maybe more if so loop
            if (line[-3].isnumeric() == True):
                z = line[-3:len(line)]
            else:
                z = line[-2:len(line)]

            line = line[line.find(".")+1:len(line)]

            # tspan: [0,1,2,3,4,5,,6,7,8,9,10]
            listArray = list(range(int(x), int(z)+1, int(y)))

        # Check if line is the number of species
        if "nspecies" in line:
            # Get the number of species
            nspecies = int(line[line.find("=")+1:len(line)+1])

        # Check if line is a species
        if "species(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the species id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the species value
            if "val" in line:
                val = line[line.find("=")+1:len(line)+1]

            # Get the species compartment
            if "compartment" in line:
                compartment = line[line.find("=")+2:len(line)-1]

            # Get see if the species is fixed
            if "fixed" in line:
                fixed = line[line.find("=")+1:len(line)+1]

            # Get the species min value
            if "min" in line:
                min = line[line.find("=")+1:len(line)+1]

            # Get the species max value
            if "max" in line:
                max = line[line.find("=")+1:len(line)+1]

            if "isres" in line:
                isres = line[line.find("=")+1:len(line)+1]

            species_dict[i] = {
                "id": id,
                "val": float(val),
                "compartment": compartment,
                "fixed": int(fixed),
                "min": int(min),
                "max": int(max),
                "isres": int(isres),
                "dcomp": 0,
            }

        if "ncompartments" in line:
            # Get the number of compartments
            ncompartments = int(line[line.find("=")+1:len(line)+1])

        if "compartment(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the compartments id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the compartments value
            if "val" in line:
                val = line[line.find("=")+1:len(line)+1]

            # Get the compartments min value
            if "min" in line:
                min = line[line.find("=")+1:len(line)+1]

            # Get the compartments max value
            if "max" in line:
                max = line[line.find("=")+1:len(line)+1]

            # Get the ???????
            if "isres" in line:
                isres = line[line.find("=")+1:len(line)+1]

            compartment_dict[i] = {
                "id": id,
                "val": val,
                "min": int(min),
                "max": int(max),
                "isres": int(isres),
            }

        if "nparameters" in line:
            # Get the number of parameters
            nparameters = int(line[line.find("=")+1:len(line)+1])

        if "parameters(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the parameters id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the parameters value
            if "val" in line:
                val = line[line.find("=")+2:len(line)-1]

            # Get the parameters min value
            if "reaction" in line:
                reaction = line[line.find("=")+2:len(line)-1]

            parameters_dict[i] = {
                "id": id,
                "val": float(val),
                "reaction": reaction,
            }

        if "nruleAss" in line:
            # Get the number of Assignment rules
            nruleAss = int(line[line.find("=")+1:len(line)+1])

        if "ruleAss(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the Assignment rules id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the Assignment rules value
            if "val" in line:
                val = line[line.find("=")+2:len(line)-1]

            ruleAss_dict[i] = {
                "id": id,
                "val": val,
            }

        if "nevent" in line:
            # Get the number of events
            nevent = int(line[line.find("=")+1:len(line)+1])

        if "event(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the events id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the events value
            if "cond" in line:
                cond = line[line.find("=")+2:len(line)-1]

            if ".v" in line:
                line = line[line.find(".v"):len(line)]
                j = line[line.find("(")+1:line.find(")")]
                if "r" in line:
                    var = line[line.find(f"r({j})=")+6:len(line)-1]
                if "l" in line:
                    vals = line[line.find(f"l({j})=")+6:len(line)-1]

                var_dict[j] = {
                    "var": var,
                    "val": vals,
                }

            event_dict[i] = {
                "id": id,
                "cond": cond,
                "val": var_dict,
            }

        if "nreaction" in line:
            # Get the number of reactions
            nreaction = int(line[line.find("=")+1:len(line)+1])

        if "reaction(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the reactions id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the rate of the reactio
            if "rate" in line:
                rate = line[line.find("=")+2:len(line)-1]

            # Get the Y of the reaction
            if "Y" in line:
                j = line[line.find("=")+2:len(line)-1]
                y_values = j.split(",")
                y_values = [int(i.strip().strip('\"')) for i in y_values]

                y_dict = {str(index+1): value for index,
                        value in enumerate(y_values)}

                reaction_dict[i] = {
                    "id": id,
                    "rate": rate,
                    "Y": y_dict,
                }

        if "nraterules" in line:
            # Get the number of rate rules
            nraterules = int(line[line.find("=")+1:len(line)+1])

        if "raterules(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the rate rules id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            # Get the rate rules value
            if "min" in line:
                min = line[line.find("=")+1:len(line)+1]

            if "max" in line:
                max = line[line.find("=")+1:len(line)+1]

            if "val" in line:
                val = line[line.find("=")+2:len(line)-1]

            if "isres" in line:
                isres = line[line.find("=")+1:len(line)+1]

            raterules_dict[i] = {
                "id": id,
                "min": int(min),
                "max": int(max),
                "val": val,
                "isres": int(isres),
            }

        if "time." in line:

            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            if "min" in line:
                min = line[line.find("=")+1:len(line)+1]

            if "max" in line:
                max = line[line.find("=")+1:len(line)+1]

            if "TAU" in line:
                tau = line[line.find("=")+1:len(line)]

            time_dict = {
                "id": id,
                "min": int(min),
                "max": int(max),
                "TAU": float(tau),
                "tspan": listArray,
            }

        if "ncontrol" in line:
            # Get the number of controls
            ncontrol = int(line[line.find("=")+1:len(line)+1])

        if "control(" in line:
            i = line[line.find("(")+1:line.find(")")]

            # Get the controls id
            if "id" in line:
                id = line[line.find("=")+2:len(line)-1]

            if "min" in line:
                min = line[line.find("=")+1:len(line)+1]

            if "max" in line:
                max = line[line.find("=")+1:len(line)+1]

            if "val" in line:
                val = line[line.find("=")+1:len(line)]

            if "constant" in line:
                constant = line[line.find("=")+1:len(line)+1]

            control_dict[i] = {
                "id": id,
                "min": int(min),
                "max": int(max),
                "val": val,
                "constant": int(constant),
            }

        if "fun_control" in line:
            funControl = line[line.find("=")+1:len(line)+1]

        if "fun_event" in line:
            funEvent = []
            j = line[line.find("=")+1:len(line)]
            if j == "[]":
                funEvent = []
            else:
                funEvent = j.split(", ")
                funEvent = [val.replace('"', '') for val in y]

        if "mlm" in line:

            if "mlm.id" in line:
                id = line[line.find("=")+2:len(line)-1]

            if "options" in line:
                start = line.find('[')
                end = line.find(']') + 1
                text = line[start:end]
                spaced_text = ' '.join(text.strip('[]'))
                numbers = spaced_text.split()
                numbers = [int(num) for num in numbers]
                options = numbers

            if "layer" in line:
                layer = line[line.find("=")+1:len(line)+1]

            if "xfun" in line:
                xfun = line[line.find("=")+1:len(line)+1]

            if "yfun" in line:
                yfun = line[line.find("=")+1:len(line)+1]

            if "mlm.nx" in line:
                nx = line[line.find("=")+1:len(line)+1]

            if "mlm.x(" in line:
                i = line[line.find("(")+1:line.find(")")]

                if "id" in line:
                    id = line[line.find("=")+2:len(line)-1]

                if "min" in line:
                    min = line[line.find("=")+1:len(line)+1]

                if "max" in line:
                    max = line[line.find("=")+1:len(line)+1]

                if "val" in line:
                    val = line[line.find("=")+2:len(line)-1]

                nx_dict[i] = {
                    "id": id,
                    "min": float(min),
                    "max": float(max),
                    "val": val,
                }

            if "ny" in line:
                ny = line[line.find("=")+1:len(line)+1]

            if "mlm.y(" in line:
                i = line[line.find("(")+1:line.find(")")]

                if "id" in line:
                    id = line[line.find("=")+2:len(line)-1]

                if "min" in line:
                    min = line[line.find("=")+1:len(line)+1]

                if "max" in line:
                    max = line[line.find("=")+1:len(line)+1]

                if "val" in line:
                    val = line[line.find("=")+2:len(line)-1]

                ny_dict[i] = {
                    "id": id,
                    "min": float(min),
                    "max": float(max),
                    "val": val,
                }

            mlm_dict = {
                "id": id,
                "neuron": 1,
                "options": options,
                "layer": layer,
                "xfun": xfun,
                "yfun": yfun,
                "nx": int(nx),
                "ny": int(ny),
                "x": nx_dict,
                "y": ny_dict,
            }

        if "symbolic" in line:
            symbolic.append(line[line.find("=")+2:len(line)-1])

        if "datasource" in line:
            datasource = int(line[line.find("=")+1:len(line)])

        if "datafun" in line:
            datafun = line[line.find("=")+1:len(line)]

        if "mode" in line:
            mode = int(line[line.find("=")+1:len(line)])

        if "method" in line:
            method = int(line[line.find("=")+1:len(line)])

        if "jacobian" in line:
            jacobian = int(line[line.find("=")+1:len(line)])

        if "hessian" in line:
            hessian = int(line[line.find("=")+1:len(line)])

        if "derivativecheck" in line:
            derivativecheck = line[line.find("=")+2:len(line)-1]

        if "niter" in line:
            niter = int(line[line.find("=")+1:len(line)])

        if "niteroptim" in line:
            niteroptim = int(line[line.find("=")+1:len(line)])

        if "nstep" in line:
            nstep = int(line[line.find("=")+1:len(line)])

        if "display" in line:
            display = line[line.find("=")+2:len(line)-1]
            if display == "off":
                display = 0
            else:
                display = 2
        if "bootstrap" in line:
            bootstrap = int(line[line.find("=")+1:len(line)])

        if "nensemble" in line:
            nensemble = int(line[line.find("=")+1:len(line)])

        if "crossval" in line:
            crossval = int(line[line.find("=")+1:len(line)])

        if "adalfa" in line:
            adalfa = float(line[line.find("=")+1:len(line)])

        if "admini" in line:
            admini = int(line[line.find("=")+1:len(line)])

        if "addrop" in line:
            addrop = int(line[line.find("=")+1:len(line)])

    dict = {
        "id": projId,
        "tspan": listArray,
        "nspecies": nspecies,
        "species": species_dict,
        "ncompartment": ncompartments,
        "compartment": compartment_dict,
        "nparameters": nparameters,
        "parameters": parameters_dict,
        "nruleAss": nruleAss,
        "ruleAss": ruleAss_dict,
        "nevent": nevent,
        "event": event_dict,
        "nreaction": nreaction,
        "reaction": reaction_dict,
        "nraterules": nraterules,
        "raterules": raterules_dict,
        "time": time_dict,
        "ncontrol": ncontrol,
        "control": control_dict,
        "fun_control": funControl,
        "fun_event": funEvent,
        "mlm": mlm_dict,
        "symbolic": symbolic,
        "datasource": datasource,
        "datafun": datafun,
        "mode": mode,
        "method": method,
        "jacobian": jacobian,
        "hessian": hessian,
        "derivativecheck": derivativecheck,
        "niter": niter,
        "niteroptim": niteroptim,
        "nstep": nstep,
        "display": display,
        "bootstrap": bootstrap,
        "nensemble": nensemble,
        "crossval": crossval,
        "adalfa": adalfa,
        "admini": admini,
        "addrop": addrop,
    }

    with open("sample.json", "w") as outfile:
        json.dump(dict, outfile)
