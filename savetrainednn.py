import numpy as np
import h5py
import torch
import torch.nn as nn
import re


def saveNN(modloc, inputs, hybout, orfile, nfile, orderedweights, ann):
    model = load_model_from_h5(ann, modloc)

    weights = list(model.parameters())
    orderedweights = np.concatenate([w.detach().cpu().numpy().flatten() for w in weights])

    nlayers = len(model.layers)
    nin = len(inputs)
    equations = []
    H = []
    wcount = 0
    
    with open(orfile, 'r') as f:
        content = f.read()
        
    regexPrefix = re.compile(r'(\w+)\.nspecies=')
    uniquePrefixes = set(regexPrefix.findall(content))
    prefix = uniquePrefixes.pop() if uniquePrefixes else 'mlm'

    for layer in range(nlayers):
        H.append([])
        layer_weights = model.layers[layer].w.data.cpu().numpy().flatten()
        layer_biases = model.layers[layer].b.data.cpu().numpy().flatten()
        nH = len(layer_biases)
        
        
        if layer == 0:
            for node in range(nH):
                H[layer].append("")
                for n in range(nin):
                    if n == 0:
                        H[layer][node] = H[layer][node] + "w" + str(1 + node + nH * n) + "*" + inputs[n]
                    else:
                        H[layer][node] = H[layer][node] + "+w" + str(1 + node + nH * n) + "*" + inputs[n]

                    if n == nin - 1:
                        H[layer][node] = H[layer][node] + "+w" + str(1 + node + nH * (n + 1))
                        H[layer][node] = "tanh(" + H[layer][node] + ")"

            wcount = wcount + len(H[layer]) * len(inputs) + len(H[layer])

        elif layer > 0 and layer < nlayers - 1:
            for node in range(nH):
                H[layer].append("")
                for n in range(len(H[layer - 1])):
                    if n == 0:
                        H[layer][node] = H[layer][node] + "w" + str(wcount + 1 + node + nH * n) + "*" + H[layer - 1][n]
                    else:
                        H[layer][node] = H[layer][node] + "+w" + str(wcount + 1 + node + nH * n) + "*" + H[layer - 1][n]

                    if n == len(H[layer - 1]) - 1:
                        H[layer][node] = H[layer][node] + "+w" + str(wcount + 1 + node + nH * (n + 1))
                        H[layer][node] = "tanh(" + H[layer][node] + ")"

            wcount = wcount + len(H[layer]) * len(H[layer - 1]) + len(H[layer])

        elif layer == nlayers - 1:
            for node in range(nH):
                H[layer].append("")
                for n in range(len(H[layer - 1])):
                    if n == 0:
                        H[layer][node] = H[layer][node] + "w" + str(wcount + 1 + node + nH * n) + "*" + H[layer - 1][n]
                    else:
                        H[layer][node] = H[layer][node] + "+w" + str(wcount + 1 + node + nH * n) + "*" + H[layer - 1][n]

                    if n == len(H[layer - 1]) - 1:
                        H[layer][node] = H[layer][node] + "+w" + str(wcount + 1 + node + nH * (n + 1))

            wcount = wcount + len(H[layer]) * len(H[layer - 1]) + len(H[layer])

    equations = H[-1]

    if len(equations) < len(hybout):
        raise ValueError("The number of generated equations is less than the number of hybrid outputs (hybout). Ensure the model architecture matches the expected outputs.")

    with open(orfile, 'r') as f, open(nfile, 'w') as h:
        lines = f.readlines()

        opos = orfile.find(".hmod")
        npos = nfile.find(".hmod")

        nbasepar = 0
        nbaseAss = 0
        nhybpar = 0
        nhybAss = 0
        skip = 0
        subAss = 0
        outhyb = 0
        Asscount = 0
        hcount = 1

        for line in lines:
            if ".nparameters=" in line:
                pos = line.index("=")
                nbasepar = int(line[pos + 1:].strip().strip(';'))
                nhybpar = nbasepar

            if ".nruleAss" in line:
                pos = line.index("=")
                nbaseAss = int(line[pos + 1:].strip().strip(';'))
                nhybAss = nbaseAss

            if (".ruleAss(" in line) and (").id" in line):
                if any(item in line for item in hybout):
                    subAss += 1

        for line in lines:
            if ".parameters(" + str(nbasepar) + ").reaction" in line:
                h.write(line)
                for w in orderedweights:
                    nhybpar += 1
                    h.write(f'{prefix}.parameters({nhybpar}).id="w{wcount - len(orderedweights) + 1 }";\n')
                    h.write(f'{prefix}.parameters({nhybpar}).val={w};\n')
                    h.write(f'{prefix}.parameters({nhybpar}).reaction="global";\n')
                    wcount += 1
                for i in range(len(H[0])):
                    nhybpar += 1
                    h.write(f'{prefix}.parameters({nhybpar}).id="H{i+1}";\n')
                    h.write(f'{prefix}.parameters({nhybpar}).val="0";\n')
                    h.write(f'{prefix}.parameters({nhybpar}).reaction="global";\n')

            elif ".parameters(" in line and ").id" in line:
                if any(item in line for item in hybout):
                    h.write(line)
                    outhyb = 1
                else:
                    h.write(line)

            elif ".parameters(" in line and ").reaction" in line:
                if outhyb == 1:
                    outhyb = 0
                    h.write(line.replace("local", "global"))
                else:
                    h.write(line)

            elif ".ruleAss(" + str(nbaseAss) + ").val" in line or "nruleAss=0" in line:
                if skip > 0:
                    skip -= 1
                elif skip == 0 and "nruleAss=0" in line:
                    h.write(line.replace(str(nbaseAss), str(nbaseAss + len(hybout) + len(H[0]) - subAss)).replace(orfile[:opos], prefix))
                else:
                    h.write(line)
                for i in range(len(H[0])):
                    nhybAss += 1
                    h.write(f'{prefix}.ruleAss({nhybAss}).id="H{i+1}";\n')
                    h.write(f'{prefix}.ruleAss({nhybAss}).val="{H[0][i]}";\n')
                for par in hybout:
                    nhybAss += 1
                    if Asscount < len(equations):
                        h.write(f'{prefix}.ruleAss({nhybAss}).id="{hybout[Asscount]}";\n')
                        h.write(f'{prefix}.ruleAss({nhybAss}).val="{equations[Asscount]}";\n')
                        Asscount += 1

            elif ".nparameters=" in line:
                h.write(line.replace(str(nbasepar), str(nbasepar + len(orderedweights) + len(H[0]))).replace(orfile[:opos], nfile[:npos]))

            elif ".nruleAss" in line:
                h.write(line.replace(str(nbaseAss), str(nbaseAss + len(hybout) + len(H[0]) - subAss)).replace(orfile[:opos], nfile[:npos]))

            elif ".ruleAss(" in line and ").id" in line:
                if any(item in line for item in hybout):
                    nhybAss -= 1
                    skip = 1
                else:
                    h.write(line)

            elif skip > 0:
                skip -= 1

            else:
                h.write(line.replace(orfile[:opos], nfile[:npos]))

                
def load_model_from_h5(model, file_path):
    with h5py.File(file_path, 'r') as h5file:
        state_dict = {}
        for key in h5file.keys():
            state_dict[key] = torch.tensor(np.array(h5file[key]))
    model.load_state_dict(state_dict)
    return model