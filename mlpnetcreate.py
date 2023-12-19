from customMLP import CustomMLP


#TODO: neuron from projhyb is a matriz like: ["tanh", "relu", "lstm", "tanh", "relu", "lstm"]

def mlpnetcreate(projhyb, neuron):
    ninp = projhyb["mlm"]["nx"]
    nout = projhyb["mlm"]["ny"]
    NH = projhyb['mlm']['options']
    H = len(NH)

    assert H <= 5, 'more than 5 hidden layers not implemented'

    layer_sizes = [ninp] + NH + [nout]

    print ("layer_sizes:", layer_sizes)

    if neuron == 1:
        layer_types = ['tanh'] * H
    elif neuron == 2:
        layer_types = ['relu'] * H
    elif neuron == 3:
        layer_types = ['lstm'] * H

    ann = CustomMLP(layer_sizes, layer_types)

    return ann
