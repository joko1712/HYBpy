from customMLP import CustomMLP


#TODO: neuron from projhyb is a matriz like: ["tanh", "relu", "lstm", "tanh", "relu", "lstm"]

def mlpnetcreate(projhyb, neuron):
    ninp = projhyb["mlm"]["nx"]
    nout = projhyb["mlm"]["ny"]
    NH = projhyb['mlm']['options']
    H = len(NH)

    neuron = projhyb['mlm']['layer']
    assert H <= 5, 'more than 5 hidden layers not implemented'

    layer_sizes = [ninp] + NH + [nout]


    if neuron == "1":
        layer_types = ['tanh'] * H
    elif neuron == "2":
        layer_types = ['relu'] * H
    elif neuron == "3":
        layer_types = ['lstm'] * H
    else:
        raise ValueError("Unsupported neuron type")

    ann = CustomMLP(layer_sizes, layer_types)

    return ann

'''
Traceback (most recent call last):
  File "/Users/joko/Documents/GitHub/HYBpy/run_hybtrain_local.py", line 53, in <module>
    projhyb, bestWeights, testing, newHmodFile = hybtrain(
                                                 ~~~~~~~~^
        projhyb, data, user_id, trained_weights, file1_path, temp_dir, run_id
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/joko/Documents/GitHub/HYBpy/hybtrain.py", line 260, in hybtrain
    ann = mlpnetcreate(projhyb, projhyb['mlm']['neuron'])
  File "/Users/joko/Documents/GitHub/HYBpy/mlpnetcreate.py", line 13, in mlpnetcreate
    assert H <= 5, 'more than 5 hidden layers not implemented'
           ^^^^^^
AssertionError: more than 5 hidden layers not implemented
'''