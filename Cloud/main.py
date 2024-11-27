from __future__ import division
import functions_framework
import os
import json
import tempfile
import shutil
import uuid
import logging
from google.cloud import storage
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import dual_annealing
import scipy.io
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset, DataLoader
from torch import optim
from sklearn.metrics import mean_squared_error, r2_score
import types
import h5py
import torch.nn as nn
import torch.nn.functional as F
import random
import re
from sympy import *
import sympy as sp
from sympy import symbols, sympify, simplify, Matrix, Eq, Symbol
from sympy.parsing.sympy_parser import parse_expr
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


projhyb_cache = {}

def control_function(t, batch):
    u = 0
    

    return u

def default_fobj(w):
    raise NotImplementedError(
        "Objective function fobj is not properly defined.")

def save_model_to_h5(model, file_path):
    state_dict = model.state_dict()
    with h5py.File(file_path, 'w') as h5file:
        for key, value in state_dict.items():
            h5file.create_dataset(key, data=value.cpu().numpy())


def mlpnetinitw(ann):

    w = []

    for layer in ann.layers:
        if isinstance(layer, TanhLayer) or isinstance(layer, ReLULayer):
            layer.w.data = torch.randn_like(
                layer.w) * np.sqrt(2 / (layer.w.size(0) + layer.w.size(1)))
            layer.b.data = torch.zeros_like(layer.b)

            w.extend(layer.w.flatten().detach().numpy())
            w.extend(layer.b.flatten().detach().numpy())

    w = np.array(w).reshape(-1, 1)
    return w, ann

def hybtrain(projhyb, file, user_id, trainedWeights, hmod_content, temp_dir):
    print("USer ID", user_id)
    fobj = default_fobj

    hmod = os.path.join(temp_dir, 'input.hmod')
    with open(hmod, 'w') as f:
        f.write(hmod_content)

    if projhyb is None:
        raise ValueError("at least 1 input required for HYBTRAIN( projhyb)")

    # Check that nensemble is less than or equal to nstep
    if projhyb['nensemble'] > projhyb['nstep']:
        raise ValueError("Too many aggregation parameters; nensemble <= nstep")

    # Check that nensemble is greater than 0
    if projhyb['nensemble'] <= 0:
        raise ValueError("nensemble must be at least 1")

    # Check that nstep is greater than 0
    if projhyb['nstep'] <= 0:
        raise ValueError("nstep must be >= 1")

    projhyb['ntrain'] = 0
    istrainSAVE = np.zeros((file['nbatch'], 1))

    projhyb['itr'] = []
    cnt_jctrain = 0  
    cnt_jcval = 0    
    cnt_jctest = 0 

    if trainedWeights is not None:
        projhyb['nstep'] = 1

    for key, batch_data in file.items():
        if not isinstance(batch_data, dict):
            continue

        # Extract all the iteration keys within the batch
        iteration_keys = [int(k) for k in batch_data.keys() if k.isdigit()]

        if not iteration_keys:
            continue

        last_iteration_key = str(max(iteration_keys))

        cnoise_array = batch_data[last_iteration_key].get("cnoise", [])

        if "train_batches" in projhyb and str(key) in projhyb["train_batches"]:
            cnt_jctrain += len(cnoise_array)
        elif "test_batches" in projhyb and str(key) in projhyb["test_batches"]:
            cnt_jctest += len(cnoise_array)

    if projhyb.get('bootstrap', None) == 1:
        if 'nbootstrap' not in projhyb:
            projhyb['nbootstrap'] = projhyb['ntrain']
            projhyb['nstep'] = projhyb['ntrain']
        else:
            projhyb['nstep'] = projhyb['nbootstrap']

        if 'nbootrate' not in projhyb:
            projhyb['nbootrate'] = 2/3

        nboot = max(1, int(projhyb['ntrain'] * projhyb['nbootrate']))

    print("\nTraining method:")

    mode = projhyb['mode']
    if mode == 1:
        print("   Mode:                   Indirect")

    elif mode == 2:
        print("   Mode:                   Direct")
    elif mode == 3:
        print("   Mode:                   Semidirect")


    jacobianNumber = projhyb.get('jacobian', [])
    # Check Jacobian mode

    jacobian = 'off'
    if jacobianNumber == 0:
        print("   Jacobian:               OFF")
    elif jacobianNumber == 1:
        print("   Jacobian:               ON")
        jacobian = 'on'

    hessianNumber = projhyb.get('hessian', [])
    # Check Hessian mode

    hessian = 'off'
    if hessianNumber == 0:
        print("   Hessian:                OFF")
    elif hessianNumber == 1:
        print("   Hessian:                ON")
        hessian = 'on'

    print(f"   Steps:                  {projhyb['nstep']}")
    print(f"   Displayed iterations:   {projhyb['niter']}")
    print(f"   Total iterations:       {projhyb['niter'] * projhyb['niteroptim']}")

    if projhyb.get("bootstrap", 0) == 1:
        print("   Bootstrap:              ON")
        print(f"   Bootstrap repetitions:    {projhyb['nbootstrap']}")
        print(f"   Bootstrap permutations: {nboot}/{projhyb['ntrain']}")
    else:
        print("   Bootstrap:              OFF")

    # TODO: CONTROL FUNCTION selection from Control_functions folder
    if projhyb["fun_control"] != 0:
        print("   Control function:       ON")
        t = 0
        batch = 0
        fun_control = control_function(t, batch)
    else:
        print("   Control function:       OFF")
        print("   ASK USER TO DEFINE CONTROL FUNCTION")

    if projhyb["crossval"] == 1:
        print("   Cross-validation:       ON")
    else:
        print("   Cross-validation:       OFF")

#######################################################################################################################
    options = {}
    species_bounds = []
    for key, species in projhyb['species'].items():
        species_min = species['min']
        species_max = species['max']
        species_bounds.append((species_min, species_max))
    NH = projhyb['mlm']['options']
    H = len(NH)
    projhyb["mlm"]['h'] = H
    projhyb["mlm"]['nl'] = 2 + H
    projhyb["mlm"]['nh'] = NH[:H]
    projhyb["mlm"]["ninp"] = projhyb["mlm"]["nx"]
    projhyb["mlm"]["nout"] = projhyb["mlm"]["ny"]
    projhyb["mlm"]['nw'] = (projhyb["mlm"]['nx'] + 1) * projhyb["mlm"]['nh'][0]

    if projhyb['method'] == 1:
        print("   Optimiser:              Trust Region Reflective")
        options = {
            'xtol': 1e-10, #1e-10
            'gtol': 1e-10,
            'ftol': 1e-10,
            'verbose': projhyb['display'],
            'max_nfev': projhyb['niter'],
            'method': 'trf',
        }

    elif projhyb['method'] == 2:
        algorithm = 'L-BFGS-B' if projhyb['jacobian'] != 1 else 'trust-constr'
        print(f"   Optimiser:              {algorithm}")
        options = { 
            'method': algorithm,
            'options': {
                'gtol': 1e-10,
                'xtol': 1e-10,
                'barrier_tol': 1e-10,
                'verbose': projhyb['display'],
                'maxiter': projhyb['niter']
            } 
        }
    elif projhyb['method'] == 3:
        print("   Optimiser:              Simulated Annealing")
        bounds = species_bounds * projhyb['mlm']['nw']
        options = {
            'maxiter': 100 * projhyb['niter'] * projhyb['niteroptim'],
            'verbose': projhyb['display']
        }

    elif projhyb['method'] == 4:
        print("   Optimiser:              Adam")
        num_epochs = projhyb['niter']
        lr = 0.001  
#######################################################################################################################

    print("\n\n")

    for i in range(1, H):
        projhyb["mlm"]['nw'] += (projhyb["mlm"]['nh']
                                 [i - 1] + 1) * projhyb["mlm"]['nh'][i]
    projhyb["mlm"]['nw'] += (projhyb["mlm"]['nh']
                             [H - 1] + 1) * projhyb["mlm"]['ny']

    print("Number of weights: ", projhyb["mlm"]['nw'])
    print("Number of inputs: ", projhyb["mlm"]['nx'])
    print("Number of outputs: ", projhyb["mlm"]['ny'])
    print("Number of hidden layers: ", projhyb["mlm"]['h'])
    print("Number of neurons in each hidden layer: ", projhyb["mlm"]['nh'])

    TrainRes = {
        'istep': 0,
        't0': time.time()
    }

    projhyb["mlm"]["DFDS"] = None
    projhyb["mlm"]["DFDRANN"] = None
    projhyb["mlm"]["DANNINPDSTATE"] = None
    projhyb["mlm"]["ANNINP"] = None
    projhyb['mlm']["FSTATE"] = None
    projhyb['mlm']['STATE_SYMBOLS'] = None
    projhyb['mlm']['NVALUES'] = None

    if 'fundata' not in projhyb['mlm'] or projhyb['initweights'] == 1:
        print('Weights initialization...1')
        ann = mlpnetcreate(projhyb, projhyb['mlm']['neuron'])
        projhyb['mlm']['fundata'] = ann
        weights, ann = ann.get_weights()
        
        ann.set_weights(weights)
        


    elif projhyb['initweights'] == 2:
        print('Read weights from file...')
        weights_data = load(projhyb['weightsfile'])
        weights = np.reshape(weights_data['wPHB0'], (-1, 1))
        projhyb['mlm']['fundata'].set_weights(weights)

    weights = weights.ravel()
    
    istep = 1

    if projhyb['mode'] == 1:

        evaluator = IndirectFunctionEvaluator(ann, projhyb, file, resfun_indirect_jac)

    elif projhyb['mode'] == 2:
            
        evaluator = IndirectFunctionEvaluator(ann, projhyb, file, resfun_direct_jac)
    
    elif projhyb['mode'] == 3:
            
        evaluator = IndirectFunctionEvaluator(ann, projhyb, file, resfun_semidirect_jac)

#######################################################################################################################
    
    bestWeights = None
    bestPerformance = float('inf')

    for istep in range(1, projhyb['nstep']+1):
        print("TESTING")

        for i in range(1, file['nbatch'] + 1):
            istrain = file[i]["istrain"]
            projhyb['istrain'] = [0] * file['nbatch']
            projhyb['istrain'][i - 1] = istrain
            print(projhyb['istrain'])

        if projhyb['bootstrap'] == 1:
            ind = sorted(np.random.permutation(projhyb['ntrain'])[:nboot])
            projhyb['istrain'][projhyb['itr']] = 0
            for idx in ind:
                projhyb['istrain'][projhyb['itr'][idx]] = 1
        
        if istep > 1:
            print('Weights initialization...2')
            ann = mlpnetcreate(projhyb, projhyb['mlm']['neuron'])
            projhyb['mlm']['fundata'] = ann
            weights, ann = ann.get_weights()
            ann.set_weights(weights)

            with open("filetest.json", "w") as f:
                json.dump(convert_numpy(weights), f)
                json.dump("weights", f)

        print(
            'ITER  RESNORM    [C]train   [C]valid   [C]test   [R]train   [R]valid   [R]test    AICc       NW   CPU')

    
        if projhyb['jacobian'] == 1:
            options['jac'] = evaluator.jac_func

        if projhyb['hessian'] == 1:
            options['hess'] = evaluator.hess_func


        if trainedWeights == None:
    
            if projhyb["method"] == 1:  # LEVENBERG-MARQUARDT
                result = least_squares(evaluator.fobj_func, x0=weights, **options)
                print("result", result.x)
                optimized_weights = result.x


            elif projhyb["method"] == 2:  # QUASI-NEWTON

                if options.get('method', None) == 'trust-constr':
                    result = minimize(evaluator.fobj_func, x0=weights, hess=None, **options)
                else:
                    result = minimize(evaluator.fobj_func, x0=weights, **options)
                
                optimized_weights = result.x
                print("result", result.x)


            elif projhyb["method"] == 3:  # Dual ANNEALING
                result = dual_annealing(evaluator.fobj_func, bounds=bounds, **options)
                optimized_weights = result.x

            elif projhyb["method"] == 4:  # ADAM

                manual_optimizer = ManualAdamOptimizer(ann, lr=lr)
                fobj_history = []

                for epoch in range(num_epochs):
                    print(f"Epoch {epoch + 1}/{num_epochs}")

                    weights, ann = ann.get_weights()
                    fobjs, gradient = evaluator.evaluate_adam(weights)
                    print("weights", weights)

                    print(f"Objective function value at start of epoch: {np.linalg.norm(fobjs)}")
                    print(f"Norm of gradient at start of epoch: {np.linalg.norm(gradient)}")

                    manual_optimizer.zero_grad()

                    manual_optimizer.step(fobjs, gradient)

                    updated_weights, ann = ann.get_weights()
                    print("updated_weights", updated_weights)
                    ann.set_weights(updated_weights)

                    updated_fobjs, updated_gradient = evaluator.evaluate_adam(updated_weights)

                    manual_optimizer.step(updated_fobjs, updated_gradient)

                    fobj_history.append(np.linalg.norm(updated_fobjs))
                    print(f"Objective function value after epoch {epoch + 1}: {np.linalg.norm(updated_fobjs)}")
                    print(f"Norm of gradient after epoch {epoch + 1}: {np.linalg.norm(updated_gradient)}")

                optimized_weights, _ = ann.get_weights()
        
        else:


            trainedWeights = np.array(trainedWeights)

            ann.set_weights(trainedWeights)
            
            optimized_weights, _ = ann.get_weights()


        fobj_value = evaluator.fobj_func(optimized_weights)
        fobj_norm = np.linalg.norm(fobj_value)


        if bestPerformance == None:
            bestPerformance = fobj_norm
            bestWeights = optimized_weights

        if bestPerformance > fobj_norm:
            bestPerformance = fobj_norm
            bestWeights = optimized_weights

    model_path = os.path.join(temp_dir, "trained_model.h5")
    save_model_to_h5(ann, model_path)

    newHmodFile = os.path.join(temp_dir, "Newhmod.hmod")
    saveNN(model_path, projhyb["inputs"], projhyb["outputs"], hmod, newHmodFile, bestWeights, ann)

    testing = teststate(ann, user_id, projhyb, file, bestWeights, temp_dir, projhyb['method'])

    plot_optimization_results(evaluator.fobj_history, evaluator.jac_norm_history)    


    return projhyb, bestWeights, testing, newHmodFile


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy(value) for value in obj)
    return obj
        

####
#   INDIRECT
####

def resfun_indirect_jac(ann, w, istrain, projhyb, file, method=1):
    print("weights", w)

# LOAD THE WEIGHTS into the ann
    ann.set_weights(w)
    ann.print_weights_and_biases()
    if not istrain:
        istrain = projhyb["istrain"]

    # ires = 11 
    ns = projhyb["nspecies"]
    print("ns", ns)
    nt = ns + projhyb["ncompartment"]
    nw = projhyb["mlm"]["nw"]
    isres = []
    isresY = []
    for i in range(1, ns + 1):
        if projhyb["species"][str(i)]["isres"] == 1: 
            isres = isres + [i]
            isresY = isresY + [i - 1]

    isres = isres
        
    nres = len(isres)
    
    npall = sum(file[i+1]["np"] for i in range(file["nbatch"]) if file[i+1]["istrain"] == 1)

    sresall = np.zeros(npall * nres)

    sjacall = np.zeros((npall * nres, nw))

    print("sresall", npall*nres)
    print("npall", nres)

    print("nn:", ann)

    COUNT = 0
    for l in range(file["nbatch"]):
        l = l + 1
        if file[l]["istrain"] == 1:
            tb = file[l]["time"]

            Y = file[l]["y"]
            Y = np.array(Y)
            Y = Y.astype(np.float64)
            Y = torch.from_numpy(Y)
            
            batch = str(l)

            sY = file[l]["sy"]
            sY = np.array(sY)
            sY = sY.astype(np.float64)
            sY = torch.from_numpy(sY)
            
            state = np.array(file[l]["y"][0])
            
            state = state[0:ns+1]

            Sw = np.zeros((nt, nw))

            for i in range(1, file[l]["np"]):
                
                statedict = file[l]["key_value_array"][i-1]

                dict_items_list = list(statedict.items())
                statedict =  dict(dict_items_list[ns:len(dict_items_list)])

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann,odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)

                
                
                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)
                state_adjusted = state_tensor[0:nres]
                Ystate = Y_select - state_adjusted.numpy()
                
                sresall[COUNT:COUNT + nres] = Ystate / sY[i, isresY].numpy()

                SYrepeat = sY[i, isresY].reshape(-1, 1).repeat(1, nw).numpy()
                result = (- Sw[isresY, :].detach().numpy()) / SYrepeat
                sjacall[COUNT:COUNT + nres, 0:nw] = result
                COUNT += nres
               
               


    valid_idx = ~np.isnan(sresall) & ~np.isinf(sresall)
    sresall = sresall[valid_idx]
    sjacall = sjacall[valid_idx, :]

    if method == 1 or method == 4:

        fobj = sresall
        jac = sjacall


        '''
        epsilon = 1e-8  
        jac_max_abs = np.max(np.abs(sjacall), axis=1, keepdims=True)
        jac_max_abs_safe = np.where(jac_max_abs > epsilon, jac_max_abs, epsilon)
        jac_normalized = sjacall / jac_max_abs_safe

        fobj_max_abs = np.max(np.abs(fobj))
        fobj_max_abs_safe = max(fobj_max_abs, epsilon)
        fobj_normalized = fobj / fobj_max_abs_safe

        print("jac", jac_normalized)

        print("fobj", fobj_normalized)
        '''
        
    else:
        fobj = np.dot(sresall.T, sresall) / len(sresall)
        jac = np.sum(2 * np.repeat(sresall.reshape(-1, 1), nw,
                     axis=1) * sjacall, axis=0) / len(sresall)
                     
    return fobj, jac

####
#   SEMIDIRECT
####

def resfun_semidirect_jac(ann, w, istrain, projhyb, file, method=1):
    print("weights", w)

    ann.set_weights(w)
    ann.print_weights_and_biases()
    if not istrain:
        istrain = projhyb["istrain"]

    ns = projhyb["nspecies"]
    nt = ns + projhyb["ncompartment"]
    nw = projhyb["mlm"]["nw"]
    isres = []
    isresY = []
    for i in range(1, ns + 1):
        if projhyb["species"][str(i)]["isres"] == 1:
            isres = isres + [i]
            isresY = isresY + [i - 1]

    isres = isres

    nres = len(isres)

    npall = sum(file[i+1]["np"] for i in range(file["nbatch"]) if file[i+1]["istrain"] == 1)

    sresall = np.zeros(npall * nres)

    sjacall = np.zeros((npall * nres, nw))

    CONT = 0

    for l in range(file["nbatch"]):
        l = l + 1
        if file[l]["istrain"] == 1:
            tb = file[l]["time"]
            Y = file[l]["y"]
            Y = np.array(Y)
            Y = Y.astype(np.float64)
            Y = torch.from_numpy(Y)

            batch = str(l)

            sY = file[l]["sy"]
            sY = np.array(sY)
            sY = sY.astype(np.float64)
            sY = torch.from_numpy(sY)

            state = np.array(file[l]["y"][0])
            Sw = np.zeros((nt, nw))
            jac = np.zeros((nt, projhyb["mlm"]["ny"]))

            for i in range(1, file[l]["np"]):
                statedict = np.array(file[l]["key_value_array"][i-1])
                print("statedict", statedict)

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann, odesfun,
                                            control_function, projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)

                ucontrol = control_function(tb[i], batch_data)
                inp = projhyb["mlm"]["xfun"](tb[i], state, ucontrol)
                _, _, DrannDw = projhyb["mlm"]["yfun"](inp, w, projhyb["mlm"]["fundata"])

                Sw = np.dot(jac, DrannDw)
                resall[COUNT:COUNT + nres] = (Y[i, isres] - state[isres]) / sY[i, isres]
                jacall[COUNT:COUNT+nres, :nw] = -Sw[isres, :] / np.tile(sY[i, isres], (nw, 1)).T
                COUNT += nres
    
    valid_idx = ~np.isnan(sresall) & ~np.isinf(sresall)
    sresall = sresall[valid_idx]
    sjacall = sjacall[valid_idx, :]

    if method == 1 or method == 4:
        fobj = sresall
        jac = sjacall
    else:
        fobj = np.dot(sresall, sresall) / len(sresall)
        jac = np.sum(2 * np.repeat(sresall.reshape(-1, 1), nw,
                     axis=1) * sjacall, axis=0) / len(sresall)


####
#   DIRECT
####

def resfun_direct_jac(ann, w, istrain, projhyb, file, method=1):
    print("weights", w)

    # LOAD THE WEIGHTS into the ANN
    ann.set_weights(w)
    ann.print_weights_and_biases()
    if not istrain:
        istrain = projhyb["istrain"]

    ns = projhyb["nspecies"]
    nt = ns + projhyb["ncompartment"]
    nw = projhyb["mlm"]["nw"]
    isres = []
    isresY = []
    for i in range(1, ns + 1):
        if projhyb["species"][str(i)]["isres"] == 1:
            isres = isres + [i]
            isresY = isresY + [i - 1]

    isres = isres

    nres = len(isres)

    npall = sum(file[i+1]["np"] for i in range(file["nbatch"]) if file[i+1]["istrain"] == 1)

    sresall = np.zeros(npall * nres)
    sjacall = np.zeros((npall * nres, nw))

    COUNT = 0
    for l in range(file["nbatch"]):
        l = l + 1
        if file[l]["istrain"] == 1:
            tb = file[l]["time"]
            Y = file[l]["y"]
            Y = np.array(Y)
            Y = Y.astype(np.float64)
            Y = torch.from_numpy(Y)

            batch = str(l)

            sY = file[l]["sy"]
            sY = np.array(sY)
            sY = sY.astype(np.float64)
            sY = torch.from_numpy(sY)

            state = np.array(file[l]["y"][0])
            state = torch.tensor(state, requires_grad=True, dtype=torch.float64)
            statedict = np.array(file[l]["key_value_array"][0])

            for i in range(1, file[l]["np"]):
                
                print("statedict", statedict)

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann, odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)

                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)
                state_adjusted = state_tensor[0:nres]
                Ystate = Y_select - state_adjusted

                sresall[COUNT:COUNT + nres] = Ystate.detach().numpy() / sY[i, isresY].numpy()

                Ystate.sum().backward()
                sjacall[COUNT:COUNT + nres, 0:nw] = state.grad[isresY, :].detach().numpy() / sY[i, isresY].numpy()
                state.grad.zero_()

                COUNT += nres
                print("#################################################")
                print("------------------LOOP", i, "------------------")
                print("#################################################")
                print("state", state)

    valid_idx = ~np.isnan(sresall) & ~np.isinf(sresall)
    sresall = sresall[valid_idx]
    sjacall = sjacall[valid_idx, :]

    if method == 1 or method == 4:
        fobj = sresall
        jac = sjacall
    else:
        fobj = np.dot(sresall.T, sresall) / len(sresall)
        jac = np.sum(2 * np.repeat(sresall.reshape(-1, 1), nw,
                     axis=1) * sjacall, axis=0) / len(sresall)

    return fobj, jac

class ManualAdamOptimizer:
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
        self.model = model
    
    def step(self, fobj, gradient):
        start = 0
        for param in self.model.parameters():
            num_elements = param.numel()

            grad_slice = gradient[start:start + num_elements]

            manual_grad = torch.from_numpy(grad_slice).view_as(param).to(param.device)

            param.grad = manual_grad

            start += num_elements

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()



class IndirectFunctionEvaluator:
    def __init__(self, ann, projhyb, file, evaluation_function):
        self.ann = ann
        self.projhyb = projhyb
        self.file = file
        self.evaluation_function = evaluation_function
        self.last_weights = None
        self.last_fobj = None
        self.last_jac = None
        self.fobj_history = []
        self.jac_norm_history = []

    def evaluate(self, weights):
        if np.array_equal(weights, self.last_weights):
            return self.last_fobj, self.last_jac
        else:
            try:
                self.last_fobj, self.last_jac = self.evaluation_function(self.ann, weights, self.projhyb['istrain'], self.projhyb, self.file, self.projhyb['method'])
                self.last_weights = weights
                self.fobj_history.append(self.last_fobj)
                self.jac_norm_history.append(np.linalg.norm(self.last_jac))
            except Exception as e:
                print(f"Error evaluating function: {e}")
                raise
            return self.last_fobj, self.last_jac

    def evaluate_adam(self, weights):
        if np.array_equal(weights, self.last_weights):
            return self.last_fobj, self.last_gradient
        else:
            try:
                residuals, jacobian = self.evaluation_function(
                    self.ann, weights, self.projhyb['istrain'],
                    self.projhyb, self.file, self.projhyb['method']
                )
                self.last_weights = weights
                self.last_fobj = residuals
                self.last_jac = jacobian

                gradient = jacobian.T @ residuals
                self.last_gradient = gradient

                self.fobj_history.append(np.linalg.norm(self.last_fobj))
                self.jac_norm_history.append(np.linalg.norm(gradient))
            except Exception as e:
                print(f"Error evaluating function: {e}")
                raise
            return self.last_fobj, self.last_gradient

    def fobj_func(self, weights):
        fobj, _ = self.evaluate(weights)
        return fobj

    def jac_func(self, weights):
        _, jac = self.evaluate(weights)
        return jac

    def torch_fobj_func(self, weights_tensor):
        weights_numpy = weights_tensor.detach().cpu().numpy()
        fobj, _ = self.evaluate(weights_numpy)

        fobj_tensor = torch.tensor(fobj, dtype=torch.float32)
        fobj_tensor.requires_grad_(True)

        if fobj_tensor.ndim > 0:
            fobj_tensor = fobj_tensor.sum()

        return fobj_tensor

    def torch_grad_func(self, weights_tensor):
        weights_numpy = weights_tensor.detach().cpu().numpy()
        _, jac = self.evaluate(weights_numpy)
        return torch.tensor(jac, dtype=torch.float32, requires_grad=True)

### MOVE THIS OUT

def plot_optimization_results(fobj_values, jacobian_matrix):
    fobj_norms = [np.linalg.norm(val) for val in fobj_values]
    fobj_norms_non_zero = [norm for norm in fobj_norms if norm > 0]
    
    gradient_norms = [np.linalg.norm(jac) for jac in jacobian_matrix]

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.semilogy(fobj_norms_non_zero, '-o', label='Objective Function Norm', markersize=4)
    plt.xlabel('Iteration (non-zero only)')
    plt.ylabel('Objective Function Norm (log scale)')
    plt.title('Objective Function Norm Over Iterations')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.semilogy(gradient_norms, '-x', label='Jacobian Norm', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Norm of Jacobian (log scale)')
    plt.title('Norm of Jacobian Over Iterations')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

def teststate(ann, user_id, projhyb, file, w, temp_dir, method=1):
    dictState = {}
    w = np.array(w)

    # Load the weights into the ann
    ann.set_weights(w)
    ann.print_weights_and_biases()

    ns = projhyb["nspecies"]
    nt = ns + projhyb["ncompartment"]
    nw = projhyb["mlm"]["nw"]
    
    isres = [i for i in range(1, ns + 1) if projhyb["species"][str(i)]["isres"] == 1]
    isresY = [i - 1 for i in isres]
    
    nres = len(isres)

    npall = sum(file[i+1]["np"] for i in range(file["nbatch"]) if isinstance(file[i+1], dict) and file[i+1]["istrain"] == 1)

    sresall = np.zeros(npall * nres)
    sjacall = np.zeros((npall * nres, nw))

    COUNT = 0
    for l in range(file["nbatch"]):
        l = l + 1
        if not isinstance(file[l], dict):
            continue
        tb = file[l]["time"]
        print("tb", tb)
        Y = np.array(file[l]["y"]).astype(np.float64)
        Y = torch.from_numpy(Y)
        
        batch = str(l)

        sY = np.array(file[l]["sy"]).astype(np.float64)
        sY = torch.from_numpy(sY)
        
        state = np.array(file[l]["y"][0])

        state = state[0:ns+1]
        Sw = np.zeros((nt, nw))


        for i in range(1, file[l]["np"]):
            statedict = file[l]["key_value_array"][i-1]

            dict_items_list = list(statedict.items())
            statedict =  dict(dict_items_list[ns:len(dict_items_list)])

            batch_data = file[l]
            _, state, Sw, hess = hybodesolver(ann, odesfun, control_function, projhyb["fun_event"], tb[i-1], tb[i], state, statedict, None, None, w, batch_data, projhyb)

            if l not in dictState:
                dictState[l] = {}
                dictState[l][0] = file[l]["y"][0]
            dictState[l][i] = state

    overall_metrics = {
        'mse_train': [],
        'mse_test': [],
        'r2_train': [],
        'r2_test': []
    }
    print("ranfe", projhyb['mlm']['nx'])
    print("Species", projhyb['species'])
    for i in range(projhyb["nspecies"]):
        actual_train = []
        actual_test = []
        predicted_train = []
        predicted_test = []
        err = []

        train_batches = [batch for batch, data in file.items() if isinstance(data, dict) and data["istrain"] == 1]
        test_batches = [batch for batch, data in file.items() if isinstance(data, dict) and data["istrain"] == 3]

        for batch in train_batches:
            for t in range(1, file[batch]["np"]):
                actual_train.append(np.array(file[batch]["y"][t-1][i]))
                if batch in dictState and t in dictState[batch]:
                    predicted_train.append(dictState[batch][t-1][i])
                else:
                    print(f"Missing prediction for train batch {batch}, time {t}")

        for batch in test_batches:
            for t in range(1, file[batch]["np"]):
                actual_test.append(np.array(file[batch]["y"][t-1][i]))
                if batch in dictState and t in dictState[batch]:
                    predicted_test.append(dictState[batch][t-1][i])
                    err.append(file[batch]["sy"][t-1][i])
                else:
                    print(f"Missing prediction for test batch {batch}, time {t}")

        actual_train = np.array(actual_train, dtype=np.float64)
        actual_test = np.array(actual_test, dtype=np.float64)
        predicted_train = np.array(predicted_train, dtype=np.float64)
        predicted_test = np.array(predicted_test, dtype=np.float64)
        err = np.array(err, dtype=np.float64)
    
        print("actual_train", actual_train)
        print("actual_test", actual_test)
        print("predicted_train", predicted_train)
        print("predicted_test", predicted_test)
        print("err", err)        

        if actual_train.shape != predicted_train.shape:
            print(f"Shape mismatch for training data: actual_train {actual_train.shape}, predicted_train {predicted_train.shape}")
            continue

        if actual_test.shape != predicted_test.shape:
            print(f"Shape mismatch for test data: actual_test {actual_test.shape}, predicted_test {predicted_test.shape}")
            continue

        mse_train = mean_squared_error(actual_train, predicted_train)
        mse_test = mean_squared_error(actual_test, predicted_test)
        r2_train = r2_score(actual_train, predicted_train)
        r2_test = r2_score(actual_test, predicted_test)

        overall_metrics['mse_train'].append(mse_train)
        overall_metrics['mse_test'].append(mse_test)
        overall_metrics['r2_train'].append(r2_train)
        overall_metrics['r2_test'].append(r2_test)

        print(f'Training MSE: {mse_train}')
        print(f'Test MSE: {mse_test}')
        print(f'Training R²: {r2_train}')
        print(f'Test R²: {r2_test}')

        mse_train = abs(mse_train)
        mse_test = abs(mse_test)
        r2_test = abs(r2_test)
        r2_train = abs(r2_train)

        textstr = '\n'.join((
            f'Training MSE: {mse_train:.4f}',
            f'Training R²: {r2_train:.4f}',
            f'Test MSE: {mse_test:.4f}',
            f'Test R²: {r2_test:.4f}',
        )) 
        
        z_score = 2.05
        margin = z_score * err

        lower_bound = predicted_test - margin
        upper_bound = predicted_test + margin
        
        for value in range(len(lower_bound)):
            if lower_bound[value] < 0:
                lower_bound[value] = 0

        x = file[train_batches[0]]["time"][:-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x, actual_test[:len(x)], err[:len(x)], fmt='o', linewidth=1, capsize=6, label="Observed data", color='green', alpha=0.5)
        ax.plot(x, predicted_test[:len(x)], label="Predicted", color='red', linewidth=1)
        #ax.fill_between(x, lower_bound[:len(x)], upper_bound[:len(x)], color='gray', label="Confidence Interval", alpha=0.5)

        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.title(f"Species {projhyb['species'][str(i+1)]['id']} ", verticalalignment='bottom', fontsize=16, fontweight='bold')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        plt.legend()
        temp = os.path.join(temp_dir, 'plots')
        user_dir = os.path.join(temp, user_id)
        date_dir = os.path.join(user_dir, time.strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)
        
        time_series_plot_filename = os.path.join(date_dir, f'metabolite_{projhyb["species"][str(i+1)]["id"]}_{uuid.uuid4().hex}.png')
        plt.savefig(time_series_plot_filename, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predicted_train, actual_train, color='blue', label='Train', alpha=0.5)
        ax.scatter(predicted_test, actual_test, color='red', label='Test', alpha=0.5)
        ax.plot([0, max(actual_test)], [0, max(actual_test)], 'r--', color='gray', label='Ideal', alpha=0.25)
        plt.xlabel('Observed')
        plt.ylabel('Predicted')
        plt.title(f"Predicted vs Observed for Metabolite {projhyb['species'][str(i+1)]['id']} ", verticalalignment='bottom', fontsize=16, fontweight='bold')
        plt.legend()
        
        predicted_vs_actual_plot_filename = os.path.join(date_dir, f'predicted_vs_observed_{projhyb["species"][str(i+1)]["id"]}_{uuid.uuid4().hex}.png')
        plt.savefig(predicted_vs_actual_plot_filename, dpi=300)
        plt.close(fig)

    overall_mse_train = np.mean(overall_metrics['mse_train'])
    overall_mse_test = np.mean(overall_metrics['mse_test'])
    overall_r2_train = np.mean(overall_metrics['r2_train'])
    overall_r2_test = np.mean(overall_metrics['r2_test'])

    return {'mse_train': overall_mse_train, 'mse_test': overall_mse_test, 'r2_train': overall_r2_train, 'r2_test': overall_r2_test}

class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, layer_types):
        super(CustomMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_types):
            if layer_type == 'tanh':
                self.layers.append(
                    TanhLayer(layer_sizes[i], layer_sizes[i + 1]))
            elif layer_type == 'relu':
                self.layers.append(
                    ReLULayer(layer_sizes[i], layer_sizes[i + 1]))
            elif layer_type == 'lstm':
                self.layers.append(
                    LSTMLayer(layer_sizes[i], layer_sizes[i + 1]))

        self.layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))


        self.scale_weights(scaling_factor=0.001)


    def forward(self, x):
        for layer in self.layers:
            x = x.to(dtype=torch.float64)

            x = layer(x)
        return x


    def scale_weights(self, scaling_factor):
        with torch.no_grad():  
            for layer in self.layers:
                for w in layer.w.data:
                    w *= scaling_factor * random.uniform(0.9, 1.1)
                layer.b.data *= scaling_factor

    
    def reinitialize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'w'):
                if isinstance(layer, (TanhLayer, Linear)): 
                    nn.init.xavier_uniform_(layer.w)
                elif isinstance(layer, ReLULayer): 
                    nn.init.kaiming_uniform_(layer.w, mode='fan_in', nonlinearity='relu')

            if hasattr(layer, 'b'):
                nn.init.constant_(layer.b, 0)

        self.scale_weights(scaling_factor=0.001)

        weights = []
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weights.append(layer.w.data.cpu().numpy().flatten())
                weights.append(layer.b.data.cpu().numpy().flatten())

        weights = np.concatenate(weights)
        return weights, self    


    def set_weights(self, new_weights):
        start = 0
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weight_num_elements = torch.numel(layer.w)
                bias_num_elements = torch.numel(layer.b)

                new_w = new_weights[start:start+weight_num_elements]
                new_b = new_weights[start+weight_num_elements:start+weight_num_elements+bias_num_elements]

                start += weight_num_elements + bias_num_elements

                new_w_tensor = torch.from_numpy(new_w).view_as(layer.w).type_as(layer.w)
                new_b_tensor = torch.from_numpy(new_b).view_as(layer.b).type_as(layer.b)

                layer.w.data = new_w_tensor
                layer.b.data = new_b_tensor


    def print_weights_and_biases(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                print(f"Layer {i} weights (w): \n{layer.w.data}")
                print(f"Layer {i} biases (b): \n{layer.b.data}\n")


    def get_weights(self):
        weights = []
        for layer in self.layers:
            w = layer.w.data.cpu().numpy().flatten()
            b = layer.b.data.cpu().numpy().flatten()
            weights.extend(w)
            weights.extend(b)
        return np.array(weights), self

    def backpropagate(self, x, ny):
        activations = [x]

        for layer in self.layers:

            x = x.to(dtype=torch.float64)
            x = layer(x)

            activations.append(x)
            
        # y = output
        y = activations[-1]

        # x = input
        x = activations[0]

        # h1 
        tensorList = []
        DrannDw = []
        output_size = self.layers[-1].w.shape[0]
        DrannDanninp = torch.eye(output_size, dtype=torch.float64)
        A1 = DrannDanninp
        tensor_size = 0
    
        for i in reversed(range(len(self.layers))):

            h1 = activations[i]
            h1l = self.layers[i-1].derivative(h1)
            h1l_reshaped = h1l.t()        

            
            h1_reshaped = torch.cat((h1.t(), torch.tensor([[1]])), dim=1)
            
            layer_dydw = torch.kron(h1_reshaped,A1)

            tensor_size = tensor_size + layer_dydw.shape[1] 
            tensorList.insert(0, layer_dydw)


            if i == 0:
                break

            A1 = -(torch.mm(DrannDanninp,self.layers[i].w) * h1l_reshaped.repeat(output_size, 1))


            DrannDanninp = A1

            h1l_reshaped = torch.cat((h1l_reshaped, torch.tensor([[1]])), dim=1)


        DrannDanninp = torch.mm(A1,self.layers[0].w)


        DrannDw = tensorList

        DrannDw = torch.cat(DrannDw, dim=1)
        DrannDw = DrannDw.view(ny, tensor_size)


        return y, DrannDanninp, DrannDw

class TanhLayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(TanhLayer, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return torch.tanh(torch.mm(self.w, x) + self.b)

    def derivative(self, x):
        return (x ** 2) -1 


class ReLULayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(ReLULayer, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.kaiming_uniform(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        xin = torch.mm(self.w, x) + self.b
        return F.relu(xin)

    def derivative(self, x):
        return (x > 0).double()


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        
        self.hidden = None
        self.cell = None

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))

        return output

    def reset_state(self):
        self.hidden = None
        self.cell = None


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b) 

    def forward(self, x):
        return torch.mm(self.w, x) + self.b

    def derivative(self, x):
        return torch.ones_like(x) 


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

    ann = CustomMLP(layer_sizes, layer_types)

    return ann

def mlpnetsetw(custom_mlp, w):
    count = 0
    for layer in custom_mlp.layers:
        input_size, output_size = layer.w.shape[1], layer.w.shape[0]
        num_weights = input_size * output_size
        num_biases = output_size

        layer_weights = w[count:count +
                          num_weights].reshape(output_size, input_size)
        layer.w.data = torch.tensor(layer_weights, dtype=layer.w.data.dtype)
        count += num_weights

        layer_biases = w[count:count + num_biases].reshape(-1, 1)
        layer.b.data = torch.tensor(layer_biases, dtype=layer.b.data.dtype)
        count += num_biases

    return custom_mlp



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
                h.write(line.replace("Newhmod", prefix))

                
def load_model_from_h5(model, file_path):
    with h5py.File(file_path, 'r') as h5file:
        state_dict = {}
        for key in h5file.keys():
            state_dict[key] = torch.tensor(np.array(h5file[key]))
    model.load_state_dict(state_dict)
    return model



def hybodesolver(ann, odesfun, controlfun, eventfun, t0, tf, state, statedict, jac, hess, w, batch, projhyb):

    print("state start of hybodesolver", state)
    t = t0
    hopt = []   

    state_symbols = []

    anninp, rann, anninp_mat = anninp_rann_func(projhyb, state)
    print("anninp", anninp)
    print("len(anninp)", len(anninp))
    print("statedict", statedict)
    anninp_mat = [expr.subs(statedict) for expr in anninp_mat]
    print("anninp_mat", anninp_mat)

    anninp_tensor = torch.tensor(anninp_mat, dtype=torch.float64)
    anninp_tensor = anninp_tensor.view(-1, 1)       

    activations = [anninp_tensor]
    y = activations[-1]

    rann_results = ann.forward(y)

    rann_results = rann_results.detach().numpy()
    
    state = extract_species_values(projhyb, state)
    values = {}

    for range_y in range(0, len(rann_results)):
        values["rann"+str(range_y+1)] = rann_results[range_y].item()

    for i in range(1, projhyb["ncompartment"]+1):
        values[str(projhyb["compartment"][str(i)]["id"])] = int(projhyb["compartment"][str(i)]["val"])


    for i in range(1, projhyb["mlm"]["ny"]+1):
        values[projhyb["mlm"]["y"][str(i)]["id"]] = values[projhyb["mlm"]["y"][str(i)]["val"]]
    
    for i in range(1, projhyb["nparameters"]+1):
        values[projhyb["parameters"][str(i)]["id"]] = projhyb["parameters"][str(i)]["val"]
    
    for i in range(1, projhyb["nspecies"]+1):
        species_id = projhyb["species"][str(i)]["id"]
        if species_id.startswith("chybrid"):
            continue
        state_symbols.append(sp.Symbol(species_id))

    for i in range(1, projhyb["ncompartment"]+1):
        compartment_id = projhyb["compartment"][str(i)]["id"]
        if compartment_id.startswith("chybrid"):
            continue
        state_symbols.append(sp.Symbol(projhyb["compartment"][str(i)]["id"]))
    
    print("state_symbols", state_symbols)
    print("len(state_symbols)", len(state_symbols)) 
    
    for key, value in statedict.items():
        values[key] = value
    
    if jac is not None:
        jac = torch.tensor(jac, dtype=torch.float64)
    fstate = fstate_func(projhyb, values)

    while t < tf:
        h = min(projhyb['time']['TAU'], tf - t)
        batch['h'] = h

        if eventfun and callable(eventfun):
            if jac is not 0:
                batch, state, dstatedstate = eventfun(t, batch, state)
                jac = dstatedstate * jac
            else:
                batch, state = eventfun(t, batch, state)

        if controlfun and callable(controlfun):
            ucontrol1 = controlfun(t, batch)
            
        else:
            ucontrol1 = []



                
        if jac != None:
            k1_state, k1_jac = odesfun(ann, t, state, jac, None, w, ucontrol1, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
        else:
            k1_state = odesfun(ann, t, state, None, None, w, ucontrol1, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
        
        control = None
        ucontrol2 = controlfun(t + h / 2, batch) if controlfun is not None else []

        h2 = h / 2
        h2 = torch.tensor(h2, dtype=torch.float64)
        k1_state = np.array(k1_state)
        k1_state = k1_state.astype(np.float64)
        k1_state = torch.from_numpy(k1_state)
        
        if jac != None:

            state1, staterann1 = update_state(state, h2, k1_state)
            
            values = rannRecalc(projhyb, staterann1, ann, values)

            k2_state, k2_jac = odesfun(ann, t + h2, state1, jac + h2 * k1_jac, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
           
            k2_state = np.array(k2_state)
            k2_state = k2_state.astype(np.float64)
            k2_state = torch.from_numpy(k2_state)

            state2, staterann2 = update_state(state, h2, k2_state)
            values = rannRecalc(projhyb, staterann2, ann, values)

            k3_state, k3_jac = odesfun(ann, t + h2, state2, jac + h2 * k2_jac, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
            
            k3_state = np.array(k3_state)
            k3_state = k3_state.astype(np.float64)
            k3_state = torch.from_numpy(k3_state)

        else:
            state1, staterann1 = update_state(state, h2, k1_state)

            values = rannRecalc(projhyb, staterann1, ann, values)

            k2_state = odesfun(ann, t + h2, state1, None, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)

            state2, staterann2 = update_state(state, h2, k2_state)

            staterann1 = rannRecalc(projhyb, staterann2, ann, values)

            k3_state = odesfun(ann, t + h2, state2, None, None, w, ucontrol2, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)

        hl= h - h / 1e10
        ucontrol4 = controlfun(t + hl, batch) if controlfun is not None else []

        if jac != None:

            state3, staterann3 = update_state(state, hl, k3_state)

            values = rannRecalc(projhyb, staterann3, ann, values)

            k4_state, k4_jac = odesfun(ann, t + hl, state3, jac + hl * k3_jac, None, w, ucontrol4, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
            k4_state = np.array(k4_state)
            k4_state = k4_state.astype(np.float64)
            k4_state = torch.from_numpy(k4_state)
        else:
            state3, staterann3 = update_state(state, hl, k3_state)

            values = rannRecalc(projhyb, staterann3, ann, values)

            k4_state = odesfun(ann, t + hl, state3, None, None, w, ucontrol4, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann)
        

        if jac != None:
            stateFinal = calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state)
            state = extract_species_values(projhyb, stateFinal)

        else :
            stateFinal = calculate_state_final_nojac(state, h, k1_state, k2_state, k3_state, k4_state)
            state = extract_species_values(projhyb, stateFinal)

        if jac != None:
            jac = jac + h * (k1_jac / 6 + k2_jac / 3 + k3_jac / 3 + k4_jac / 6)

        t = t + h
    
    return t, stateFinal, jac, hess

def anninp_rann_func(projhyb, state):


    species_values = extract_species_values(projhyb, state)

    print("species_values", species_values)

    totalsyms = ["t", "dummyarg1", "dummyarg2", "w"]

    for i in range(1, projhyb["nspecies"]+1):
        totalsyms.append(projhyb["species"][str(i)]["id"])  # Species

    for i in range(1, projhyb["ncompartment"]+1):
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
    anninp_mat = []    
    rann = []


    for i in range(1,  projhyb["mlm"]["nx"]+1):
        totalsyms.append(projhyb["mlm"]["x"][str(i)]["id"])

        val_str = projhyb["mlm"]["x"][str(i)]["val"]
        max_str = projhyb["mlm"]["x"][str(i)]["max"]

        try:
            val_expr = sp.sympify(val_str)
            max_expr = sp.sympify(max_str)
        except Exception as e:
            raise ValueError(f"Error sympifying val or max: {e}")


        if not isinstance(val_expr, sp.Symbol):
            val_expr = sp.Symbol(val_str)


        anninp.append(val_expr/max_expr)

        val_mat = val_expr.evalf(subs=species_values)
        max_mat = max_expr.evalf(subs=species_values)

        anninp_mat.append(val_mat/max_mat)

    for i in range(1, projhyb["mlm"]["ny"]+1):
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["id"])  # Ann outputs
        totalsyms.append(projhyb["mlm"]["y"][str(i)]["val"])

        rann.append(sp.sympify(projhyb["mlm"]["y"][str(i)]["val"]))

    totalsyms = symbols(totalsyms)


    anninp_symbol = sp.sympify(anninp)

    return anninp_symbol, rann, anninp_mat


def extract_species_values(projhyb, state):
    species_values = {}

    print("state",state)
    
    '''
    for key, species in projhyb['species'].items():
        species_id = species['id']
        print("species_id", species_id)
        species_val = state[int(key)-1]
        print("species_val", species_val)
        species_values[species_id] = float(species_val)
        print("species_values", species_values)
    '''

    
    for i in range(0, len(state)-1):
        species_id = projhyb['species'][str(i+1)]["id"]
        species_val = state[i]
        species_values[species_id] = float(species_val)
    
    
    species_values['V'] = float(state[-1])

    print("species_values_extra", species_values)

    return species_values


def update_state(state, h2, k1_state):
    new_state = {}
    stateRann = []
    
    for index, (species_id, value) in enumerate(state.items()):

        if index < len(k1_state):
            new_value = value + h2 * k1_state[index]
            new_state[species_id] = new_value
        else:
            new_state[species_id] = value

        stateRann.append(new_value)
    
    return new_state, stateRann


def calculate_state_final(state, h, k1_state, k2_state, k3_state, k4_state):
    stateFinal = []
    
    for i, value in enumerate(state.values()):
        new_value = value + h * (k1_state[i] / 6 + k2_state[i] / 3 + k3_state[i] / 3 + k4_state[i] / 6)
        stateFinal.append(new_value.item())

    
    return stateFinal



def calculate_state_final_nojac(state, h, k1_state, k2_state, k3_state, k4_state):
    stateFinal = []

    for i, value in enumerate(state.values()):
        new_value = value + h * (k1_state[i] / 6 + k2_state[i] / 3 + k3_state[i] / 3 + k4_state[i] / 6)
        stateFinal.append(new_value)

    
    return stateFinal


def ensure_dict(statedict):
    if isinstance(statedict, np.ndarray):
        if statedict.ndim == 0:
            statedict = statedict.item()  
        elif statedict.ndim == 1:
            raise ValueError("Cannot convert 1-D array to a dictionary without keys.")
        else:
            raise ValueError("statedict is an array with unexpected dimensions.")
    
    if not isinstance(statedict, dict):
        raise ValueError("statedict should be a dictionary or a compatible array.")
    
    return statedict


def rannRecalc(projhyb, state, ann, values):
    anninp, rann, anninp_mat = anninp_rann_func(projhyb, state)
    anninp_mat = [expr.subs(values) for expr in anninp_mat]


    anninp_tensor = torch.tensor(anninp_mat, dtype=torch.float64)
    anninp_tensor = anninp_tensor.view(-1, 1)       

    activations = [anninp_tensor]
    
    y = activations[-1]

    rann_results = ann.forward(y)

    rann_results = rann_results.detach().numpy()

    for range_y in range(0, len(rann_results)):
        values["rann"+str(range_y+1)] = rann_results[range_y].item()

    return values


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

    print("fState", fState)
    print("len(fState)", len(fState))

    return fState


def parse_rule_val(rule_val):
    # Extract all unique variables from the string
    symbols_in_expr = {}
    for term in rule_val.replace('*', ' ').replace('/', ' ').split():
        symbols_in_expr[term] = sp.Symbol(term)
    
    # Parse the expression using these symbols
    parsed_expr = parse_expr(rule_val, local_dict=symbols_in_expr)
    return parsed_expr


def numerical_derivativeXY(x, y, values, delta=1e-5):
    derivatives = []
    for symbol in y:
        original_value = values[str(symbol)]
        
        values[symbol] = original_value + delta
        x_plus_delta = [expr.evalf(subs=values) for expr in x]
        
        values[symbol] = original_value - delta
        x_minus_delta = [expr.evalf(subs=values) for expr in x]
        
        derivative_for_symbol = [(fp - fm) / (2 * delta) for fp, fm in zip(x_plus_delta, x_minus_delta)]
        derivatives.append(derivative_for_symbol)
        
        values[symbol] = original_value
    
    return derivatives

def computeDFDS(projhyb, fstate, state_symbols, NValues):
    if projhyb['mlm']['DFDS'] is None:
        DfDs = numerical_diferentiation_torch(fstate, state_symbols, NValues)
        projhyb['mlm']['DFDS'] = DfDs
    else:
        DfDs = projhyb['mlm']['DFDS']

    print("DfDs before subs", DfDs)
    
    DfDs = DfDs.subs(NValues)
    
        
    DfDs = np.array(DfDs).reshape(len(fstate), len(state_symbols))


    if np.iscomplexobj(DfDs):
        DfDs = DfDs.real

    DfDs = torch.from_numpy(DfDs.astype(np.float64))
    return DfDs

def computeDFDRANN(projhyb, fstate, rann_symbol, NValues):

    if projhyb['mlm']['DFDRANN'] is None:
        DfDrann = numerical_diferentiation_torch(fstate, rann_symbol, NValues)
        projhyb['mlm']['DFDRANN'] = DfDrann
    else:
        DfDrann = projhyb['mlm']['DFDRANN']

        
    DfDrann = DfDrann.subs(NValues)
    DfDrann = np.array(DfDrann).reshape(len(fstate), projhyb["mlm"]["ny"])

    print("DfDrann", DfDrann)
    print("DFdrann size", DfDrann.size)

    if np.iscomplexobj(DfDrann):
        DfDrann = DfDrann.real
    DfDrann = torch.from_numpy(DfDrann.astype(np.float64))
    return DfDrann

def computeDANNINPDSTATE(projhyb, anninp, state_symbols, NValues):
    if projhyb['mlm']['DANNINPDSTATE'] is None:

        DanninpDstate = numerical_diferentiation_torch(anninp, state_symbols, NValues)

        projhyb['mlm']['DANNINPDSTATE'] = DanninpDstate
    else:
        DanninpDstate = projhyb['mlm']['DANNINPDSTATE']

    DanninpDstate = DanninpDstate.subs(NValues)

    DanninpDstate = np.array(DanninpDstate)
    if len(anninp) > 1:
        
        DanninpDstate = DanninpDstate.reshape(len(anninp), len(state_symbols))

    if np.iscomplexobj(DanninpDstate):
        DanninpDstate = DanninpDstate.real
    DanninpDstate = torch.from_numpy(DanninpDstate.astype(np.float64))
    return DanninpDstate

def computeBackpropagation(ann, anninp_tensor, projhyb):
    y, DrannDanninp, DrannDw = ann.backpropagate(anninp_tensor, projhyb['mlm']['ny'])

    return y, DrannDanninp, DrannDw

def computeDRANNDS(DrannDanninp, DanninpDstate):

    return torch.mm(DrannDanninp, DanninpDstate)

def computeDfDrannDrannDw(DfDrann, DrannDw):


    return torch.mm(DfDrann, DrannDw)

def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb, fstate, anninp, anninp_tensor, state_symbols, values, rann):

    if jac is None and hess is None:
        NValues = {**values, **state}
        fstate = [expr.subs(NValues) for expr in fstate]
        return fstate

    print("fst", fstate)
    print("state", state)
    print("jac", jac)
    print("anninp", anninp)
    print("anninp_tensor", anninp_tensor)
    print("state_symbols", state_symbols)
    print("values", values)
    print("rann", rann)


    if projhyb['mode'] == 1:
        NValues = {**values, **state}
        fstate = projhyb_cache.get('FSTATE', sp.sympify(fstate))
        state_symbols = projhyb_cache.get('STATE_SYMBOLS', sp.sympify(state_symbols))
        anninp = projhyb_cache.get('ANNINP', sp.sympify(anninp))

        projhyb_cache.update({
            'FSTATE': fstate,
            'STATE_SYMBOLS': state_symbols,
            'ANNINP': anninp
        })

        rann_symbol = rann

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_function = {
                executor.submit(computeDFDS, projhyb, fstate, state_symbols, NValues): 'DfDs',
                executor.submit(computeDFDRANN, projhyb, fstate, rann_symbol, NValues): 'DfDrann',
                executor.submit(computeDANNINPDSTATE, projhyb, anninp, state_symbols, NValues): 'DanninpDstate',
                executor.submit(computeBackpropagation, ann, anninp_tensor, projhyb): 'backpropagation'
            }

            results = {}
            for future in as_completed(future_to_function):
                key = future_to_function[future]
                results[key] = future.result()


        DfDs = results['DfDs']
        DfDrann = results['DfDrann']
        DanninpDstate = results['DanninpDstate']
        y, DrannDanninp, DrannDw = results['backpropagation']

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_function = {
                executor.submit(computeDRANNDS, DrannDanninp, DanninpDstate): 'DrannDs',
                executor.submit(computeDfDrannDrannDw, DfDrann, DrannDw): 'DfDrannDrannDw'
            }

            results = {}
            for future in as_completed(future_to_function):
                key = future_to_function[future]
                results[key] = future.result()

        DrannDs = results['DrannDs']
        DfDrannDrannDw = results['DfDrannDrannDw']
        print("DrannDs size", DrannDs.size())
        print("DfDrannDrannDw size", DfDrannDrannDw.size())
        print("DfDrann size", DfDrann.size())
        print("DfDs size", DfDs.size())

        DfDsDfDrannDrannDs = DfDs + torch.mm(DfDrann, DrannDs)
        print("DfDsDfDrannDrannDs size", DfDsDfDrannDrannDs.size())

        print(" torch.mm(DfDsDfDrannDrannDs, jac) size", torch.mm(DfDsDfDrannDrannDs, jac).size())
        fjac = torch.mm(DfDsDfDrannDrannDs, jac) + DfDrannDrannDw
        print("fjac size", fjac.size()) 


        fstate = [expr.subs(NValues) for expr in fstate]
        return fstate, fjac

    elif projhyb['mode'] == 3:
        anninp, rann, _ = anninp_rann_func(projhyb)
        fstate = fstate_func(projhyb)
        DfDs = Matrix([fstate]).jacobian(Matrix([state]))
        DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))
        fjac = DfDs * jac + DfDrann
        return fstate, fjac, None

    return None, None, None


def numerical_diferentiation_torch(x, y, values):
    values_dict = values.copy()
    values_dict_filtered = values.copy()
    derivatives = []
    
    x = Matrix(x)

    matrix = x.jacobian(y)

    return matrix

@functions_framework.http
def hybtrain_cloud_function(request):
    try:
        # Parse JSON payload from the request
        request_json = request.get_json(silent=True)

        # Extract data from the request
        projhyb = request_json.get('projhyb')
        data = request_json.get('data')
        user_id = request_json.get('user_id')
        trained_weights = request_json.get('trained_weights')
        hmod_content = request_json.get('hmod_content')

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Call hybtrain function
        projhyb_result, trainData, metrics, newHmodFile, plot_files = hybtrain(
            projhyb, data, user_id, trained_weights, hmod_content, temp_dir
        )

        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        bucket_name = os.environ.get("STORAGE_BUCKET_NAME")
        bucket = storage_client.bucket(bucket_name)

        folder_id = str(uuid.uuid4())

        # Upload newHmodFile to GCS
        new_hmod_filename = os.path.basename(newHmodFile)
        blob = bucket.blob(f"{user_id}/{folder_id}/{new_hmod_filename}")
        blob.upload_from_filename(newHmodFile)
        blob.make_public()
        new_hmod_url = blob.public_url

        # Upload plots to GCS
        plot_urls = []
        for plot_file in plot_files:
            plot_filename = os.path.basename(plot_file)
            plot_blob = bucket.blob(f"{user_id}/plots/{folder_id}/{plot_filename}")
            plot_blob.upload_from_filename(plot_file)
            plot_blob.make_public()
            plot_urls.append(plot_blob.public_url)

        # Prepare response data
        response_data = {
            'projhyb': projhyb_result,
            'trainData': trainData,
            'metrics': metrics,
            'new_hmod_url': new_hmod_url,
            'new_hmod_filename': new_hmod_filename,
            'plot_urls': plot_urls
        }

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return (json.dumps(response_data), 200, {'Content-Type': 'application/json'})

    except Exception as e:
        logging.error("Error in hybtrain_cloud_function: %s", str(e), exc_info=True)
        return (json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'})