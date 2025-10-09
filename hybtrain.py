import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import dual_annealing
import scipy.io
from scipy import stats
import json
import time
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlpnetinit import mlpnetinitw
from mlpnetcreate import mlpnetcreate
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim 
from mlpnetsetw import mlpnetsetw
from hybodesolver import hybodesolver
from odesfun import odesfun
from Control_functions.control_function_chass import control_function
import customMLP as mlp
from sklearn.metrics import mean_squared_error, r2_score
import types
import os
import uuid
import h5py
from savetrainednn import saveNN
import logging
from hybodesolver_neuralode import hybodesolver_neuralode
from scipy.stats import t as t_dist

def default_fobj(w):
    raise NotImplementedError(
        "Objective function fobj is not properly defined.")

def save_model_to_h5(model, file_path):
    state_dict = model.state_dict()
    with h5py.File(file_path, 'w') as h5file:
        for key, value in state_dict.items():
            h5file.create_dataset(key, data=value.cpu().numpy())

def make_mask(file, target_split, kfolds_idx=None):
    mask = []

    k = (kfolds_idx - 1) if kfolds_idx is not None else None

    for i in range(1, file['nbatch'] + 1):
        lab = file[i]['istrain']

        if isinstance(lab, list) or isinstance(lab, np.ndarray):
            if k is None:
                sel = (lab[0] == target_split)
            else:
                sel = (0 <= k < len(lab)) and (lab[k] == target_split)
        else:
            sel = (lab == target_split)

        mask.append(1 if sel else 0)

    return np.array(mask, dtype=int)


def hybtrain(projhyb, file, user_id, trainedWeights, hmod, temp_dir, run_id, thread=None):
    projhyb["run_id"] = run_id
    projhyb["thread"] = thread
    new_data = {}
    for key, value in file.items():
        try:
            int_key = int(key)
            new_data[int_key] = value
        except ValueError:
            new_data[key] = value

    file = new_data
    fobj = default_fobj

    if projhyb is None:
        raise ValueError("at least 1 input required for HYBTRAIN( projhyb)")

    
    if projhyb['nensemble'] <= 0:
        raise ValueError("nensemble must be at least 1")

    if projhyb['kfolds'] < projhyb['nensemble']:
        raise ValueError("kfolds must be >= nensemble")

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
        print("   Optimiser:              Dual Annealing")
        bounds = species_bounds * projhyb['mlm']['nw']
        options = {
            'maxiter': 10 * projhyb['niter'] * projhyb['niteroptim'],
        }

    elif projhyb['method'] == 4:
        print("   Optimiser:              Adam")
        num_epochs = projhyb['niter']
        lr = 0.001  

    elif projhyb['method'] == 5:
        print("   Optimiser:              Adam with New ODE Solver")
        num_epochs = projhyb['niter']
        lr = 0.001
#######################################################################################################################

    print("\n\n")

    '''
    for i in range(1, H):
        projhyb["mlm"]['nw'] += (projhyb["mlm"]['nh']
                                [i - 1] + 1) * projhyb["mlm"]['nh'][i]
    projhyb["mlm"]['nw'] += (projhyb["mlm"]['nh']
                            [H - 1] + 1) * projhyb["mlm"]['ny']
    '''

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
        '''
        
            weights = [-2.08210221e+01, -2.92648787e+01, -2.99616485e+00, -3.91873435e-01,
                        3.45529229e-01,  2.67547769e+00,  3.70737593e-01, -3.88445614e-01,
                        2.17403921e-02, -5.50984481e+00,  2.70463125e+00,  9.69372025e-04,
                        -8.27238985e+00,  9.76706417e-01, -1.80689630e+00,  2.59146860e-01,
                        -2.80553531e+00, -9.70047111e-01]
            

            weights = [-20.8210221361445, -29.2648787002075, -2.99616485291983, -0.391873434528431, 
                        0.345529228671208, 2.67547768790112, 0.370737593202186, -5.50984481311063, -8.27238985127028, 
                        -0.388445614425761, 2.70463125251191, 0.976706416952729, 0.0217403921016785, 0.000969372025403460, 
                        -1.80689629911906, 0.259146859752498, -2.80553530793061, -0.970047111391006]
            
            
            weights = [-0.05165798, -0.00395335,  0.0011845,  -0.00152316,  0.00073078, -0.00047021, -0.06688604,  
            0.00180839, -0.00095668,  0.06751172, -0.00933701,  0.00603163, -0.08951404, -0.00475901,  0.0039245,   
            0.2426019,   0.08341522,  0.04603258]

            weights = [-3.53128530e-03, -3.64901283e-03, -6.95024195e-04, -2.85856595e-03,
                        -5.72278609e-03,  8.94225432e-03, -7.36665101e-03, -1.41053907e-04,
                        -5.15862938e-03, -5.09503627e-03, -2.84337087e-04,  6.03933941e-03,
                        9.41305666e-04, 6.22917697e-03, -1.12921277e-02,  2.65024872e-03,
                        5.76868488e-03, -6.40813954e-04,  5.44998166e-03,  3.45273420e-04,
                        6.07342871e-03, -3.29895230e-03,  4.99788226e-04, 9.35608992e-04,
                        2.32294709e-03,  9.51788165e-03,  1.51280977e-03, -5.63223687e-03,
                        -7.07439327e-03,  7.08675558e-03,  2.00843228e-03, -4.95971486e-03,
                        1.95212150e-03, -1.47335719e-04,  2.43811046e-04, -2.69257092e-04,
                        4.67362160e-03, -2.73212714e-03,  1.09595711e-02, -6.76758513e-03,
                        3.36307047e-03, -8.42507510e-03, -2.45127986e-03,  7.38494327e-03,
                        -1.94209984e-03, -1.27761048e-03,  2.03488980e-03, -2.40765340e-03,
                        9.57643390e-04,  1.86669282e-03, -1.14740347e-03, -2.87645922e-03,
                        7.78870967e-03,  3.95378095e-03,  4.50924730e-03, -6.77753730e-03,
                        -4.73618884e-03, -2.22725296e-02,  1.69542144e-02,  3.56126289e-04,
                        1.34426003e-02,  6.97196956e-07,  9.26678843e-05, -4.25188021e-09]
            
            weights = [ 0.46595111,  0.10995199,  0.15210865,  0.13458372,  0.10450351, -0.05684199,  0.3605971,  -1.14533527,
            1.08100574, 0.49413338, -0.6545019,   1.12397129, -0.78435885, -2.40410101, -0.28280282,  0.24305272, -0.02864941,  0.88284648]
        
            weights= [-2.08210402e+01, -2.92648655e+01, -2.99617932e+00, -3.92740113e-01, 3.46148463e-01,
            2.67545926e+00,  3.71076656e-01, -3.89435816e-01, 2.17389612e-02, -5.50932823e+00,  2.70377037e+00,  9.69186093e-04,
            -8.27198878e+00,  9.77532981e-01, -1.80689401e+00,  2.58970797e-01, -2.80444098e+00, -9.70044794e-01]

            weights = np.array(weights)
        '''
    
        ann.set_weights(weights)
        


    elif projhyb['initweights'] == 2:
        print('Read weights from file...')
        weights_data = load(projhyb['weightsfile'])
        weights = np.reshape(weights_data['wPHB0'], (-1, 1))
        projhyb['mlm']['fundata'].set_weights(weights)
    
    projhyb["mlm"]['nw'] = len(weights)


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
    all_weights = []  
    val_errors = []


    for istep in range(1, projhyb['nstep']+1):
        for kfold in range(1, projhyb['kfolds']+1):
            projhyb['currentfold'] = kfold - 1 

            print(f"\n--- Training step {istep}, Fold {kfold} ---")
            for i in range(1, file['nbatch'] + 1):
                print(f"Batch {i}: istrain = {file[i]['istrain']}")
                istrain = file[i]["istrain"]
                projhyb['istrain'] = [0] * file['nbatch']
                projhyb['istrain'][i - 1] = istrain

            for i in range(1, file['nbatch'] + 1):
                istrain = file[i]["istrain"]
                projhyb['istrain'] = [0] * file['nbatch']
                projhyb['istrain'][i - 1] = istrain

            if projhyb['bootstrap'] == 1:
                ind = sorted(np.random.permutation(projhyb['ntrain'])[:nboot])
                projhyb['istrain'][projhyb['itr']] = 0
                for idx in ind:
                    projhyb['istrain'][projhyb['itr'][idx]] = 1
            
            
        
            ann = mlpnetcreate(projhyb, projhyb['mlm']['neuron'])
            projhyb['mlm']['fundata'] = ann
            weights, ann = ann.get_weights()
            ann.set_weights(weights)

            with open("filetest.json", "w") as f:
                json.dump(convert_numpy(weights), f)
                json.dump("weights", f)
        
            if projhyb['jacobian'] == 1:
                options['jac'] = evaluator.jac_func
                if projhyb['method'] == 3:
                    options['minimizer_kwargs'] = {
                        'method': 'L-BFGS-B',
                        'jac': evaluator.jac_func,
                        'options': {'maxiter': 100}
                    }

            if projhyb['hessian'] == 1:
                options['hess'] = evaluator.hess_func

            '''

                [-20.8210221361445, -29.2648787002075, -2.99616485291983, -0.391873434528431, 0.345529228671208, 2.67547768790112, 0.370737593202186, -5.50984481311063, -8.27238985127028, -0.388445614425761, 2.70463125251191, 0.976706416952729, 0.0217403921016785, 0.000969372025403460, -1.80689629911906, 0.259146859752498, -2.80553530793061, -0.970047111391006]

                            #transporse the weights when the layer.w is 3x3
                    
                    if layer.w.shape == torch.Size([3, 3]):
                        with torch.no_grad():
                            layer.w.copy_(layer.w.t().clone())
                    
                w = [-1.30284588e-03, -1.14245236e-02, -7.91831059e-04,  4.40199659e-03,
                        1.72756969e-03, -5.00282668e-03, -2.01651446e-03, -3.93168300e-03,
                        9.56139266e-04, -1.25006042e-03, -1.57937766e-03, -2.50723996e-03,
                        1.62134778e-03, -1.16238460e-02, -1.66073496e-04,  5.14919755e-03,
                        2.14947514e-03,  1.17489032e-04, -2.03476453e-03,  1.54795275e-01,
                        1.70781475e-01,  4.07317944e-05,  1.00251807e+00,  9.94084879e-02,
                        6.72249841e-06,  4.22301199e-02]

                w = [ 1.51872094e-03, -1.59585915e-03, -7.12253433e-04,  1.24342755e-04,
                        7.74540120e-04,  1.99441788e-02, -1.26969915e-02, -3.67767926e-03,
                        -1.40537984e-03, -5.17297002e-03, -1.85277002e-03,  1.74874954e-03,
                        6.56016046e-04, -1.54538042e-04, -5.62466811e-04,  2.64828828e-02,
                        -1.29078782e-02, -2.86998296e-03, -3.28937350e-03, -1.46798282e-02,
                        1.08838826e-02,  8.35603951e-03,  4.62468320e-03, -4.93299581e-03,
                        -3.38708910e-02,  9.56367378e-03, -8.93223669e-03, -3.34692422e-03,
                        5.63855163e-04,  2.80681532e-03,  2.34064976e-03, -4.90464784e-03,
                        -2.41182354e-03,  1.33099790e-03,  5.88513180e-03,  4.52984615e-03,
                        -4.60275981e-03, -1.85567481e-03, 4.55726909e-04,  2.12187582e-03,
                        -3.07892454e-03,  3.05248249e-03,  1.20104149e-03, -3.40335053e-04,
                        -1.44833498e-03,  1.40478742e-03, -1.47642337e-03, -6.05061645e-04,
                        1.23577693e-04,  6.19980360e-04,  9.93972677e-06, -1.17452681e-04,
                        -3.83150019e-06,  9.81991234e-05,  1.88650875e-04, -2.61882162e-08,
                        -5.74176106e-07,  4.48708336e-07,-1.96318907e-07,  5.34379366e-07,
                        -9.05059523e-04, -3.78891547e-03,  9.28477323e-04, -1.22795051e-03,
                        5.03147974e-03, -1.02264114e-02, -1.03661777e-02, -3.96977224e-03,
                        4.06746594e-03,  1.40531047e-02, -1.66942884e-02,  8.71160820e-03,
                        -1.79788117e-02,  1.99990204e-02, -9.71130284e-03, -1.26890938e-03,
                        8.88607362e-03, -3.87793796e-03,  5.82036485e-03, -1.08326037e-02,
                        6.17182015e-03, -1.52597979e-02,  1.01063687e-02, -1.36405502e-02,
                        1.84950105e-02,  8.53358615e-05,  3.37067528e-05,  3.55832699e-05,
                        -8.44830702e-05,  4.73012733e-05, -2.19510522e-04,  1.01640910e-03,
                        4.29801868e-02,  4.68624654e-05,  2.67207005e-05, -2.18452390e-02,
                        8.78507425e-06, -1.62294776e-04,  5.69775024e-04, -1.32757487e-02,
                        6.36695478e-05, -1.50478056e-05,  1.04180612e-02, -4.69567708e-05,
                        -1.05798717e-05, -4.99704584e-05,  2.04628426e-03, -1.88611539e-05,
                        5.00014177e-05, -2.44433960e-02, -2.20355572e-05,  9.94252867e-05,
                        -2.14544245e-04, -2.20562387e-03, -2.46881624e-05,  3.35512951e-05,
                        -2.15364604e-03,  5.72864861e-06,  1.70329603e-06, -1.20269089e-04,
                        -2.59561229e-02,  5.50277599e-05,  6.37817347e-05,  4.35776302e-02,
                        4.96333638e-05,  1.54874169e-01,  1.70770553e-01,  3.46536551e-05,
                        1.00330855e+00,  9.96498529e-02,  6.14672585e-06,  4.25715988e-02]

                w =  [9.84452500e-04, 3.50165149e-05, 1.15788796e-02, 2.94717984e-04,
                        2.61447847e-04, 3.10083638e-01, 9.53791945e-02, 3.72253961e-01]


                optimized_weights = w   

                w =  [9.84452500e-04, 3.50165149e-05, 1.15788796e-02, 2.94717984e-04,
                    2.61447847e-04, 3.10083638e-01, 9.53791945e-02, 3.72253961e-01]

                trainedWeights = w
            '''

            if trainedWeights == None:

                simulation = 0
        
                if projhyb["method"] == 1:  # LEVENBERG-MARQUARDT
                    #result = least_squares(evaluator.fobj_func, x0=weights, **options)
                    #print("result", result.x)
                    #optimized_weights = result.x

                    result = least_squares(evaluator.fobj_func, x0=weights, **options)

                    optimized_weights = evaluator.get_best_weights()

                    #all_weights.append(optimized_weights.copy())
                    all_weights.append(result.x.copy())
                    val_errors.append(evaluator.best_val_loss)

                    evaluator.restartEvaluator()


                elif projhyb["method"] == 2:  # QUASI-NEWTON

                    if options.get('method', None) == 'trust-constr':
                        result = minimize(evaluator.fobj_func, x0=weights, hess=None, **options)
                    else:
                        result = minimize(evaluator.fobj_func, x0=weights, **options)
                    
                    #optimized_weights = result.x
                    optimized_weights = evaluator.get_best_weights()


                elif projhyb["method"] == 3:  # Dual ANNEALING
                    # Dual Annealing does not support jac or hess
                    # valid_da_options = {'maxiter', 'initial_temp', 'restart_temp_ratio', 'visit', 'accept', 'maxfun', 'seed', 'no_local_search', 'callback', 'x0'}
                    valid_options = {'maxiter', 'minimizer_kwargs'}
                    d_A_options = {k: v for k, v in options.items() if k in valid_options}

                    result = dual_annealing(evaluator.fobj_func, bounds=bounds, **d_A_options)
                    #optimized_weights = result.x
                    optimized_weights = evaluator.get_best_weights()

                elif projhyb["method"] == 4:  # ADAM method
                    manual_optimizer = ManualAdamOptimizer(ann, lr=0.01, weight_decay=1e-5)
                    fobj_history = []
                    gradient_norms = []

                    weights = ann.get_weights_solo()

                    refresh_rate = max(1, int(num_epochs * 0.002))

                    for epoch in range(num_epochs):
                        manual_optimizer.zero_grad()

                        weights = ann.get_weights_solo()
                        fobjs, gradient = evaluator.evaluate_adam(weights)

                        grad_norm = np.linalg.norm(gradient)
                        obj_value = np.mean(fobjs ** 2)

                        if np.isnan(grad_norm) or grad_norm > 1e10:
                            raise ValueError(f"Gradient norm is invalid at epoch {epoch + 1}: {grad_norm}")

                        manual_optimizer.step(gradient)
                        manual_optimizer.step_scheduler(obj_value)

                        fobj_history.append(obj_value)
                        gradient_norms.append(grad_norm)

                        if epoch % refresh_rate == 0:
                            print(f"Epoch {epoch}: Loss={obj_value:.6f}, GradNorm={grad_norm:.6f}")

                    optimized_weights = ann.get_weights_solo()

                elif projhyb["method"] == 5:
                    print("   Optimiser:              ADAM with Neural ODE solver")

                    y0 = file[1]["y"][0]
                    if isinstance(y0, list):
                        y0_tensor = torch.tensor(y0, dtype=torch.float32)
                    else:
                        y0_tensor = torch.tensor([y0], dtype=torch.float32)

                    initial_state = y0_tensor.unsqueeze(1) if y0_tensor.ndim == 1 else y0_tensor.T

                    time_array = torch.tensor(file[1]["time"], dtype=torch.float32)

                    target_traj_list = []
                    for i in range(len(time_array)):
                        y_val = file[1]["y"][i]
                        if isinstance(y_val, list):
                            target_traj_list.append(y_val)
                        else:
                            target_traj_list.append([y_val])
                    target_traj = torch.tensor(target_traj_list, dtype=torch.float32)

                    optimizer = torch.optim.Adam(ann.parameters(), lr=1e-3)
                    num_epochs = projhyb.get("niter", 100)

                    for epoch in range(num_epochs):
                        optimizer.zero_grad()
                        pred_traj = hybodesolver_neuralode(ann, initial_state, time_array)
                        loss = torch.mean((pred_traj - target_traj) ** 2)
                        loss.backward()
                        optimizer.step()

                        if epoch % 10 == 0:
                            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.6f}")

                    optimized_weights = ann.get_weights_solo()
            else:
                
                simulation = 1

                trainedWeights = np.array(trainedWeights)

                ann.set_weights(trainedWeights)
                
                optimized_weights, _ = ann.get_weights()

        
        weights_matrix = np.stack(all_weights, axis=0)
        mean_weights = np.mean(weights_matrix, axis=0)




        fobj_value = evaluator.fobj_func(mean_weights)  
        fobj_norm  = np.linalg.norm(fobj_value)


        if bestPerformance == None:
            bestPerformance = fobj_norm
            bestWeights = mean_weights

        if bestPerformance > fobj_norm:
            bestPerformance = fobj_norm
            bestWeights = mean_weights

    val_errors = np.array(val_errors)
    all_weights = np.array(all_weights, dtype=object)

    sorted_indices = np.argsort(val_errors)
    top_n_indices = sorted_indices[:projhyb['nensemble']]
    top_n_weights = [all_weights[i] for i in top_n_indices]

    model_path = os.path.join(temp_dir, "trained_model.h5")
    save_model_to_h5(ann, model_path)

    newHmodFile = os.path.join(temp_dir, "Newhmod.hmod")
    saveNN(model_path, projhyb["inputs"], projhyb["outputs"], hmod, newHmodFile, bestWeights, ann)

    testing = teststate(ann, user_id, projhyb, file, top_n_weights, temp_dir, simulation, projhyb['method'], evaluator)

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

def resfun_indirect_jac(ann, w, istrain, projhyb, file, method=1, mask_type=None):

# LOAD THE WEIGHTS into the ann
    ann.set_weights(w)

    # ires = 11 
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
    
    current_fold = projhyb.get("currentfold", 0)  # Default to 0 if not set

    npall = sum(
        file[i + 1]["np"]
        for i in range(file["nbatch"])
        if isinstance(file[i + 1]["istrain"], list) and file[i + 1]["istrain"][current_fold] == 1
    )

    sresall = np.zeros(npall * nres)

    sjacall = np.zeros((npall * nres, nw))

    if mask_type == "train":
        nM = 1
    elif mask_type == "val":
        nM = 2


    COUNT = 0
    for l in range(file["nbatch"]):
        l = l + 1
        if file[l]["istrain"][projhyb['currentfold']] == nM:
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

                if projhyb.get("thread") and not getattr(projhyb["thread"], "do_run", True):
                    logging.info(f"Training interrupted by thread flag: {projhyb['run_id']}")
                    return projhyb, {}, {}, None

                statedict = file[l]["key_value_array"][i-1]

                dict_items_list = list(statedict.items())
                statedict =  dict(dict_items_list[ns:len(dict_items_list)])

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann,odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)

                
                
                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)

                state_tensor = state_tensor.to(dtype=torch.float32)
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
    

    ann.set_weights(w)
    #ann.print_weights_and_biases()
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

                if projhyb.get("thread") and not getattr(projhyb["thread"], "do_run", True):
                    logging.info(f"Training interrupted by thread flag: {projhyb['run_id']}")
                    return projhyb, {}, {}, None

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

    # LOAD THE WEIGHTS into the ANN
    ann.set_weights(w)
    #ann.print_weights_and_biases()
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
            state = state.to(dtype=torch.float32)
            statedict = np.array(file[l]["key_value_array"][0])

            for i in range(1, file[l]["np"]):

                if projhyb.get("thread") and not getattr(projhyb["thread"], "do_run", True):
                    logging.info(f"Training interrupted by thread flag: {projhyb['run_id']}")
                    return projhyb, {}, {}, None
                
                print("statedict", statedict)

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann, odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)

                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)
                state_tensor = state_tensor.to(dtype=torch.float32)
                state_adjusted = state_tensor[0:nres]
                Ystate = Y_select - state_adjusted

                sresall[COUNT:COUNT + nres] = Ystate.detach().numpy() / sY[i, isresY].numpy()

                Ystate.sum().backward()
                sjacall[COUNT:COUNT + nres, 0:nw] = state.grad[isresY, :].detach().numpy() / sY[i, isresY].numpy()
                state.grad.zero_()

                COUNT += nres

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
    def __init__(self, model, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.model = model
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            betas=betas, 
            eps=eps,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )

    def step(self, gradient):
        start = 0
        for param in self.model.parameters():
            num_elements = param.numel()
            grad_slice = gradient[start:start + num_elements]
            manual_grad = torch.from_numpy(grad_slice.astype(np.float32)).view_as(param).to(param.device)
            param.grad = manual_grad.detach()
            start += num_elements

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step_scheduler(self, loss_value):
        self.scheduler.step(loss_value)

'''
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
        try:
            residuals, jacobian = self.evaluation_function(
                self.ann, weights, self.projhyb['istrain'],
                self.projhyb, self.file, self.projhyb['method']
            )

            residuals = residuals.astype(np.float32, copy=False)
            jacobian = jacobian.astype(np.float32, copy=False)

            gradient = jacobian.T @ residuals

            self.last_weights = np.copy(weights)
            self.last_fobj = residuals
            self.last_jac = jacobian
            self.last_gradient = gradient

            self.fobj_history.append(np.linalg.norm(residuals))
            self.jac_norm_history.append(np.linalg.norm(gradient))

            logging.debug(f"Objective norm: {np.linalg.norm(residuals):.6f}, Gradient norm: {np.linalg.norm(gradient):.6f}")

        except Exception as e:
            logging.error(f"Error evaluating function: {e}", exc_info=True)
            raise

        return residuals, gradient
'''

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

        self.best_val_loss = float('inf')
        self.best_val_weights = None
        self.val_loss_history = []

    def evaluate(self, weights):
        if np.array_equal(weights, self.last_weights):
            return self.last_fobj, self.last_jac
        else:
            train_mask = make_mask(self.file, 1, self.projhyb['kfolds'])

            self.last_fobj, self.last_jac = self.evaluation_function(
                self.ann, weights, train_mask, self.projhyb, self.file, self.projhyb['method'], mask_type='train'
            )
            self.last_weights = weights

            self.fobj_history.append(self.last_fobj)
            self.jac_norm_history.append(np.linalg.norm(self.last_jac))

            self.check_validation_loss(weights)

            return self.last_fobj, self.last_jac

    def evaluate_adam(self, weights):
        residuals, jacobian = self.evaluation_function(
            self.ann, weights, self.projhyb['istrain'], self.projhyb, self.file, self.projhyb['method']
        )

        residuals = residuals.astype(np.float32, copy=False)
        jacobian = jacobian.astype(np.float32, copy=False)

        gradient = jacobian.T @ residuals

        self.last_weights = np.copy(weights)
        self.last_fobj = residuals
        self.last_jac = jacobian
        self.last_gradient = gradient

        self.fobj_history.append(np.linalg.norm(residuals))
        self.jac_norm_history.append(np.linalg.norm(gradient))

        self.check_validation_loss(weights)

        return residuals, gradient

    def check_validation_loss(self, weights):
        val_mask = make_mask(self.file, 2, self.projhyb['kfolds'])  
        val_residuals, _ = self.evaluation_function(
            self.ann, weights, val_mask, self.projhyb, self.file, self.projhyb['method'], mask_type='val'
        )
        val_loss = np.linalg.norm(val_residuals)
        self.val_loss_history.append(val_loss)

        if not np.isnan(val_loss) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_weights = np.copy(weights)

    def get_best_weights(self):
        return self.best_val_weights if self.best_val_weights is not None else self.last_weights

    def fobj_func(self, weights):
        fobj, _ = self.evaluate(weights)
        return fobj

    def jac_func(self, weights):
        _, jac = self.evaluate(weights)
        return jac

    def torch_fobj_func(self, weights_tensor):
        weights_numpy = weights_tensor.detach().cpu().numpy()
        residuals, _ = self.evaluate(weights_numpy)

        residuals_tensor = torch.tensor(residuals, dtype=torch.float32, requires_grad=True).to(weights_tensor.device)
        return torch.sum(residuals_tensor ** 2)

    def torch_grad_func(self, weights_tensor):
        weights_numpy = weights_tensor.detach().cpu().numpy()
        _, jac = self.evaluate(weights_numpy)
        return torch.tensor(jac, dtype=torch.float32, requires_grad=True)

    def restartEvaluator(self):
        self.last_weights = None
        self.last_fobj = None
        self.last_jac = None
        self.fobj_history = []
        self.jac_norm_history = []
        self.best_val_loss = float('inf')
        self.best_val_weights = None
        self.val_loss_history = []

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

def teststate(ann, user_id, projhyb, file, weights, temp_dir, simulation, method=1, evaluator=None):
    if simulation == 1:

        dictState = {}
        w = np.array(weights)

        # Load the weights into the ann
        ann.set_weights(w)
        #ann.print_weights_and_biases()

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
        if simulation == 1:
            for l in range(file["nbatch"]):
                l = l + 1
                if not isinstance(file[l], dict):
                    continue

                if file[l]["istrain"] not in [1, 3]:
                    continue

                tb = file[l]["time"]
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

                break

            for i in range(projhyb["nspecies"]):
                actual_test = []
                predicted_test = []
                err = []

                # Collect data for plotting from the processed batch
                if l in dictState:
                    for t in range(1, file[l]["np"]):
                        actual_test.append(np.array(file[l]["y"][t-1][i]))
                        if t in dictState[l]:
                            predicted_test.append(dictState[l][t-1][i])
                            err.append(file[l]["sy"][t-1][i])

                actual_test = np.array(actual_test, dtype=np.float64)
                predicted_test = np.array(predicted_test, dtype=np.float64)
                err = np.array(err, dtype=np.float64)
            
                # Plot time series
                x = file[l]["time"][:-1]

                fig, ax = plt.subplots(figsize=(10, 6))
                #ax.errorbar(x, actual_test[:len(x)], err[:len(x)], fmt='o', linewidth=1, capsize=6, label="Observed data", color='green', alpha=0.5)
                ax.plot(x, predicted_test[:len(x)], label="Predicted", color='red', linewidth=1)

                plt.xlabel('Time (s)')
                plt.ylabel('Concentration')
                plt.title(f"Specie {projhyb['species'][str(i+1)]['id']}", verticalalignment='bottom', fontsize=16, fontweight='bold')
                plt.legend()

                temp = os.path.join(temp_dir, 'plots')
                user_dir = os.path.join(temp, user_id)
                date_dir = os.path.join(user_dir,time.strftime("%Y%m%d%H%M"))
                os.makedirs(date_dir, exist_ok=True)
                
                plot_filename = os.path.join(date_dir, f'metabolite_{projhyb["species"][str(i+1)]["id"]}_{uuid.uuid4().hex}.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close(fig)

            return {"Simulated_Model": True}

        
    else:
        is_single_model = (projhyb.get("kfolds", 1) == 1 and len(weights) == 1)

        overall_metrics = {
            'mse_train': [],
            'mse_test': [],
            'r2_train': [],
            'r2_test': []
        }

        ensemble_predictions = {}  

        for k, w in enumerate(weights):
            clean_w = np.array(w, dtype=np.float32)
            ann.set_weights(clean_w)
            dictState = {}

            ns = projhyb["nspecies"]
            nt = ns + projhyb["ncompartment"]
            nw = projhyb["mlm"]["nw"]
            isres = [i for i in range(1, ns + 1) if projhyb["species"][str(i)]["isres"] == 1]
            isresY = [i - 1 for i in isres]
            nres = len(isres)

            train_batches = [i for i in file if isinstance(file[i], dict) and file[i]["istrain"][k] == 1]
            val_batches   = [i for i in file if isinstance(file[i], dict) and file[i]["istrain"][k] == 2]
            test_batches  = [i for i in file if isinstance(file[i], dict) and file[i]["istrain"][k] == 3]

            for l in range(1, file["nbatch"] + 1):
                if not isinstance(file[l], dict): continue

                tb = file[l]["time"]
                state = np.array(file[l]["y"][0])[0:ns+1]
                Sw = np.zeros((nt, nw))

                for i in range(1, file[l]["np"]):
                    statedict = dict(list(file[l]["key_value_array"][i-1].items())[ns:])
                    _, state, Sw, hess = hybodesolver(
                        ann, odesfun, control_function, projhyb["fun_event"],
                        tb[i-1], tb[i], state, statedict, None, None,
                        np.array(w), file[l], projhyb
                    )
                    if l not in dictState:
                        dictState[l] = {}
                        dictState[l][0] = file[l]["y"][0]
                    dictState[l][i] = state

            for i in range(projhyb["nspecies"]):
                for batch in test_batches:
                    for t_idx in range(1, file[batch]["np"]):
                        time_idx = t_idx - 1
                        try:
                            predicted_value = float(dictState[batch][t_idx - 1][i])
                        except Exception as e:
                            print(f"[Warning] Could not cast prediction to float at batch {batch}, t={t_idx}, species={i+1}: {e}")
                            continue

                        species_id = projhyb["species"][str(i+1)]["id"]
                        if species_id not in ensemble_predictions:
                            ensemble_predictions[species_id] = {}
                        if batch not in ensemble_predictions[species_id]:
                            ensemble_predictions[species_id][batch] = {}
                        if time_idx not in ensemble_predictions[species_id][batch]:
                            ensemble_predictions[species_id][batch][time_idx] = []

                        ensemble_predictions[species_id][batch][time_idx].append(predicted_value)


                actual_train, actual_test = [], []
                predicted_train, predicted_test = [], []
                err = []

                for batch in train_batches + val_batches:
                    for t_idx in range(1, file[batch]["np"]):
                        actual_train.append(file[batch]["y"][t_idx - 1][i])
                        predicted_train.append(dictState[batch][t_idx - 1][i])

                for batch in test_batches:
                    for t_idx in range(1, file[batch]["np"]):
                        actual_test.append(file[batch]["y"][t_idx - 1][i])
                        predicted_test.append(dictState[batch][t_idx - 1][i])
                        err.append(file[batch]["sy"][t_idx - 1][i])

                actual_train = np.array(actual_train)
                predicted_train = np.array(predicted_train)
                actual_test = np.array(actual_test)
                predicted_test = np.array(predicted_test)
                err = np.array(err)

                if actual_train.shape != predicted_train.shape or actual_test.shape != predicted_test.shape:
                    print(f"[Ensemble {k}] Shape mismatch for species {i+1}")
                    continue

                mse_train = mean_squared_error(actual_train, predicted_train)
                mse_test = mean_squared_error(actual_test, predicted_test)
                r2_train = r2_score(actual_train, predicted_train)
                r2_test = r2_score(actual_test, predicted_test)

                overall_metrics["mse_train"].append(mse_train)
                overall_metrics["mse_test"].append(mse_test)
                overall_metrics["r2_train"].append(r2_train)
                overall_metrics["r2_test"].append(r2_test)

        temp = os.path.join(temp_dir, 'plots')
        user_dir = os.path.join(temp, user_id)
        date_dir = os.path.join(user_dir, time.strftime("%Y%m%d%H%M"))
        os.makedirs(date_dir, exist_ok=True)

        for i in range(projhyb["nspecies"]):
            species_id = projhyb["species"][str(i+1)]["id"]
            batch = list(ensemble_predictions[species_id].keys())[0]
            x = file[batch]["time"][:-1]

            y_true = []
            y_mean = []
            y_ci = []

            for t_idx in sorted(ensemble_predictions[species_id][batch].keys()):
                preds = np.array(ensemble_predictions[species_id][batch][t_idx])
                mean_pred = np.mean(preds)
                std_pred = np.std(preds, ddof=1)

                n_preds = len(preds)
                t_score_val = t_dist.ppf(1 - 0.025, df=n_preds - 1) if n_preds > 1 else 2.0
                ci = t_score_val * std_pred / np.sqrt(float(n_preds))

                y_mean.append(mean_pred)
                y_ci.append(ci)
                y_true.append(file[batch]["y"][t_idx][i])

            y_mean = np.array(y_mean)
            y_ci = np.array(y_ci)
            y_true = np.array(y_true)
            lower_bound = np.clip(y_mean - y_ci, 0, None)
            upper_bound = y_mean + y_ci

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(x, y_true, err[:len(x)], fmt='o', linewidth=1, capsize=6, label="Observed data", color='green', alpha=0.5)
            
            ax.fill_between(x, lower_bound, upper_bound, color='gray', alpha=0.3, label='95% CI')

            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            if is_single_model:
                ax.plot(x, y_mean, label="Model Prediction", color='red')
                ax.set_title(f"{species_id}")
            else:
                ax.plot(x, y_mean, label="Ensemble Mean", color='red')
                ax.set_title(f"{species_id}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

            plt.savefig(os.path.join(date_dir, f'Plot_{species_id}.png'), dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(predicted_train, actual_train, color='blue', label='Train + Val', alpha=0.5)
            ax.scatter(predicted_test, actual_test, color='red', label='Test', alpha=0.5)

            max_val = max(np.max(predicted_test), np.max(actual_test), np.max(predicted_train), np.max(actual_train))
            ax.plot([0, max_val], [0, max_val], 'r--', color='gray', label='Ideal', alpha=0.25)

            ax.set_xlabel('Observed')
            ax.set_ylabel('Predicted')
            if is_single_model:
                ax.set_title(f"Predicted vs Observed for {species_id}")
            else:
                ax.set_title(f"Predicted vs Observed for {species_id} (Ensemble)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

            plt.savefig(os.path.join(date_dir, f'ParatyPlot_{species_id}.png'), dpi=300)
            plt.close(fig)


        '''
        save_path=os.path.join(temp_dir, "train_val_error.png")
        train_errors = [np.linalg.norm(r) if isinstance(r, np.ndarray) else r for r in evaluator.fobj_history]
        val_errors = evaluator.val_loss_history

        plt.figure(figsize=(8,5))
        plt.plot(train_errors, label="Training Error")
        plt.plot(val_errors, label="Validation Error")
        plt.xlabel("Iteration")
        plt.ylabel("Error (L2 norm)")
        plt.title("Training vs Validation Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        '''

        overall_mse_train = np.mean(overall_metrics['mse_train'])
        overall_mse_test = np.mean(overall_metrics['mse_test'])
        overall_r2_train = np.mean(overall_metrics['r2_train'])
        overall_r2_test = np.mean(overall_metrics['r2_test'])

        return {'mse_train': overall_mse_train, 'mse_test': overall_mse_test, 'r2_train': overall_r2_train, 'r2_test': overall_r2_test}