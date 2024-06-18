import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import dual_annealing
import scipy.io
import json
import time
import matplotlib.pyplot as plt
from mlpnetinit import mlpnetinitw
from mlpnetcreate import mlpnetcreate
import torch
from torch.utils.data import Dataset, DataLoader
from mlpnetsetw import mlpnetsetw
from hybodesolver import hybodesolver
from odesfun import odesfun
from Control_functions.control_function_chass import control_function
import customMLP as mlp
from sklearn.metrics import mean_squared_error, r2_score
import types
import os


def default_fobj(w):
    raise NotImplementedError(
        "Objective function fobj is not properly defined.")


def hybtrain(projhyb, file, user_id):
    print("USer ID", user_id)
    fobj = default_fobj

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
    # INDIRECT IS THE ONLY THAT IS IMPLEMENTED
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
        print("   Optimiser:              Levenberg-Marquardt")
        options = {
            'xtol': 1e-8, #1e-10
            'verbose': projhyb['display'],
            'max_nfev': 100 * projhyb['niter'] * projhyb['niteroptim'],
            'method': 'trf',
        }

    elif projhyb['method'] == 2:
        algorithm = 'L-BFGS-B' if projhyb['jacobian'] != 1 else 'trust-constr'
        print(f"   Optimiser:              {algorithm}")
        options = {
            'method': algorithm,
            'options': {
                'verbose': projhyb['display'],
                'maxiter':100 * projhyb['niter'] * projhyb['niteroptim']
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
        num_epochs = 1  
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

    elif projhyb['initweights'] == 2:
        print('Read weights from file...')
        weights_data = load(projhyb['weightsfile'])
        weights = np.reshape(weights_data['wPHB0'], (-1, 1))
        projhyb['mlm']['fundata'].set_weights(weights)

    weights = weights.ravel()
    #projhyb["w"]= weights
    
    istep = 1

    if projhyb['mode'] == 1:

        evaluator = IndirectFunctionEvaluator(ann, projhyb, file, resfun_indirect_jac)

#######################################################################################################################
    
    for istep in range(1, projhyb['nstep']):
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
            weights, ann = ann.reinitialize_weights()
            projhyb["w"] = weights
        print(
            'ITER  RESNORM    [C]train   [C]valid   [C]test   [R]train   [R]valid   [R]test    AICc       NW   CPU')

        
        if projhyb['jacobian'] == 1:
            options['jac'] = evaluator.jac_func

        if projhyb['hessian'] == 1:
            options['hess'] = evaluator.hess_func
        
        
        if projhyb["method"] == 1:  # LEVENBERG-MARQUARDT
            print("optios", options)
            print("weights", weights)
            result = least_squares(evaluator.fobj_func, x0=weights, **options)
            print("result", result.x)
            optimized_weights = result.x


        elif projhyb["method"] == 2:  # QUASI-NEWTON
            print("optios", options)
            print("weights", weights)

            if options.get('method', None) == 'trust-constr':
                result = minimize(evaluator.fobj_func, x0=weights, hess=None, **options)
            else:
                result = minimize(evaluator.fobj_func, x0=weights, **options)
            
            optimized_weights = result.x
            print("result", result.x)


        elif projhyb["method"] == 3:  # SIMULATED ANNEALING
            result = dual_annealing(evaluator.fobj_func, bounds=bounds, **options)
            optimized_weights = result.x

        elif projhyb["method"] == 4:  # ADAM
            optimized_weights = adam_optimizer_train(ann, projhyb, evaluator, num_epochs, lr)
    
    '''
    w = [-1.30284588e-03, -1.14245236e-02, -7.91831059e-04,  4.40199659e-03,
            1.72756969e-03, -5.00282668e-03, -2.01651446e-03, -3.93168300e-03,
            9.56139266e-04, -1.25006042e-03, -1.57937766e-03, -2.50723996e-03,
            1.62134778e-03, -1.16238460e-02, -1.66073496e-04,  5.14919755e-03,
            2.14947514e-03,  1.17489032e-04, -2.03476453e-03,  1.54795275e-01,
            1.70781475e-01,  4.07317944e-05,  1.00251807e+00,  9.94084879e-02,
            6.72249841e-06,  4.22301199e-02]
    
    optimized_weights = w   
    '''
    testing = teststate(ann, user_id, projhyb, file, optimized_weights, projhyb['method'])

    plot_optimization_results(evaluator.fobj_history, evaluator.jac_norm_history)    

    return projhyb, optimized_weights

def callback_wrapper(x, TrainRes, projhyb, istep):
    try:
        with open("results.json", "r") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        data = {}

    if "results" not in data:
        data["results"] = {}

    result_entry = {
        "solution": x.x,
        'message': x.message,
        "success": x.success,
        "fun": x.fun,
        "cost": x.cost,
        "grad": x.grad,
        "optimality": x.optimality,
        "Nfev": x.nfev, 
        "njev": x.njev,
        "jac": x.jac,
    }

    result_entry_converted = convert_numpy(result_entry)

    data["results"][TrainRes["istep"]] = result_entry_converted

    with open("results.json", "w") as file:
        json.dump(data, file, indent=4)

    witer = x
    optstate = 'iter'
    istep = istep + 1
    print(TrainRes["istep"])

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

    print("npall", npall*nres)
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
            Sw = np.zeros((nt, nw))

            for i in range(1, file[l]["np"]):
                statedict = np.array(file[l]["key_value_array"][i-1])
                print("statedict", statedict)

                batch_data = file[l]
                _, state, Sw, hess = hybodesolver(ann,odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, statedict, Sw, 0, w, batch_data, projhyb)
                
                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)
                state_adjusted = state_tensor[0:nres]
                Ystate = Y_select - state_adjusted.numpy()

                print("Ystate", Ystate)
                print("sY", sY)
                print("iresY", isresY)
                
                sresall[COUNT:COUNT + nres] = Ystate / sY[i, isresY].numpy()

                SYrepeat = sY[i, isresY].reshape(-1, 1).repeat(1, nw).numpy()
                result = (- Sw[isresY, :].detach().numpy()) / SYrepeat
                sjacall[COUNT:COUNT + nres, 0:nw] = result
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


def adam_optimizer_train(ann, projhyb, evaluator, num_epochs, lr):
    initial_weights, _ = ann.get_weights()
    initial_weights_tensor = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.Adam([initial_weights_tensor], lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        objFunc = evaluator.torch_fobj_func(initial_weights_tensor)

        objFunc.backward()

        print(f'Gradients after backward pass: {initial_weights_tensor.grad}')

        optimizer.step()

        optimized_weights = initial_weights_tensor.detach().cpu().numpy()
        ann.set_weights(optimized_weights)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {objFunc.item()}')
            print(f'Weights after epoch {epoch+1}: {optimized_weights}')

        initial_weights_tensor = torch.tensor(optimized_weights, dtype=torch.float32, requires_grad=True)

    final_weights = initial_weights_tensor.detach().cpu().numpy()
    return final_weights

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

def teststate(ann, user_id, projhyb, file, w, method=1):
    dictState = {}
    w = np.array(w)

    # LOAD THE WEIGHTS into the ann
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
        Sw = np.zeros((nt, nw))

        for i in range(1, file[l]["np"]):
            statedict = np.array(file[l]["key_value_array"][i-1])

            batch_data = file[l]
            _, state, Sw, hess = hybodesolver(ann, odesfun, control_function, projhyb["fun_event"], tb[i-1], tb[i], state, statedict, None, None, w, batch_data, projhyb)

            if l not in dictState:
                dictState[l] = {}
            dictState[l][i] = state

    for i in range(0, projhyb['mlm']['nx']):
        actual_train = []
        actual_test = []
        predicted_train = []
        predicted_test = []
        err = []

        train_batches = [batch for batch, data in file.items() if isinstance(data, dict) and data["istrain"] == 1]
        test_batches = [batch for batch, data in file.items() if isinstance(data, dict) and data["istrain"] == 3]

        for batch in train_batches:
            for t in range(1, file[batch]["np"]):
                actual_train.append(np.array(file[batch]["y"][t][i]))
                if batch in dictState and t in dictState[batch]:
                    predicted_train.append(dictState[batch][t][i])
                else:
                    print(f"Missing prediction for train batch {batch}, time {t}")

        for batch in test_batches:
            for t in range(1, file[batch]["np"]):
                actual_test.append(np.array(file[batch]["y"][t][i]))
                if batch in dictState and t in dictState[batch]:
                    predicted_test.append(dictState[batch][t][i])
                    err.append(file[batch]["sy"][t][i])
                else:
                    print(f"Missing prediction for test batch {batch}, time {t}")

        actual_train = np.array(actual_train, dtype=np.float64)
        actual_test = np.array(actual_test, dtype=np.float64)
        predicted_train = np.array(predicted_train, dtype=np.float64)
        predicted_test = np.array(predicted_test, dtype=np.float64)
        err = np.array(err, dtype=np.float64)

        if actual_train.shape != predicted_train.shape:
            print(f"Shape mismatch for training data: actual_train {actual_train.shape}, predicted_train {predicted_train.shape}")
            continue

        if actual_test.shape != predicted_test.shape:
            print(f"Shape mismatch for test data: actual_test {actual_test.shape}, predicted_test {predicted_test.shape}")
            continue
        
        mse_train = mean_squared_error(actual_train, predicted_train)
        print(f'Training MSE: {mse_train}')
        mse_test = mean_squared_error(actual_test, predicted_test)
        print(f'Test MSE: {mse_test}')

        r2_train = r2_score(actual_train, predicted_train)
        print(f'Training R²: {r2_train}')
        r2_test = r2_score(actual_test, predicted_test)
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

        x = file[train_batches[0]]["time"]  # Assuming time is the same for all batches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(x, actual_train[:len(x)], err[:len(x)], fmt='o', linewidth=1, capsize=6, label="Observed data", color='green', alpha=0.5)
        ax.plot(x, predicted_train[:len(x)], label="Predicted", color='red', linewidth=1)
        ax.fill_between(x, lower_bound[:len(x)], upper_bound[:len(x)], color='gray', label="Confidence Interval", alpha=0.5)

        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.title(f"Metabolite {projhyb['species'][str(i+1)]['id']} ", verticalalignment='bottom', fontsize=16, fontweight='bold')

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.legend()
        user_dir = os.path.join('plots', user_id)
        date_dir = os.path.join(user_dir, time.strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)
        
        # Save the plot
        plot_filename = os.path.join(date_dir, f'metabolite_{projhyb["species"][str(i+1)]["id"]}_{time.strftime("%H%M%S")}.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close(fig)