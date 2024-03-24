import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.optimize import check_grad
import scipy.io
import json
import time
import matplotlib.pyplot as plt
from mlpnetinit import mlpnetinitw
from mlpnetcreate import mlpnetcreate
import torch
from mlpnetsetw import mlpnetsetw
from hybodesolver import hybodesolver
from odesfun import odesfun
from Control_functions.control_function_chass import control_function
import customMLP as mlp
from hybtrainiterres import hybtrainiterres

with open("sample.json", "r") as f:
    projhyb = json.load(f)

def default_fobj(w):
    raise NotImplementedError(
        "Objective function fobj is not properly defined.")


def hybtrain(projhyb, file):

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

    projhyb['itr'] = []  # List to keep track of training indices.
    cnt_jctrain = 0  # Counter for the number of training noise samples.
    cnt_jcval = 0    # Counter for the number of validation noise samples.
    cnt_jctest = 0   # Counter for the number of test noise samples.

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
    # Check training mode
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
    print(
        f"   Total iterations:       {projhyb['niter'] * projhyb['niteroptim']}")

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

    if projhyb['method'] == 1:
        print("   Optimiser:              Levenberg-Marquardt")
        options = {
            'xtol': 1e-15, #1e-15
            'ftol': 1e-15,
            'gtol': 1e-14,
            'verbose': 2,#projhyb['display'],
            'max_nfev': 100 * projhyb['niter'] * projhyb['niteroptim'], # projhyb['niter'] * projhyb['niteroptim'], # Ideally, this should be set to: 100 * projhyb['niter'] * projhyb['niteroptim']
            'method': 'lm',
        }

    elif projhyb['method'] == 2:
        algorithm = 'L-BFGS-B' if projhyb['jacobian'] != 1 else 'trust-constr'
        print(f"   Optimiser:              {algorithm}")
        options = {
            'method': algorithm,
            'hess': hessian,
            'options': {
                'disp': projhyb['display'],
                'maxiter': projhyb['niter'] * projhyb['niteroptim']
            },
            'callback': lambda xk: outFun1(xk, state)
        }

    elif projhyb['method'] == 3:
        print("   Optimiser:              Simulated Annealing")
        bounds = [(-20, 20)] * projhyb['mlm']['nw']
        options = {
            'maxiter': projhyb['niter'] * projhyb['niteroptim'],
            'disp': projhyb['display'],
            'callback': lambda xk: outFun1(xk, state)
        }

    elif projhyb['method'] == 4:
        print("   Optimiser:              Adam")
        npall = sum(len(projhyb['batch'][l]['t']) for l in range(
            projhyb['nbatch']) if projhyb['istrain'][l] == 1)
        options['niter'] = projhyb['niter'] * projhyb['niteroptim']

    print("\n\n")

    NH = projhyb['mlm']['options']
    H = len(NH)
    projhyb["mlm"]['h'] = H
    projhyb["mlm"]['nl'] = 2 + H
    projhyb["mlm"]['nh'] = NH[:H]
    projhyb["mlm"]['nw'] = (projhyb["mlm"]['nx'] + 1) * projhyb["mlm"]['nh'][0]
    projhyb["mlm"]["ninp"] = projhyb["mlm"]["nx"]
    projhyb["mlm"]["nout"] = projhyb["mlm"]["ny"]

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
        'witer': np.zeros((projhyb['nstep'] * projhyb['niter'] * 2, projhyb['mlm']['nw'])),
        'wstep': np.zeros((projhyb['nstep'], projhyb['mlm']['nw'])),
        'istrain': [],
        'count': 0,
        'resnorm': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjctrain': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrtrain': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjcval': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrval': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjctest': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrtest': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'AICc': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'mj': np.zeros((projhyb['nstep'], 1)),
        'iter': 0,
        'istep': 0,
        't0': time.time()
    }

    projhyb["mlm"]["DFDS"] = None
    projhyb["mlm"]["DFDRANN"] = None
    projhyb["mlm"]["DANNINPDSTATE"] = None
    projhyb["mlm"]["ANNINP"] = None
    projhyb['mlm']["FSTATE"] = None
    projhyb['mlm']['STATE_SYMBOLS'] = None

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

    for istep in range(1, projhyb['nstep']):
        for i in range(1, file['nbatch'] + 1):
            istrain = file[str(i)]["istrain"]
            projhyb['istrain'] = [0] * file['nbatch']
            projhyb['istrain'][i - 1] = istrain
            print(projhyb['istrain'])

        if projhyb['bootstrap'] == 1:
            ind = sorted(np.random.permutation(projhyb['ntrain'])[:nboot])
            projhyb['istrain'][projhyb['itr']] = 0
            for idx in ind:
                projhyb['istrain'][projhyb['itr'][idx]] = 1

        TrainRes['istrain'].extend(projhyb['istrain'])

        evaluator = IndirectFunctionEvaluator(ann, projhyb)
        
        '''
        if projhyb['mode'] == 1:  # INDIRECT
            if projhyb['method'] == 4:
                def fobj_func(weights):
                    fobj, _ = resfun_indirect_jac(ann, weights, projhyb['istrain'], projhyb, projhyb['method'])
                    return fobj
                def jac_func(weights):
                    _, jac = resfun_indirect_jac(ann, weights, projhyb['istrain'], projhyb, projhyb['method'])
                    return jac
            elif projhyb['jacobian'] == 0:
                def fobj_func(weights):
                    fobj = resfun_indirect(weights, projhyb['istrain'], projhyb, projhyb['method'])
                    return fobj
            elif projhyb['jacobian'] == 1:
                def fobj_func(weights):
                    fobj, _ = resfun_indirect_jac(ann, weights, projhyb['istrain'], projhyb, projhyb['method'])
                    return fobj
                def jac_func(weights):
                    _, jac = resfun_indirect_jac(ann, weights, projhyb['istrain'], projhyb, projhyb['method'])
                    return jac
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        elif projhyb['mode'] == 2:  # DIRECT
            if projhyb['method'] == 4:
                fobj,jac = resfun_direct_jac(weights, istrain, projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 0:
                fobj = resfun_direct(weights, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 1:
                fobj,jac = resfun_direct_jac(weights, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        elif projhyb['mode'] == 3:  # SEMIDIRECT
            if projhyb['method'] == 4:
                fobj,jac = resfun_semidirect_jac(weights, istrain, projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 0:
                fobj = resfun_semidirect(weights, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 1:
                fobj,jac = resfun_semidirect_jac(weights, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        #????
        
        if projhyb['mlm']['nx'] == 0:
            for nparam in range(projhyb['mlm']['ny']):
                weights[nparam] = projhyb['mlm']['y'][nparam]['init']
        '''
        if istep > 1:
            print('Weights initialization...2')
            weights, ann = ann.reinitialize_weights()
            projhyb["w"] = weights

        print(
            'ITER  RESNORM    [C]train   [C]valid   [C]test   [R]train   [R]valid   [R]test    AICc       NW   CPU')

        if projhyb["method"] == 1:  # LEVENBERG-MARQUARDT
            print("optios", options)
            print("weights", weights)

            result = least_squares(evaluator.fobj_func, weights, jac=evaluator.jac_func, **options)

            callback_wrapper(result, TrainRes, projhyb, istep)
            

        elif projhyb["method"] == 2:  # QUASI-NEWTON
            result = minimize(fobj, weights, method='BFGS', options=options)
            wfinal, fval = result.x, result.fun

        elif projhyb["method"] == 3:  # SIMULATED ANNEALING
            from scipy.optimize import dual_annealing
            result = dual_annealing(fobj, bounds=list(
                zip(ParsLB, ParsUB)), **optopts)
            wfinal, fval = result.x, result.fun

        elif projhyb["method"] == 4:  # ADAMS
            wfinal, fval = adamunlnew(fobj, weights, ofun1, projhyb, options)
                
    TrainRes["finalcpu"] = time.process_time() - TrainRes["t0"]
    projhyb["istrain"] = istrainSAVE

    sort_indices = np.argsort(TrainRes["mj"], axis=0).ravel()

    TrainRes["mj"] = TrainRes["mj"][sort_indices]
    TrainRes["wstep"] = TrainRes["wstep"][sort_indices, :]

    trainData = TrainRes
    scipy.io.savemat('hybtrain_results.mat', {"trainData": TrainRes})

#######################################################################################################################
    '''
    # Plot training results
    plt.figure()
    x = np.arange(1, TrainRes["iter"] + 1)
    plt.semilogy(x, TrainRes["resnorm"][:TrainRes["iter"]],
                 'g-', linewidth=2, label='fobj')
    plt.semilogy(x, TrainRes["sjctrain"][:TrainRes["iter"]],
                 'b-', linewidth=2, label='train')
    plt.semilogy(x, TrainRes["sjcval"][:TrainRes["iter"]],
                 'b:', linewidth=2, label='valid')
    plt.semilogy(x, TrainRes["sjctest"][:TrainRes["iter"]],
                 'r-', linewidth=2, label='test')

    plt.gca().set_xlim([1, max(result.x)])
    ymin = min([np.min(TrainRes["resnorm"][:TrainRes["iter"]]), np.min(TrainRes["sjctrain"][:TrainRes["iter"]]), np.min(
        TrainRes["sjcval"][:TrainRes["iter"]]), np.min(TrainRes["sjctest"][:TrainRes["iter"]])])
    ymax = max([np.max(TrainRes["resnorm"][:TrainRes["iter"]]), np.max(TrainRes["sjctrain"][:TrainRes["iter"]]), np.max(
        TrainRes["sjcval"][:TrainRes["iter"]]), np.max(TrainRes["sjctest"][:TrainRes["iter"]])])

    plt.gca().set_ylim(
        [10**np.floor(np.log10(ymin)), 10**np.ceil(np.log10(ymax))])
    plt.xlabel('# iteration')
    plt.ylabel('MSE')
    plt.title('concentrations')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.show()

    data = TrainRes['mj'][:, 1]

    fig, ax = plt.subplots(2, 2)
    N, bins, _ = ax[0, 0].hist(data, edgecolor='b', facecolor=[
                               1, 0.2, 0.87], linewidth=2)
    ax[0, 0].set_linewidth(2)

    xmin = np.floor(min(bins) * 0.8)
    xmax = np.floor(max(bins) * 1.2)

    # failsafe for when the minimum and maximum are very close
    if xmin == xmax:
        xmax = xmin + 1

    xscal = np.linspace(xmin, xmax, 5)
    ymax = (np.floor(max(N) / 5) + 1) * 5

    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_xticks(xscal)
    ax[0, 0].set_ylim([0, ymax])
    ax[0, 0].set_yticks(np.arange(0, ymax + 1, ymax / 5))

    ax[0, 0].set_xlabel('MSE_c (train)', fontname='Times', fontsize=18)
    ax[0, 0].set_ylabel('count', fontname='Times', fontsize=18)

    for tick in ax[0, 0].get_xticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)

    for tick in ax[0, 0].get_yticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)

    plt.tight_layout()
    plt.show()

    # Histogram of MSE concentrations for validation
    data = TrainRes['mj'][:, 2]
    N, bins, _ = ax[0, 1].hist(data, edgecolor='b', facecolor=[
                               1, 0.2, 0.87], linewidth=2)
    ax[0, 1].set_linewidth(2)

    xmin = np.floor(min(bins) * 0.8)
    xmax = np.floor(max(bins) * 1.2)

    # failsafe for when the minimum and maximum are very close
    if xmin == xmax:
        xmax = xmin + 1

    xscal = np.linspace(xmin, xmax, 5)
    ymax = (np.floor(max(N) / 5) + 1) * 5

    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_xticks(xscal)
    ax[0, 1].set_ylim([0, ymax])
    ax[0, 1].set_yticks(np.arange(0, ymax + 1, ymax / 5))

    ax[0, 1].set_xlabel('MSE_c (valid)', fontname='Times', fontsize=18)
    ax[0, 1].set_ylabel('count', fontname='Times', fontsize=18)

    # Adjusting font and size of tick labels
    for tick in ax[0, 1].get_xticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)
    for tick in ax[0, 1].get_yticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)

    # Histogram of MSE rates training
    data = TrainRes['mj'][:, 4]
    N, bins, _ = ax[1, 0].hist(data, edgecolor='b', facecolor=[
                               1, 0.2, 0.87], linewidth=2)
    ax[1, 0].set_linewidth(2)

    xmin = np.floor(min(bins) * 0.8)
    xmax = np.floor(max(bins) * 1.2)

    # failsafe for when the minimum and maximum are very close
    if xmin == xmax:
        xmax = xmin + 1

    xscal = np.linspace(xmin, xmax, 5)
    ymax = (np.floor(max(N) / 5) + 1) * 5

    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_xticks(xscal)
    ax[1, 0].set_ylim([0, ymax])
    ax[1, 0].set_yticks(np.arange(0, ymax + 1, ymax / 5))

    ax[1, 0].set_xlabel('MSE_r (train)', fontname='Times', fontsize=18)
    ax[1, 0].set_ylabel('count', fontname='Times', fontsize=18)

    # Adjusting font and size of tick labels
    for tick in ax[1, 0].get_xticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)
    for tick in ax[1, 0].get_yticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)

    # Histogram of MSE rates validation
    data = TrainRes['mj'][:, 5]
    N, bins, _ = ax[1, 1].hist(data, edgecolor='b', facecolor=[
                               1, 0.2, 0.87], linewidth=2)
    ax[1, 1].set_linewidth(2)

    xmin = np.floor(min(bins) * 0.8)
    xmax = np.floor(max(bins) * 1.2)

    # failsafe for when the minimum and maximum are very close
    if xmin == xmax:
        xmax = xmin + 1

    xscal = np.linspace(xmin, xmax, 5)
    ymax = (np.floor(max(N) / 5) + 1) * 5

    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_xticks(xscal)
    ax[1, 1].set_ylim([0, ymax])
    ax[1, 1].set_yticks(np.arange(0, ymax + 1, ymax / 5))

    ax[1, 1].set_xlabel('MSE_r (valid)', fontname='Times', fontsize=18)
    ax[1, 1].set_ylabel('count', fontname='Times', fontsize=18)

    # Adjusting font and size of tick labels
    for tick in ax[1, 1].get_xticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)
    for tick in ax[1, 1].get_yticklabels():
        tick.set_fontname("Times")
        tick.set_fontsize(18)

    plt.tight_layout()
    plt.show()

    if projhyb['crossval'] == 1:
        ind = np.where(TrainRes['sjcval'][:TrainRes['iter']]
                       == np.min(TrainRes['sjcval'][:TrainRes['iter']]))
        istep = ind[0][0]
    else:
        ind = np.where(TrainRes['sjctrain'][:TrainRes['iter']] == np.min(
            TrainRes['sjctrain'][:TrainRes['iter']]))
        istep = ind[0][0]

    print("\n\nBest step:", istep)
    wfinal = TrainRes['witer'][istep, :]
    projhyb['w'] = wfinal
    projhyb['wensemble'] = TrainRes['wstep']

    header_format = "{:<5} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<3}"
    data_format = "{:<5} {:<10.2E} {:<10.2E} {:<10.2E} {:<10.2E} {:<10.2E} {:<10.2E} {:<10.2E} {:<10.2E} {:<3}"

    print(header_format.format("STEP", "RESNORM", "[C]train", "[C]valid", "[C]test", "[R]train", "[R]valid", "[R]test", "AICc", "NW"))
    print(data_format.format(
        istep,
        TrainRes['resnorm'][istep],
        TrainRes['sjctrain'][istep],
        TrainRes['sjcval'][istep],
        TrainRes['sjctest'][istep],
        TrainRes['sjrtrain'][istep],
        TrainRes['sjrval'][istep],
        TrainRes['sjrtest'][istep],
        TrainRes['AICc'][istep],
        projhyb['mlm']['nw']
    ))

    print("AVE", " ".join(["{:<10.2E}".format(
        np.mean(TrainRes['mj'][:, i])) for i in range(8)]) + " None")
    print("STD", " ".join(["{:<10.2E}".format(
        np.std(TrainRes['mj'][:, i])) for i in range(8)]) + " None")
    print("CPU:", "{:<10.2E}".format(TrainRes['finalcpu']))

    '''

    return projhyb, trainData

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
    # outFun1(witer, optimValues, optstate, projhyb, TrainRes)
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

def outFun1(witer, optimValues, optstate, projhyb, TrainRes):
    stop = False
    optnew = None
    changed = False

    if projhyb['method'] == 3:
        witer = optimValues['x']
    
    if optstate == 'init':
        TrainRes['iter0'] = TrainRes['iter'] + 1
        TrainRes['count'] = 0
    elif optstate == 'iter':
        TrainRes['count'] += 1
        fvaliter = optimValues.get('fun', sum(x**2 for x in optimValues.get('residual', [])) / len(optimValues.get('residual', [1])))
        
        if TrainRes['count'] >= projhyb['niteroptim']:
            TrainRes['count'] = 0
            TrainRes = hybtrainiterres(TrainRes, witer, fvaliter, projhyb)
    elif optstate == 'done':
        fvaliter = optimValues.get('fun', sum(x**2 for x in optimValues.get('residual', [])) / len(optimValues.get('residual', [1])))
        TrainRes = hybtrainiterres(TrainRes, witer, fvaliter, projhyb)
        TrainRes['istep'] += 1
    elif optstate == 'interrupt':
        
        pass
    else:
        print("Unexpected optimization state:", optstate)

    return stop, optnew, changed
        
####
#   INDIRECT
####

def resfun_indirect_jac(ann, w, istrain, projhyb, method=1):
    print("weights", w)

# LOAD THE WEIGHTS into the ann
    ann.set_weights(w)
    ann.print_weights_and_biases()
    with open("file.json", "r") as read_file:
        file = json.load(read_file)
    if not istrain:
        istrain = projhyb["istrain"]

    # ires = 11 
    ns = projhyb["nspecies"]
    nt = ns + projhyb["ncompartments"]
    nw = projhyb["mlm"]["nw"]
    isres = []
    isresY = []
    for i in range(1, ns + 1):
        if projhyb["species"][str(i)]["isres"] == 1: 
            isres = isres + [i]
            isresY = isresY + [i - 1]

    isres = isres
        
    nres = len(isres)
    
    npall = sum(file[str(i+1)]["np"] for i in range(file["nbatch"]) if file[str(i+1)]["istrain"] == 1)

    sresall = np.zeros(npall * nres)

    sjacall = np.zeros((npall * nres, nw))

    print("npall", npall*nres)
    print("npall", nres)

    print("nn:", ann)

    COUNT = 0
    for l in range(file["nbatch"]):
        l = l + 1
        if file[str(l)]["istrain"] == 1:
            tb = file[str(l)]["time"]
            Y = file[str(l)]["y"]
            Y = np.array(Y)
            Y = Y.astype(np.float64)
            Y = torch.from_numpy(Y)
            
            batch = str(l)

            sY = file[str(l)]["sy"]
            sY = np.array(sY)
            sY = sY.astype(np.float64)
            sY = torch.from_numpy(sY)
            
            state = np.array(file[str(l)]["y"][0])
            Sw = np.zeros((nt, nw))

            for i in range(1, file[str(l)]["np"]):
                batch_data = file[str(l+1)]
                _, state, Sw, hess = hybodesolver(ann,odesfun,
                                            control_function , projhyb["fun_event"], tb[i-1], tb[i],
                                            state, Sw, 0, w, batch_data, projhyb)
                
                Y_select = Y[i, isresY]
                state_tensor = torch.tensor(state, dtype=torch.float64)
                state_adjusted = state_tensor[0:nres]
                Ystate = Y_select - state_adjusted.numpy()

                sresall[COUNT:COUNT + nres] = Ystate / sY[i, isresY].numpy()

                SYrepeat = sY[i, isresY].reshape(-1, 1).repeat(1, nw).numpy()
                result = (- Sw[isresY, :].detach().numpy()) / SYrepeat
                sjacall[COUNT:COUNT + nres, 0:nw] = result
                COUNT += nres
                print("#################################################")
                print("------------------LOOP", i, "------------------")
                print("#################################################")
                print("state", state)


    if method == 1 or method == 4:
        fobj = sresall
        jac = sjacall

        '''
        jac_max_abs = np.max(np.abs(jac))
        jac_scaled = jac / jac_max_abs
        '''
    else:
        fobj = np.dot(sresall_filtered.T, sresall_filtered) / len(sresall_filtered)
        jac = np.sum(2 * np.repeat(sresall_filtered.reshape(-1, 1), nw,
                     axis=1) * sjacall_filtered, axis=0) / len(sresall_filtered)
                     
    return fobj, jac

def resfun_indirect_fminunc(w, istrain, projhyb):
    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    if 'mlmsetwfunc' in projhyb and projhyb['mlmsetwfunc'] is not None:
        projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
            projhyb['mlm']['fundata'], w) 

    sres = 0
    sjac = np.zeros(nw)

    COUNT = 0
    print(file["nbatch"])
    for l in range(file['nbatch']):
        if file[str(l)]["istrain"] == 1:
            tb = file[str(l)]["time"]
            Y = file[str(l)]["y"]

            if 'u' not in file[str(l)]:
                upars = []
            else:
                upars = file[str(l)]["u"]  # VERIFICAR !!!!!!!!!!!!!!!!

            sY = file[str(l)]["sy"]

            np = file[str(l)]["np"]

            # Convert the row into a column vector

            first = file[str(l)]["time"][0]
            print(first)
            n_columns = len(file[str(l)][str(first)]["state"])
            print(n_columns)
            first_row = file[str(l)]["y"][:n_columns]
            print(first_row)
            first_row_column = np.array(first_row).reshape(1, n_columns)
            print(first_row_column)

            state = np.array(file[str(l)]['y'][0]).reshape(-1, 1)
            Sw = np.zeros((ns, nw))

            for i in range(1, np):
                # Assuming `hybodesolver` is a previously defined function
                _, state, Sw = hybodesolver(projhyb['fun_hybodes_jac'],
                                            projhyb['fun_control'],
                                            projhyb['fun_event'],
                                            tb[i-1], tb[i], state, Sw, 0, w, upars, projhyb)
                for j in range(nres):
                    k1 = isres[j]
                    if not np.isnan(Y[i][k1]):
                        res = (Y[i][k1] - state[k1][0]) / sY[i][k1]
                        sres = sres + res**2
                        sjac = sjac + (-2 * res / sY[i][k1]) * Sw[k1, :]
                        COUNT += 1

    sres = sres / COUNT
    sjac = sjac / COUNT
    return sres, sjac


def resfun_indirect_fminunc_hess(w, istrain, projhyb):
    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = len(isres)

    if projhyb['mlmsetwfunc']:
        projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
            projhyb['mlm']['fundata'], w)  # set weights

    sres = 0
    sjac = np.zeros(nw)
    shess = np.zeros((nw, nw))

    COUNT = 0

    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            upars = projhyb['batch'][l]['u']
            sY = projhyb['batch'][l]['sc']
            np = len(tb)

            state = projhyb['batch'][l]['state'][0, :].T
            Sw = np.zeros((ns, nw))
            Hsw = np.zeros((ns * nw, nw))

            for i in range(1, np):
                _, state, Sw, Hsw = hybodesolver(
                    projhyb['fun_hybodes_jac_hess'], projhyb['fun_control'], projhyb['fun_event'], tb[i-1], tb[i],
                    state, Sw, Hsw, w, upars, projhyb
                )

                for j in range(nres):
                    k1 = isres[j]
                    if not np.isnan(Y[i, k1]):
                        COUNT += 1
                        res = (Y[i, k1] - state[k1, 0]) / sY[i, k1]
                        sres += res * res
                        sjac -= 2 * res / sY[i, k1] * Sw[k1, :]
                        shess += (2 / sY[i, k1] ** 2) * np.outer(Sw[k1, :nw], Sw[k1, :nw]) - (
                            2 * res / sY[i, k1]) * Hsw[(k1-1) * nw: k1 * nw, :nw]

    sres /= COUNT
    sjac /= COUNT
    shess /= COUNT

    return sres, sjac, shess


def resfun_indirect(w, istrain=None, projhyb=None, method=1):
    if projhyb is None:
        raise ValueError("projhyb argument is required.")

    if istrain is None:
        istrain = projhyb['istrain']

    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
        projhyb['mlm']['fundata'], w)  # set weights

    npall = sum(projhyb['batch'][i]['np']
                for i in range(projhyb['nbatch']) if istrain[i] == 1)
    resall = np.zeros(npall * nres)

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            upars = projhyb['batch'][l]['u']
            sY = projhyb['batch'][l]['sc']
            np_ = len(tb)
            state = projhyb['batch'][l]['state'][0, :]

            for i in range(1, np_):
                _, state = hybodesolver(
                    projhyb['fun_hybodes'], projhyb['fun_control'], projhyb['fun_event'],
                    tb[i - 1], tb[i], state, 0, 0, w, upars, projhyb
                )
                resall[COUNT:COUNT +
                       nres] = (Y[i, isres] - state[isres]) / sY[i, isres]
                COUNT += nres

    # finally remove missing values from residuals
    resall = resall[~np.isnan(resall)]
    resall = resall[~np.isinf(resall)]

    if method == 1:
        return resall
    else:
        return resall.T @ resall / len(resall)

####
#   DIRECT
####


def resfun_direct_jac(w, istrain=None, projhyb=None, method=1):
    if projhyb is None:
        raise ValueError("projhyb argument is required.")

    if istrain is None:
        istrain = projhyb['istrain']

    isress = projhyb['isresstate']
    ns = len(isress)
    nw = projhyb['mlm']['nw']

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
        projhyb['mlm']['fundata'], w)  # set weights

    npall = sum(projhyb['batch'][i]['np']
                for i in range(projhyb['nbatch']) if istrain[i] == 1)
    resall = np.zeros(npall * ns)
    jacall = np.zeros((npall * ns, nw))

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            r = projhyb['batch'][l]['rnoise'] 
            upars = projhyb['batch'][l]['u']
            sr = projhyb['batch'][l]['sr']
            np_ = len(tb)

            for i in range(np_):
                tt = tb[i]
                state = projhyb['batch'][l]['state'][i, :]
                ucontrol = projhyb['fun_control'](tt, upars)
                rhyb_v, _, _, DrhybDw = projhyb['fun_hybrates_jac'](
                    tt, state, w, ucontrol, projhyb)

                resall[COUNT:COUNT+ns] = (r[i, isress] -
                                          rhyb_v[isress]) / sr[i, isress]
                jacall[COUNT:COUNT+ns, :] = -DrhybDw[isress, :] / \
                    np.repeat(sr[i, isress], nw).reshape(-1, 1)

                COUNT += ns

    # finally remove missing values from residuals
    ind = ~np.isnan(resall)
    resall = resall[ind]
    jacall = jacall[ind, :]

    ind = ~np.isinf(resall)  # Remove infinity values from residuals
    resall = resall[ind]
    jacall = jacall[ind, :]

    if method == 1:  # levenbergmarquardt
        fobj = resall
        jac = jacall
    else:
        fobj = resall.T @ resall / len(resall)
        jac = np.sum(2 * np.repeat(resall, nw).reshape(-1, nw)
                     * jacall, axis=0) / len(resall)

    return fobj, jac

####
#   SEMIDIRECT
####


def resfun_semidirect(w, istrain=None, projhyb=None, method=1):
    if projhyb is None:
        raise ValueError("projhyb argument is required.")

    if istrain is None:
        istrain = projhyb['istrain']

    isress = projhyb['isresstate']
    ns = len(isress)

    npall = sum(projhyb['batch'][i]['np']
                for i in range(projhyb['nbatch']) if istrain[i] == 1)
    resall = np.zeros(npall * ns)

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
        projhyb['mlm']['fundata'], w)

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            r = projhyb['batch'][l]['rnoise']
            upars = projhyb['batch'][l]['u']
            sr = projhyb['batch'][l]['sr']
            np_ = len(tb)

            for i in range(np_ - 1):
                tt = tb[i]
                state = projhyb['batch'][l]['state'][i, :]
                ucontrol = projhyb['fun_control'](tt, upars)

                rhyb_v, _ = projhyb['fun_hybrates'](
                    tt, state, w, ucontrol, projhyb)

                if i == 0:
                    resall[COUNT:COUNT+ns] = np.zeros(ns)

                resall[COUNT+ns:COUNT+2*ns] = resall[COUNT:COUNT+ns] + \
                    (r[i, :] - rhyb_v[isress]) / sr[i, isress]

                COUNT += ns

    ind = ~np.isnan(resall)
    resall = resall[ind]

    ind = ~np.isinf(resall)
    resall = resall[ind]

    if method == 1:
        fobj = resall
    else:
        fobj = resall.T @ resall / len(resall)

    return fobj

def resfun_semidirect_jac(w, projhyb, istrain=None, method=1):
    if istrain is None:
        istrain = projhyb['istrain']

    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    npall = sum(projhyb['batch'][i]['np']
                for i in range(projhyb['nbatch']) if istrain[i] == 1)
    resall = np.zeros(npall * nres)
    jacall = np.zeros((npall * nres, nw))

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
        projhyb['mlm']['fundata'], w)

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            upars = projhyb['batch'][l]['u']
            sY = projhyb['batch'][l]['sc']
            state = projhyb['batch'][l]['state'][0, :]
            Sw = np.zeros((ns, nw))
            jac = np.zeros((ns, projhyb['mlm']['ny']))

            for i in range(1, projhyb['batch'][l]['np']):
                _, state, jac = hybodesolver(projhyb['fun_hybodes_jac'],
                                             projhyb['fun_control'],
                                             projhyb['fun_event'],
                                             tb[i-1], tb[i],
                                             state, jac, 0, w, projhyb['batch'][l],
                                             projhyb)

                ucontrol = projhyb['fun_control'](tb[i], projhyb['batch'][l])
                inp = projhyb['mlm']['xfun'](tb[i], state, ucontrol)
                _, _, DrannDw = projhyb['mlm']['yfun'](
                    inp, w, projhyb['mlm']['fundata'])

                Sw = np.dot(jac, DrannDw)
                resall[COUNT:COUNT +
                       nres] = (Y[i, isres] - state[isres]) / sY[i, isres]
                jacall[COUNT:COUNT+nres, :nw] = -Sw[isres, :] / \
                    np.tile(sY[i, isres], (nw, 1)).T
                COUNT += nres

    ind = ~np.isnan(resall)
    resall = resall[ind]
    jacall = jacall[ind, :]
    ind = ~np.isinf(resall)
    resall = resall[ind]
    jacall = jacall[ind, :]

    if method == 1 or method == 4:
        fobj = resall
        jac = jacall
    else:
        fobj = np.dot(resall, resall) / len(resall)
        jac = np.sum(2 * np.tile(resall, (nw, 1)).T *
                     jacall, axis=0) / len(resall)

    return fobj, jac


def resfun_semidirect_jac_batch(w, istrain, projhyb, method=1):
    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](
        projhyb['mlm']['fundata'], w)  # set weights

    COUNT = 1
    mse = 0
    grads = np.zeros(nw)

    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            upars = projhyb['batch'][l]['u']
            sY = projhyb['batch'][l]['sc']
            state = projhyb['batch'][l]['state'][0, :].T
            Sw = np.zeros((ns, nw))
            DstateDrann = np.zeros((ns, projhyb['mlm']['ny']))

            for i in range(1, projhyb['batch'][l]['np']):
                _, state, DstateDrann = hybodesolver(
                    projhyb['fun_hybodes_jac'], projhyb['fun_control'], projhyb['fun_event'], tb[i-1], tb[i],
                    state, DstateDrann, 0, w, projhyb['batch'][l], projhyb
                )

                res = np.zeros(ns)
                res[isres] = (Y[i, isres] - state[isres]).T / sY[i, isres]
                ind = ~np.isnan(res)  # missing values
                mse_i = res[ind] @ res[ind].T

                DmseDsate = np.zeros(ns)
                DmseDsate[isres] = -2 * res / sY[i, isres]
                DmseDsate[~ind] = 0  # missing values

                DmseDrann = DmseDsate @ DstateDrann

                ucontrol = projhyb['fun_control'](tb[i], projhyb['batch'][l])
                inp = projhyb['mlm']['xfun'](tb[i], state, ucontrol)
                _, _, DmseDw = projhyb['mlm']['yfun'](
                    inp, w, projhyb['mlm']['fundata'], DmseDrann)

                mse += mse_i
                grads += DmseDw

                COUNT += nres

    mse /= COUNT
    grads /= COUNT

    return mse, grads

class IndirectFunctionEvaluator:
    def __init__(self, ann, projhyb):
        self.ann = ann
        self.projhyb = projhyb
        self.last_weights = None
        self.last_fobj = None
        self.last_jac = None
        self.scalar =None

    def evaluate(self, weights):
        if not np.array_equal(weights, self.last_weights):
            self.last_fobj, self.last_jac = resfun_indirect_jac(self.ann, weights, self.projhyb['istrain'], self.projhyb, self.projhyb['method'])
            self.last_weights = weights
            print("fobj", self.last_fobj)
            print("jac", self.last_jac)
            #TODO: fobj = np.mean(self.last_fobj) perguntar porque o scipy não aceita vetor so aceita um scalar value for "cost" or "fitness" value
            #self.scalar = np.sum(self.last_fobj)
            #TrainRes["istep"] += 1
            #weights, ann = mlp.CustomMLP.get_weights(ann)
            #projhyb["w"] = weights
        return self.last_fobj, self.last_jac#, self.scalar

    def fobj_func(self, weights):
        fobj, _ = self.evaluate(weights)
        return fobj

    def jac_func(self, weights):
        _, jac = self.evaluate(weights)
        return jac