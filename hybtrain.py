import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize


with open("sample.json", "r") as read_file:
    projhyb = json.load(read_file)


def hybtrain(projhyb):

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

    projhyb = {}
    projhyb['ntrain'] = 0
    projhyb['istrain'] = [0] * projhyb['nbatch']
    istrainSAVE = [0] * projhyb['nbatch']

    projhyb['itr'] = []  # List to keep track of training indices.
    cnt_jctrain = 0  # Counter for the number of training noise samples.
    cnt_jcval = 0    # Counter for the number of validation noise samples.
    cnt_jctest = 0   # Counter for the number of test noise samples.

    # Loop through each batch in projhyb.
    for i in range(projhyb['nbatch']):
        # Copy the 'istrain' value from the current batch to the 'istrain' list.
        projhyb['istrain'][i] = projhyb['batch'][i]['istrain']

        # If the current batch is a training batch:
        if projhyb['batch'][i]['istrain'] == 1:
           # Increment the training batch counter.
            projhyb['ntrain'] += 1
            # Append the current index to the training indices list.
            projhyb['itr'].append(i)
            # Count the noise samples in the current training batch.
            cnt_jctrain += len(projhyb['batch'][i]['cnoise'])

        # If the current batch is a test batch:
        elif projhyb['batch'][i]['istrain'] == 3:
            # Count the noise samples in the current test batch.
            cnt_jctest += len(projhyb['batch'][i]['cnoise'])

    if projhyb.get('bootstrap', None) == 1:
        if 'nbootstrap' not in projhyb:
            projhyb['nbootstrap'] = projhyb['ntrain']
            projhyb['nstep'] = projhyb['ntrain']
        else:
            projhyb['nstep'] = projhyb['nbootstrap']

        # If 'nbootrate' key is not present in the dictionary:
        if 'nbootrate' not in projhyb:
            projhyb['nbootrate'] = 2/3  # Set a default boot rate of 2/3.

        # Calculate the actual number of bootstrap samples based on the boot rate.
        nboot = max(1, int(projhyb['ntrain'] * projhyb['nbootrate']))

    ofun1 = lambda x1, x2, x3: outFun1(x1, x2, x3, projhyb)

    print("\nTraining method:")

    # Check training mode
    if projhyb['mode'] == 1:
        print("   Mode:                   Indirect")
    elif projhyb['mode'] == 2:
        print("   Mode:                   Direct")
    elif projhyb['mode'] == 3:
        print("   Mode:                   Semidirect")

    # Check Jacobian mode
    jacobian = 'off'
    if projhyb['jacobian'] == 0:
        print("   Jacobian:               OFF")
    elif projhyb['jacobian'] == 1:
        print("   Jacobian:               ON")
        jacobian = 'on'

    # Check Hessian mode
    hessian = 'off'
    if projhyb['hessian'] == 0:
        print("   Hessian:               OFF")
    elif projhyb['hessian'] == 1:
        print("   Hessian:               ON")
        hessian = 'on'

    print(f"   Steps:                  {projhyb['nstep']}")
    print(f"   Displayed iterations:   {projhyb['niter']}")
    print(
        f"   Total iterations:       {projhyb['niter'] * projhyb['niteroptim']}")

    if projhyb['bootstrap'] == 1:
        print("   Bootstrap:              ON")
        print(f"   Bootstrap repetitions:    {projhyb['nbootstrap']}")
        print(f"   Bootstrap permutations: {nboot}/{projhyb['ntrain']}")
    else:
        print("   Bootstrap:              OFF")

    if projhyb['method'] == 1:
        options = {
            'method': 'lm',
            'jac': jacobian,
            'max_nfev': 100000,
            'xtol': 1e-9,
            'ftol': 1e-12,
            'verbose': projhyb['display'],
        }

        options['max_nfev'] = projhyb['niter'] * projhyb['niteroptim']

        options['x_scale'] = ofun1

        result = least_squares(fun, x0, **options)

        print(f"   Optimiser:              Levenberg-Marquardt")

    elif projhyb['method'] == 2:
        algorithm = 'L-BFGS-B'  # default quasi-Newton method in SciPy

        if projhyb['jacobian'] == 1:
            algorithm = 'trust-constr'

        print(f"   Optimiser:              {algorithm}")

        options = {
            'method': algorithm,
            'jac': jacobian,
            'hess': hessian,
            'options': {
                'disp': projhyb['display'],
                'maxiter': projhyb['niter'] * projhyb['niteroptim'],
            },
            'callback': ofun1,
        }

        if projhyb['derivativecheck']:
            if not derivative_check(fun, jacobian, x0):
                print("Stopping optimization due to derivative check failure.")
                exit()

        result = minimize(fun, x0, **options)

    elif projhyb['method'] == 3:  # Not needed
        print("   Optimiser:              Simulated Annealing")

        options = {
            'display': projhyb['display'],
            'maxiter': projhyb['niter'] * projhyb['niteroptim'],
            'outputfcn': ofun1
        }
        ParsLB = -20 * np.ones(projhyb['mlm']['nw'])
        ParsUB = 20 * np.ones(projhyb['mlm']['nw'])

    elif projhyb['method'] == 4:
        print("   Optimiser:              Adam")
        npall = 0
        for l in range(projhyb['nbatch']):
            if projhyb['istrain'][l] == 1:
                npall += len(projhyb['batch'][l]['t'])

    options['niter'] = projhyb['niter'] * projhyb['niteroptim']

    print("\n\n")

    TrainRes = {
        'witer': [[0] * projhyb['mlm']['nw'] for _ in range(projhyb['nstep'] * projhyb['niter'] * 2)],
        'wstep': [[0] * projhyb['mlm']['nw'] for _ in range(projhyb['nstep'])],
        'istrain': [],
        'resnorm': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjctrain': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrtrain': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjcval': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrval': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjctest': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'sjrtest': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'AICc': [0] * (projhyb['nstep'] * projhyb['niter'] * 2),
        'mj': [0] * projhyb['nstep'],
        'iter': 0,
        'istep': 0,
        't0': time.time()
    }

    if 'initweights' not in projhyb:
        projhyb['initweights'] = 1

    if projhyb['initweights'] == 1:
        print('Weights initialization...')
        weights, ann = mlpnetinitw(projhyb['mlm']['fundata'])
        projhyb['mlm']['fundata'] = ann

    elif projhyb['initweights'] == 2:
        print('Read weights from file...')
        weights_data = load(projhyb['weightsfile'])
        weights = np.reshape(weights_data['wPHB0'], (-1, 1))
        projhyb['mlm']['fundata'] = feval(
            projhyb['mlmsetwfunc'], projhyb['mlm']['fundata'], weights)

    for istep in range(1, projhyb['nstep'] + 1):

        for i in range(projhyb['nbatch']):
            projhyb['istrain'][i] = projhyb['batch'][i]['istrain']

        if projhyb['bootstrap'] == 1:
            ind = sorted(np.random.permutation(projhyb['ntrain'])[:nboot])
            projhyb['istrain'][projhyb['itr']] = 0
            for idx in ind:
                projhyb['istrain'][projhyb['itr'][idx]] = 1

        TrainRes['istrain'].extend(projhyb['istrain'])

        if projhyb['mode'] == 1:  # INDIRECT
            if projhyb['method'] == 4:
                def fobj(w, istrain): return resfun_indirect_jac(
                    w, istrain, projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 0:
                def fobj(w): return resfun_indirect(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 1:
                def fobj(w): return resfun_indirect_jac(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        elif projhyb['mode'] == 2:  # DIRECT
            if projhyb['method'] == 4:
                def fobj(w, istrain): return resfun_direct_jac(
                    w, istrain, projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 0:
                def fobj(w): return resfun_direct(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 1:
                def fobj(w): return resfun_direct_jac(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        elif projhyb['mode'] == 3:  # SEMIDIRECT
            if projhyb['method'] == 4:
                def fobj(w, istrain): return resfun_semidirect_jac(
                    w, istrain, projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 0:
                def fobj(w): return resfun_semidirect(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['jacobian'] == 1:
                def fobj(w): return resfun_semidirect_jac(
                    w, projhyb['istrain'], projhyb, projhyb['method'])
            elif projhyb['hessian'] == 1:
                assert projhyb['hessian'] == 1, 'Hessian not yet implemented'

        if istep > 1:
            print('Weights initialization...')
            weights, ann = mlpnetinitw(projhyb['mlm']['fundata'])
            projhyb['mlm']['fundata'] = ann

        if projhyb['mlm']['nx'] == 0:
            for nparam in range(projhyb['mlm']['ny']):
                weights[nparam] = projhyb['mlm']['y'][nparam]['init']

        if method == 1:  # LEVENBERG-MARQUARDT
            result = least_squares(fobj, weights, method='lm', **options)
            wfinal, fval = result.x, result.cost

        elif method == 2:  # QUASI-NEWTON
            result = minimize(fobj, weights, method='BFGS', options=options)
            wfinal, fval = result.x, result.fun

        elif method == 3:  # SIMULATED ANNEALING
            from scipy.optimize import dual_annealing
            result = dual_annealing(fobj, bounds=list(
                zip(ParsLB, ParsUB)), **optopts)
            wfinal, fval = result.x, result.fun

        elif method == 4:  # ADAMS
            wfinal, fval = adamunlnew(fobj, weights, ofun1, projhyb, options)

    return projhyb, trainData

def mlpnetinitw(ann):
    w = []

    if ann['h'] == 1:
        ann['layer'][0]['w'] = np.random.randn(ann['nh'], ann['nx']) * np.sqrt(2 / (ann['nx'] + ann['nh']))
        ann['layer'][0]['b'] = np.zeros((ann['nh'], 1))
        ann['layer'][1]['w'] = np.random.randn(ann['ny'], ann['nh']) * np.sqrt(2 / (ann['nh'] + ann['ny']))
        ann['layer'][1]['b'] = np.zeros((ann['ny'], 1))
        
        w.extend(ann['layer'][0]['w'].ravel())
        w.extend(ann['layer'][0]['b'].ravel())
        w.extend(ann['layer'][1]['w'].ravel())
        w.extend(ann['layer'][1]['b'].ravel())

    else:
        count = 0
        for i in range(ann['h']):
            ann['layer'][i]['w'] = np.random.randn(ann['nh'][i], ann['nx'] if i == 0 else ann['nh'][i-1]) * np.sqrt(2 / (ann['nh'][i] + (ann['nx'] if i == 0 else ann['nh'][i-1])))
            ann['layer'][i]['b'] = np.zeros((ann['nh'][i], 1))

            w.extend(ann['layer'][i]['w'].ravel())
            w.extend(ann['layer'][i]['b'].ravel())

        ann['layer'][ann['h']]['w'] = np.random.randn(ann['ny'], ann['nh'][-1]) * np.sqrt(2 / (ann['ny'] + ann['nh'][-1]))
        ann['layer'][ann['h']]['b'] = np.zeros((ann['ny'], 1))
        
        w.extend(ann['layer'][ann['h']]['w'].ravel())
        w.extend(ann['layer'][ann['h']]['b'].ravel())

    w = np.array(w).reshape(-1, 1)
    ann['w'] = w
    return w, ann


def outFun1(witer, optimValues, optstate, projhyb):
    stop = False
    optnew = None
    changed = False

    if projhyb['method'] == 3:
        witer = optimValues['x']

    global TrainRes

    if optstate == 'init':
        TrainRes['iter0'] = TrainRes['iter'] + 1
        TrainRes['count'] = 0
        return stop, optnew, changed

    elif optstate == 'iter':
        TrainRes['count'] += 1
        if projhyb['method'] == 1:
            fvaliter = sum([x * x for x in optimValues['residual']]
                           ) / len(optimValues['residual'])
        else:
            fvaliter = optimValues['fval']

        if TrainRes['count'] < projhyb['niteroptim']:
            pass
        else:
            TrainRes['count'] = 0
            TrainRes = hybtrainiterres(TrainRes, witer, fvaliter, projhyb)

    elif optstate == 'done':
        if projhyb['method'] == 1:
            fvaliter = sum([x * x for x in optimValues['residual']]
                           ) / len(optimValues['residual'])
        else:
            fvaliter = optimValues['fval']

        TrainRes = hybtrainiterres(TrainRes, witer, fvaliter, projhyb)
        TrainRes['istep'] += 1

    elif optstate == 'interrupt':
        pass

    else:
        print("Big error")
        print(optstate)
        return stop, optnew, changed

    return stop, optnew, changed


def derivative_check(fun, jac, x0, tol=1e-6):
    f0 = fun(x0)
    j0 = jac(x0)
    for i in range(len(x0)):
        x1 = np.array(x0)
        x1[i] += tol
        f1 = fun(x1)
        numerical_derivative = (f1 - f0) / tol
        if np.abs(numerical_derivative - j0[i]) > tol:
            print("Derivative check failed.")
            return False
    print("Derivative check passed.")
    return True


####
#   INDIRECT
####



def resfun_indirect_jac(w, istrain, projhyb, method=1):
    if not istrain:
        istrain = projhyb["istrain"]

    ns = projhyb["nstate"]
    nw = projhyb["mlm"]["nw"]
    isres = projhyb["isres"]
    nres = projhyb["nres"]
    projhyb["mlm"]["fundata"] = projhyb["mlmsetwfunc"](
        projhyb["mlm"]["fundata"], w)

    npall = sum(projhyb["batch"][i]["np"]
                for i in range(projhyb["nbatch"]) if istrain[i] == 1)

    sresall = np.zeros(npall * nres)
    sjacall = np.zeros((npall * nres, projhyb["mlm"]["nw"]))

    COUNT = 0
    for l in range(projhyb["nbatch"]):
        if istrain[l] == 1:
            tb = projhyb["batch"][l]["t"]
            Y = projhyb["batch"][l]["state"]
            batch = projhyb["batch"][l]
            sY = projhyb["batch"][l]["sc"]
            state = projhyb["batch"][l]["state"][0, :].T
            Sw = np.zeros((ns, nw))

            for i in range(1, projhyb["batch"][l]["np"]):
                _, state, Sw = hybodesolver(projhyb["fun_hybodes_jac"],
                                            projhyb["fun_control"], projhyb["fun_event"], tb[i-1], tb[i],
                                            state, Sw, 0, [], batch, projhyb)

                sresall[COUNT:COUNT +
                        nres] = (Y[i, isres] - state[isres, 0]) / sY[i, isres]
                sjacall[COUNT:COUNT+nres, :nw] = - Sw[isres, :] / \
                    np.repeat(sY[i, isres].reshape(-1, 1), nw, axis=1)
                COUNT += nres

    ind = ~np.isnan(sresall)
    sresall = sresall[ind]
    sjacall = sjacall[ind, :]
    ind = ~np.isinf(sresall)
    sresall = sresall[ind]
    sjacall = sjacall[ind, :]

    fobj = np.nan
    if method == 1 or method == 4:
        fobj = sresall
        jac = sjacall
    else:
        fobj = np.dot(sresall.T, sresall) / len(sresall)
        jac = np.sum(2 * np.repeat(sresall.reshape(-1, 1), nw,
                     axis=1) * sjacall, axis=0) / len(sresall)

    return fobj, jac

def resfun_indirect_fminunc(w, istrain, projhyb):
    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    if 'mlmsetwfunc' in projhyb and projhyb['mlmsetwfunc'] is not None:
        # Assuming `projhyb['mlmsetwfunc']` is a callable function
        projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](projhyb['mlm']['fundata'], w)  # set weights

    sres = 0
    sjac = np.zeros(nw)

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            upars = projhyb['batch'][l]['u']
            sY = projhyb['batch'][l]['sc']
            np = len(tb)

            state = np.array(projhyb['batch'][l]['state'][0]).reshape(-1, 1)  # Convert the row into a column vector
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

def resfun_indirect_jac(w, istrain, projhyb, method=1):
    if not istrain:
        istrain = projhyb['istrain']

    ns = projhyb['nstate']
    nw = projhyb['mlm']['nw']
    isres = projhyb['isres']
    nres = projhyb['nres']

    projhyb['mlm']['fundata'] = projhyb['mlmsetwfunc'](projhyb['mlm']['fundata'], w)  # set weights

    npall = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            npall += projhyb['batch'][l]['np']

    sresall = np.zeros(npall * nres)
    sjacall = np.zeros((npall * nres, nw))

    COUNT = 0
    for l in range(projhyb['nbatch']):
        if istrain[l] == 1:
            tb = projhyb['batch'][l]['t']
            Y = projhyb['batch'][l]['state']
            batch = projhyb['batch'][l]
            sY = projhyb['batch'][l]['sc']
            state = np.array(projhyb['batch'][l]['state'][0]).reshape(-1, 1)
            Sw = np.zeros((ns, nw))

            for i in range(1, projhyb['batch'][l]['np']):
                # Assuming hybodesolver is a pre-defined function
                _, state, Sw = hybodesolver(projhyb['fun_hybodes_jac'],
                                            projhyb['fun_control'],
                                            projhyb['fun_event'],
                                            tb[i-1], tb[i], state, Sw, 0, None, batch, projhyb)

                sresall[COUNT:COUNT+nres] = (Y[i, isres] - state[isres, 0]) / sY[i, isres]
                sjacall[COUNT:COUNT+nres, :] = -Sw[isres, :] / np.repeat(sY[i, isres].reshape(-1, 1), nw, axis=1)
                COUNT += nres

    # Remove missing values from residuals
    ind = ~np.isnan(sresall)
    sresall = sresall[ind]
    sjacall = sjacall[ind, :]

    ind = ~np.isinf(sresall)  # Remove infinity values from residuals
    sresall = sresall[ind]
    sjacall = sjacall[ind, :]

    fobj = float('nan')
    if method == 1 or method == 4:
        fobj = sresall
        jac = sjacall
    else:
        fobj = np.dot(sresall.T, sresall) / len(sresall)
        jac = np.sum(2 * np.repeat(sresall.reshape(-1, 1), nw, axis=1) * sjacall, axis=0) / len(sresall)

    return fobj, jac






