import pandas as pd
import numpy as np

df = pd.read_csv('chassbatch1 copy.csv')

time_data = ['Time']
time_data.extend(df.iloc[:, 0].tolist())

odd_columns_data = [df.columns[1::2].tolist()]
odd_columns_data.extend(df.iloc[:, 1::2].values.tolist())

even_columns_data = [df.columns[2::2].tolist()]
even_columns_data.extend(df.iloc[:, 2::2].values.tolist())


print("Time data:", time_data)
print("Odd columns data:", odd_columns_data)
print("Even columns data:", even_columns_data)


time_matrix = np.array(time_data)
odd_columns_matrix = np.array(odd_columns_data)
even_columns_matrix = np.array(even_columns_data)


print("Time matrix:", time_matrix)
print("Odd columns matrix:", odd_columns_matrix)
print("Even columns matrix:", even_columns_matrix)




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
