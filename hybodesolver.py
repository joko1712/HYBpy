import numpy as np
import torch

def hybodesolver(ann,odesfun, controlfun, eventfun, t0, tf, state, jac, hess, w, batch, projhyb):
    t = t0
    hopt = []

    jac = torch.tensor(jac, dtype=torch.float32)


    while t < tf:
        h = min(projhyb['time']['TAU'], tf - t)
        batch['h'] = h

        print(eventfun)
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
        
        if jac is not 0:
            k1_state, k1_jac = odesfun(ann, t, state, jac, None, w, ucontrol1, projhyb)
        else:
            k1_state = odesfun(ann,t, state, None, None, w, ucontrol1, projhyb)

        # FIX THIS 
        print("CONTROLFUN:", controlfun)
        print("Control", type(controlfun))
        control = None
        #ucontrol2 = controlfun(t + h / 2, batch) if controlfun is not None else []
        ucontrol2 = controlfun() if control is not None else []

        print("k1_state:", k1_state)
        print("k1_jac:", k1_jac)
        print("state:", state)
        print("h", h/2)
        h2 = h / 2
        h2 = torch.tensor(h2, dtype=torch.float64)
        k1_state = np.array(k1_state)


        k1_state = k1_state.astype(np.float32)
        k1_state = torch.from_numpy(k1_state)
        print("k1_state:type", k1_state.dtype)

        print("state", state)
        state = np.array(state)
        state = state.astype(np.float32)
        state = torch.from_numpy(state)

        
        print(state + h2 * k1_state)
        print("jac", jac)

        print("h2.dtype", h2.dtype)
        print("h2", h2)

        print("jac", jac)

        h2k1_jac = torch.mul(h2, k1_jac)

        print("h2k1_jac.size", h2k1_jac.size())
        print("jac.size", jac.size())
        jach2 = jac + h2k1_jac

        if jac is not 0:
            k2_state, k2_jac = odesfun(ann,t + h2, state + h2 * k1_state, jac + h2 * k1_jac, None, w, ucontrol2, projhyb)
            print("k1_state", k1_state)
            print("k1_jac", k1_jac)
            print("k2_state", k2_state)
            print("k2_jac", k2_jac)
            k2_state = np.array(k2_state)
            k2_state = k2_state.astype(np.float32)
            k2_state = torch.from_numpy(k2_state)

            k3_state, k3_jac = odesfun(ann,t + h2, state + h2 * k2_state, jac + h2 * k2_jac, None, w, ucontrol2, projhyb)
            k3_state = np.array(k3_state)
            k3_state = k3_state.astype(np.float32)
            k3_state = torch.from_numpy(k3_state)

        else:
            k2_state = odesfun(ann,t + h2, state + h2 * k1_state, None, None, w, ucontrol2, projhyb)
            k3_state = odesfun(ann,t + h2, state + h2 * k2_state, None, None, w, ucontrol2, projhyb)

        hl = h - h / 1e10
        ucontrol4 = controlfun(t + hl, batch) if controlfun is not None else []

        if jac is not None:
            k4_state, k4_jac = odesfun(ann,t + hl, state + hl * k3_state, jac + hl * k3_jac, None, w, ucontrol4, projhyb)
            k4_state = np.array(k4_state)
            k4_state = k4_state.astype(np.float32)
            k4_state = torch.from_numpy(k4_state)
        else:
            k4_state = odesfun(ann,t + hl, state + hl * k3_state, None, None, w, ucontrol4, projhyb)

        state = state + h * (k1_state / 6 + k2_state / 3 + k3_state / 3 + k4_state / 6)

        if jac is not None:
            jac = jac + h * (k1_jac / 6 + k2_jac / 3 + k3_jac / 3 + k4_jac / 6)

        t = t + h

    return t, state, jac, hess