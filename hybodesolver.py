import numpy as np

def hybodesolver(odesfun, controlfun, eventfun, t0, tf, state, jac, hess, w, batch, projhyb):
    t = t0
    hopt = []

    while t < tf:
        h = min(projhyb['time']['TAU'], tf - t)
        batch['h'] = h

        if eventfun is not []:
            if jac is not 0:
                batch, state, dstatedstate = eventfun(t, batch, state)
                jac = dstatedstate * jac
            else:
                batch, state = eventfun(t, batch, state)

        ucontrol1 = controlfun(t, batch) if controlfun is not None else []

        if jac is not 0:
            k1_state, k1_jac = odesfun(t, state, jac, None, w, ucontrol1, projhyb)
        else:
            k1_state = odesfun(t, state, None, None, w, ucontrol1, projhyb)

        ucontrol2 = controlfun(t + h / 2, batch) if controlfun is not None else []

        if jac is not 0:
            k2_state, k2_jac = odesfun(t + h / 2, state + h / 2 * k1_state, jac + h / 2 * k1_jac, None, w, ucontrol2, projhyb)
            k3_state, k3_jac = odesfun(t + h / 2, state + h / 2 * k2_state, jac + h / 2 * k2_jac, None, w, ucontrol2, projhyb)
        else:
            k2_state = odesfun(t + h / 2, state + h / 2 * k1_state, None, None, w, ucontrol2, projhyb)
            k3_state = odesfun(t + h / 2, state + h / 2 * k2_state, None, None, w, ucontrol2, projhyb)

        hl = h - h / 1e10
        ucontrol4 = controlfun(t + hl, batch) if controlfun is not None else []

        if jac is not None:
            k4_state, k4_jac = odesfun(t + hl, state + hl * k3_state, jac + hl * k3_jac, None, w, ucontrol4, projhyb)
        else:
            k4_state = odesfun(t + hl, state + hl * k3_state, None, None, w, ucontrol4, projhyb)

        state = state + h * (k1_state / 6 + k2_state / 3 + k3_state / 3 + k4_state / 6)

        if jac is not None:
            jac = jac + h * (k1_jac / 6 + k2_jac / 3 + k3_jac / 3 + k4_jac / 6)

        t = t + h

    return t, state, jac, hess