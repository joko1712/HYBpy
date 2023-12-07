def odesfun(t, state, jac, hess, w, ucontrol, projhyb):

    if jac is None and hess is None:
        anninp = projhyb['mlm']['xfun'](t, state, ucontrol)
        rann = projhyb['mlm']['yfun'](anninp, w, projhyb['mlm']['fundata'])
        fstate = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol)

        return fstate, None, None

    else:

        if projhyb['mode'] == 1:
            anninp, DanninpDstate = projhyb['mlm']['xfun'](t, state, ucontrol)
            rann, DrannDanninp, DrannDw = projhyb['mlm']['yfun'](anninp, w, projhyb['mlm']['fundata'])
            fstate, DfDs, DfDrann = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol)
            DrannDs = DrannDanninp * DanninpDstate
            fjac = (DfDs + DfDrann * DrannDs) * jac + DfDrann * DrannDw

            return fstate, fjac, None

        elif projhyb['mode'] == 3:
            anninp = projhyb['mlm']['xfun'](t, state, ucontrol)
            rann = projhyb['mlm']['yfun'](anninp, w, projhyb['mlm']['fundata'])
            fstate, DfDs, DfDrann = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol)
            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None
