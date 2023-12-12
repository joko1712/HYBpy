def odesfun(ann, t, state, jac, hess, w, ucontrol, projhyb):

    ann.load_state_dict({"weights": w})

    if jac is None and hess is None:
        anninp = anninp_func()
        rann = rann_func()
        fstate = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol) # FALTA

        return fstate, None, None

    else:

        if projhyb['mode'] == 1:
            anninp = anninp_func()
            DanninpDstate = Matrix([anninp]).jacobian(Matrix([state]))

            rann = rann_func()
            _, DrannDanninp, DrannDw = ann.backpropagate(anninp)

            fstate = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol) # FALTA

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            DrannDs = DrannDanninp * DanninpDstate

            fjac = (DfDs + DfDrann * DrannDs) * jac + DfDrann * DrannDw

            return fstate, fjac, None

        elif projhyb['mode'] == 3:
            anninp = anninp_func()

            rann = rann_func()

            fstate = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol)

            DfDs = Matrix([fstate]).jacobian(Matrix([state]))
            DfDrann = Matrix([fstate]).jacobian(Matrix([rann]))

            fjac = DfDs * jac + DfDrann

            return fstate, fjac, None

    return None, None, None

     
def anninp_func():
    anninp = []

    for i in range(1,  data["mlm"]["nx"]+1):
        totalsyms.append(data["mlm"]["x"][str(i)]["id"])

        val = sp.sympify(data["mlm"]["x"][str(i)]["val"])
        max = sp.sympify(data["mlm"]["x"][str(i)]["max"])

        anninp.append(val/max)

    for i in range(1, len(anninp)+1):
        anninp.append(anninp[i-1]/data["mlm"]["x"][str(i)]["max"])

    anninp_tensor = torch.tensor(anninp, dtype=torch.float32)
    return anninp_tensor


def rann_func():
    rann = []

    for i in range(1, data["mlm"]["ny"]+1):
        rann.append(sp.sympify(data["mlm"]["y"][str(i)]["id"]))

    return rann


#TODO: ASK Where fstate = projhyb['userdefun_parametric_odes'](t, state, rann, ucontrol) is defined