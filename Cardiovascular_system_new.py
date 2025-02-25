import numpy as np
import math

from Activation_Functions import activation_U, activation_H, activation_Naghavi, g_function


def frac(x):
    return x - math.floor(x)

def cardiovascular_system(t, state, params, heart_control_inputs, resp_control_inputs, updates, all_time, num_removed):
    """
    Pulmonary circulation state variables: VT_pa, VT_pp, VT_pv, Q_pa
    Cardiovascular system state variables: VT_la, VT_lv, VT_ra, VT_rv
    Systemic circulation state variables: VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, VT_ev, P_sp, V_sa, P_sa, Q_sa, VT_vc

    Other inputs from heart controller: Vu_ev, Vu_amv, Vu_rmv, Vu_sv, R_ep, R_amp, R_rmp, R_sp, R_bp, R_hp, HR, Emax_lv,
                                        Emax_rv, I, beta, U
                       resp controller: BF, TI, VT

    """

    # State variables
    (VT_pa, VT_pp, VT_pv, Q_pa,
     VT_la, VT_lv, VT_ra, VT_rv,
     VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, VT_ev, P_sp, V_sa, P_sa, Q_sa, VT_vc, beta) = state


    ## Muscle Pump
    A_im = params["A_im"]
    Tc = params["Tc"]
    T_im = params["T_im"]

    # alp ranges between 0 (corresponding to the beginning of muscle contraction) and 1
    alp = (t % Tc) / Tc

    if (Tc / T_im) >= alp >= 0:
        psi = np.sin(np.pi * (T_im / Tc) * alp)
    elif (Tc / T_im) <= alp <= 1:
        psi = 0

    P_im = A_im * psi

    # p_im is 0 in resting conditions
    P_im = 0



    ## Respiratory Pump

    # constant parameters
    g_abd = params["g_abd"]
    g_thor = params["g_thor"]
    P_abdmax_n = params["P_abdmax_n"]
    P_abdmin_n = params["P_abdmin_n"]
    P_thormax_n = params["P_thormax_n"]
    P_thormin_n = params["P_thormin_n"]
    VT_n = params["VT_n"]

    # inputs from respiratory controller outputs
    T_resp = 1/resp_control_inputs["BF"][-1]
    TI = resp_control_inputs["TI"][-1]
    VT = resp_control_inputs["VT"][-1]

    VT_change = VT - VT_n # units of L
    TE = T_resp - TI
    P_abdmax = P_abdmax_n + g_abd * VT_change
    P_thormax = P_thormax_n + g_thor * VT_change
    P_abdmin = P_abdmin_n + g_abd * VT_change
    P_thormin = P_thormin_n + g_thor * VT_change

    first = TI / T_resp
    second = (TI + TE) / T_resp
    third = (TI / 2) / T_resp
    S = (t % T_resp) / T_resp

    if 0 <= S < first:
        P_thor = P_thormax - (P_thormax - P_thormin) * (T_resp / TI) * S

    elif first <= S <= second:
        P_thor = P_thormax - (P_thormax - P_thormin) * ((TI + TE - T_resp * S) / TE)

    # elif second <= S <= 1:
    #     P_thor = P_thormax

    if 0 <= S < third:
        P_abd = P_abdmax - (P_abdmax - P_abdmin) * (T_resp / (TI / 2)) * S

    elif third <= S < first:
        P_abd = P_abdmin

    elif first <= S <= second:
        P_abd = P_abdmax - (P_abdmax - P_abdmin) * ((TI + TE - T_resp * S) / TE)

    # elif second <= S <= 1:
    #     P_abd = P_abdmax

    ## Pulmonary Circulation
    C_pa = params["C_pa"]
    C_pp = params["C_pp"]
    C_pv = params["C_pv"]
    L_pa = params["L_pa"]
    R_pa = params["R_pa"]
    R_pp = params["R_pp"]
    R_pv = params["R_pv"]
    Vu_pa = params["Vu_pa"]
    Vu_pp = params["Vu_pp"]
    Vu_pv = params["Vu_pv"]


    if VT_pa > Vu_pa:
        V_pa = VT_pa - Vu_pa
    else:
        V_pa = 0

    P_pa = V_pa / C_pa + P_thor # 6-16mmHg

    if VT_pp > Vu_pp:
        V_pp = VT_pp - Vu_pp
    else:
        V_pp = 0

    P_pp = V_pp / C_pp

    if VT_pv > Vu_pv:
        V_pv = VT_pv - Vu_pv
    else:
        V_pv = 0

    P_pv = V_pv / C_pv



    ## The Heart

    # constant parameters
    C_la = params["C_la"]
    C_ra = params["C_ra"]
    KE_lv = params["KE_lv"]
    KE_rv = params["KE_rv"]
    KR_lv = params["KR_lv"]
    KR_rv = params["KR_rv"]
    ksys = params["ksys"]
    P0_lv = params["P0_lv"]
    P0_rv = params["P0_rv"]
    R_la = params["R_la"]
    R_ra = params["R_ra"]
    Tsys_0 = params["Tsys_0"]
    Vu_la = params["Vu_la"]
    Vu_lv = params["Vu_lv"]
    Vu_ra = params["Vu_ra"]
    Vu_rv = params["Vu_rv"]

    # inputs from the cardiovascular controller
    T = 1/heart_control_inputs["HR"][-1] # heart period
    Emax_lv = heart_control_inputs["Emax_lv"][-1]
    Emax_rv = heart_control_inputs["Emax_rv"][-1]
    I = heart_control_inputs["I"][-1]


    if VT_la > Vu_la:
        V_la = VT_la - Vu_la
    else:
        V_la = 0





    Emax_la = 0.45
    P0_la = 0.45
    KE_la = 0.05

    Emax_ra = 0.45
    P0_ra = 0.45
    KE_ra = 0.05

    # phi_atr = activation_U(beta, 1)
    phi_atr = activation_H(t, 1)
    # phi_atr = activation_Naghavi(t, 1)
    # phi_atr = g_function(t, 1)


    # P_la = (V_la / C_la) + P_thor
    Pmax_la = phi_atr * Emax_la * (VT_la - Vu_la) + (1 - phi_atr) * P0_la * (np.exp(KE_la * VT_la) - 1)
    ##############################################################




    # V_lv can be growing but there should not be any flow (Q) into the ventricles?
    if VT_lv > Vu_lv:
        V_lv = VT_lv - Vu_lv
    else:
        V_lv = 0

    # phi = activation_U(beta, 0)
    phi = activation_H(t, 0)
    # phi = activation_Naghavi(t, 0)
    # phi = g_function(t, 0)





    Pmax_lv = phi * Emax_lv * (VT_lv - Vu_lv) + (1 - phi) * P0_lv * (np.exp(KE_lv * VT_lv) - 1)

    # P_sa: aortic pressure 80-120 mmhg?
    if Pmax_lv > P_sa:
        Q_lv = (Pmax_lv - P_sa) / (KR_lv * Pmax_lv)
        P_lv = P_sa + P_thor
    else:
        Q_lv = 0
        P_lv = Pmax_lv + P_thor

    # P_lv = Pmax_lv - (KR_lv * Pmax_lv) * Q_lv


    ##################
    if Pmax_la > P_lv:
        Qi_lv = (Pmax_la - P_lv) / R_la
        P_la = P_lv + P_thor
    else:
        Qi_lv = 0
        P_la = Pmax_la + P_thor

    # P_la = Pmax_la - R_la * Qi_lv
    ####################




    if VT_ra > Vu_ra:
        V_ra = VT_ra - Vu_ra
    else:
        V_ra = 0





    ##################################
    # P_ra = (V_ra / C_ra) + P_thor
    Pmax_ra = phi_atr * Emax_ra * (VT_ra - Vu_ra) + (1 - phi_atr) * P0_ra * (np.exp(KE_ra * VT_ra) - 1)
    ##################################






    if VT_rv > Vu_rv:
        V_rv = VT_rv - Vu_rv
    else:
        V_rv = 0


    Pmax_rv = phi * Emax_rv * (VT_rv - Vu_rv) + (1 - phi) * P0_rv * (np.exp(KE_rv * VT_rv) - 1)

    if Pmax_rv > P_pa:
        Q_rv = (Pmax_rv - P_pa) / (KR_rv * Pmax_rv)
        P_rv = P_pa + P_thor
    else:
        Q_rv = 0
        P_rv = Pmax_rv + P_thor

    # P_rv = Pmax_rv - (KR_rv * Pmax_rv) * Q_rv

    ####################################
    if Pmax_ra > P_rv:
        Qi_rv = (Pmax_ra - P_rv) / R_ra
        P_ra = P_rv + P_thor
    else:
        Qi_rv = 0
        P_ra = Pmax_ra + P_thor

    # P_ra = Pmax_ra - R_ra * Qi_rv
    ####################################


    # P_pv and P_sa don't seem to stabilise
    ####### moved from the pulmonary circulation system here
    Q_la = (P_pv + P_thor - P_la) / R_pv
    Q_pp = (P_pp - P_pv) / R_pp

    dVT_pa_dt = Q_rv - Q_pa
    dVT_pp_dt = Q_pa - Q_pp
    dVT_pv_dt = Q_pp - Q_la
    dQ_pa_dt = (P_pa - R_pa * Q_pa - P_pp) / L_pa

    #######

    dVT_lv_dt = Qi_lv - Q_lv
    dVT_la_dt = Q_la - Qi_lv

    if VT_lv > Vu_lv:
        dV_lv_dt = dVT_lv_dt   # Added this myself
    else:
        dV_lv_dt = 0           # Added this myself

    Wh_lv = (P_thor - P_lv) * dV_lv_dt



    ## systemic arteries
    C_sa = params["C_sa"]
    L_sa = params["L_sa"]
    R_sa = params["R_sa"]
    Vu_sa = params["Vu_sa"]


    ## vena cava circulation
    D1 = params["D1"]
    D2 = params["D2"]
    K1_vc = params["K1_vc"]
    K2_vc = params["K2_vc"]
    Kr_vc = params["Kr_vc"]
    Rvc_n = params["Rvc_n"]
    Vu_vc = params["Vu_vc"]
    Vvc_max = params["Vvc_max"] # highest at end diastole
    Vvc_min = params["Vvc_min"]


    if VT_vc > Vu_vc:
        V_vc = VT_vc - Vu_vc
    else:
        V_vc = 0

    if V_vc < Vu_vc:
        P_vc = D2 + K2_vc * np.exp(V_vc / Vvc_min) + P_thor
    else:
        P_vc = D1 + K1_vc * (V_vc - Vu_vc) + P_thor

    R_vc = Kr_vc * (Vvc_max / V_vc) ** 2 + Rvc_n

    # removed Q_ra if statement so that back flow is possible into the vena cava
    Q_ra = (P_vc - P_ra) / R_vc

    dVT_rv_dt = Qi_rv - Q_rv
    dVT_ra_dt = Q_ra - Qi_rv

    if VT_rv > Vu_rv:
        dV_rv_dt = dVT_rv_dt  # Added this myself
    else:
        dV_rv_dt = 0  # Added this myself

    Wh_rv = (P_thor - P_rv) * dV_rv_dt



    ## systemic peripheral and venous circulation
    C_ep = params["C_ep"]
    C_sp = params["C_sp"]
    C_bp = params["C_bp"]
    C_hp = params["C_hp"]
    C_rmp = params["C_rmp"]
    C_amp = params["C_amp"]
    V_tot = params["V_tot"]
    R_ev_n = params["R_ev_n"]
    R_sv_n = params["R_sv_n"]
    R_bv_n = params["R_bv_n"]
    R_hv_n = params["R_hv_n"]
    R_rmv_n = params["R_rmv_n"]
    R_amv_n = params["R_amv_n"]
    C_ev = params["C_ev"]
    C_sv = params["C_sv"]
    C_bv = params["C_bv"]
    C_hv = params["C_hv"]
    C_rmv = params["C_rmv"]
    C_amv = params["C_amv"]
    Vu_ep = params["Vu_ep"]
    Vu_sp = params["Vu_sp"]
    Vu_bp = params["Vu_bp"]
    Vu_hp = params["Vu_hp"]
    Vu_rmp = params["Vu_rmp"]
    Vu_amp = params["Vu_amp"]
    kr_am = params["kr_am"]
    P_0 = params["P_0"]
    Vu_bv = params["Vu_bv"]
    Vu_hv = params["Vu_hv"]

    # input from other systems
    Vu_ev = heart_control_inputs["Vu_ev"][-1]
    Vu_amv = heart_control_inputs["Vu_amv"][-1]
    Vu_rmv = heart_control_inputs["Vu_rmv"][-1]
    Vu_sv = heart_control_inputs["Vu_sv"][-1]
    R_ep = heart_control_inputs["R_ep"][-1]
    R_amp = heart_control_inputs["R_amp"][-1]
    R_rmp = heart_control_inputs["R_rmp"][-1]
    R_sp = heart_control_inputs["R_sp"][-1]
    R_bp = heart_control_inputs["R_bp"][-1]
    R_hp = heart_control_inputs["R_hp"][-1]

    # splanchnic
    V_sp = C_sp * P_sp

    if VT_sv >= Vu_sv:
        V_sv = VT_sv - Vu_sv
        P_sv = V_sv/C_sv
    else:
        V_sv = 0
        P_sv = 0

    Q_sp = (P_sp - P_sv) / R_sp

    P_s = P_abd

    if P_vc < P_s:
        R_sv = R_sv_n * ((P_sv - P_vc) / (P_sv - P_s))
    else:
        R_sv = R_sv_n

    if P_sv >= P_vc:
        Q_sv = (P_sv - P_vc) / R_sv
    else:
        Q_sv = 0

    dVT_sv_dt = Q_sp - Q_sv

    # brain
    V_bp = C_bp * P_sp

    if VT_bv >= Vu_bv:
        V_bv = VT_bv - Vu_bv
        P_bv = V_bv / C_bv
    else:
        V_bv = 0
        P_bv = 0

    Q_bp = (P_sp - P_bv) / R_bp

    P_b = 0

    if P_vc < P_b:
        R_bv = R_bv_n * ((P_bv - P_vc) / (P_bv - P_b))
    else:
        R_bv = R_bv_n

    if P_bv >= P_vc:
        Q_bv = (P_bv - P_vc) / R_bv
    else:
        Q_bv = 0

    dVT_bv_dt = Q_bp - Q_bv

    # coronary circulation
    V_hp = C_hp * P_sp


    if VT_hv >= Vu_hv:
        V_hv = VT_hv - Vu_hv
        P_hv = V_hv / C_hv
    else:
        V_hv = 0
        P_hv = 0


    Q_hp = (P_sp - P_hv) / R_hp


    P_h = 0

    if P_vc < P_h:
        R_hv = R_hv_n * ((P_hv - P_vc) / (P_hv - P_h))
    else:
        R_hv = R_hv_n

    if P_hv >= P_vc:
        Q_hv = (P_hv - P_vc) / R_hv
    else:
        Q_hv = 0

    dVT_hv_dt = Q_hp - Q_hv

    # resting muscle
    V_rmp = C_rmp * P_sp

    if VT_rmv >= Vu_rmv:
        V_rmv = VT_rmv - Vu_rmv
        P_rmv = V_rmv / C_rmv
    else:
        V_rmv = 0
        P_rmv = 0


    Q_rmp = (P_sp - P_rmv) / R_rmp

    P_rm = 0

    if P_vc < P_rm:
        R_rmv = R_rmv_n * ((P_rmv - P_vc) / (P_rmv - P_rm))
    else:
        R_rmv = R_rmv_n

    if P_rmv >= P_vc:
        Q_rmv = (P_rmv - P_vc) / R_rmv
    else:
        Q_rmv = 0

    dVT_rmv_dt = Q_rmp - Q_rmv



    # active muscle
    V_amp = C_amp * P_sp

    if VT_amv >= Vu_amv:
        V_amv = VT_amv - Vu_amv
        P_amv = V_amv / C_amv + P_im
    else:
        V_amv = 0
        P_amv = P_im + P_0 * (1 - (VT_amv / Vu_amv) ** -1.5)
        # P_amv = P_0 + P_im

    Q_amp = (P_sp - P_amv) / R_amp


    P_am = 0

    if I > 0:
        R_amv = kr_am / VT_amv
    elif P_vc < P_am:
        R_amv = R_amv_n * ((P_amv - P_vc) / (P_amv - P_am))
    else:
        R_amv = R_amv_n


    if P_amv >= P_vc:
        Q_amv = (P_amv - P_vc) / R_amv
    else:
        Q_amv = 0

    dVT_amv_dt = Q_amp - Q_amv

    if P_amv < 0:
        a = 2

    # extrasplanchnic
    V_ep = C_ep * P_sp

    C_jp = C_ep + C_sp + C_bp + C_hp + C_rmp + C_amp
    Vu_jp = Vu_ep + Vu_sp + Vu_bp + Vu_hp + Vu_rmp + Vu_amp
    Vu_jv = Vu_ev + Vu_sv + Vu_bv + Vu_hv + Vu_rmv + Vu_amv

    V_u = Vu_sa + Vu_pa + Vu_pp + Vu_pv + Vu_ra + Vu_la + Vu_jp + Vu_jv

    # added myself
    check = V_ep + V_amp + V_bp + V_hp+ V_rmp + V_sp + V_amv

    left_over_volume = (V_tot - V_sa - V_ra - V_rv - V_la - V_lv - V_pa - V_pp - V_pv - V_sv - V_rmv - V_amv - V_bv
            - V_hv - V_vc - V_u - P_sp * C_jp)

    # if left_over_volume < 0:
    #     raise ValueError("Error: wrong")

    P_ev = left_over_volume / C_ev

    Q_ep = (P_sp - P_ev) / R_ep

    # if VT_ev >= Vu_ev:
    #     V_ev = VT_ev - Vu_ev
    #     P_ev = V_ev/C_ev
    # else:
    #     V_ev = 0
    #     P_ev = 0


    P_e = 0

    if P_vc < P_e:
        R_ev = R_ev_n * ((P_ev - P_vc) / (P_ev - P_e))
    else:
        R_ev = R_ev_n


    if P_ev >= P_vc:
        Q_ev = (P_ev - P_vc) / R_ev
    else:
        Q_ev = 0

    Q_vc = Q_ev + Q_sv + Q_bv + Q_hv + Q_rmv + Q_amv
    Q_jp = Q_ep + Q_sp + Q_bp + Q_hp + Q_rmp + Q_amp

    dP_sa_dt = (Q_lv - Q_sa) / C_sa
    dVT_ev_dt = Q_ep - Q_ev
    dVT_vc_dt = Q_vc - Q_ra
    dP_sp_dt = (Q_sa - Q_jp) / C_jp
    dV_sa_dt = dP_sa_dt * C_sa
    # VT_sa = V_sa + Vu_sa
    dQ_sa_dt = ((P_sa - P_thor) - R_sa * Q_sa - P_sp) / L_sa

    d_beta_dt = heart_control_inputs["HR"][-1]


    if t != 0:
        if t < all_time[-1]:
            for key in [
                "Q_pp", "Q_bp", "Q_hp", "Q_rmp", "Q_amp", "Q_la", "Q_lv", "Q_ra", "Q_rv",
                "Wh_lv", "Wh_rv", "U", "dP_sa_dt", "P_sa", "P_ra", "P_la", "P_lv", "P_rv",
                "Pmax_lv", "Pmax_rv", "V_rv", "V_ra", "V_lv", "V_la", "VT_rv", "VT_ra",
                "VT_lv", "VT_la", "P_pa", "P_pp", "P_pv", "P_thor", "V_vc", "P_vc",
                "Qi_lv", "Qi_rv", "phi", "S", "V_pv", "V_pp", "V_pa",  "P_amv", "P_ev", "V_u", "V_sv", "V_rmv", "V_amv", "V_bv",
                "V_hv", "P_sp", "Q_sa", "Q_jp", "Q_vc", "VT_amv", "P_im", "Q_amv", "Q_sp", "Q_pa", "phi_atr", "P_abd", "beta", "Q_ep", "Pmax_la", "Pmax_ra"
            ]:
                del updates[key][-num_removed:]

    # t_eval = updates["t_eval1"][0]
    # tolerance = 1e-3
    # if t > Next_Conditions["time_history"][-1]:
    data_to_append = {
        "Q_pp": Q_pp, "Q_bp": Q_bp, "Q_hp": Q_hp, "Q_rmp": Q_rmp, "Q_amp": Q_amp,
        "Q_la": Q_la, "Q_lv": Q_lv, "Q_ra": Q_ra, "Q_rv": Q_rv, "Wh_lv": Wh_lv, "Wh_rv": Wh_rv,
        "dP_sa_dt": dP_sa_dt, "P_sa": P_sa, "P_ra": P_ra, "P_la": P_la, "P_lv": P_lv,
        "P_rv": P_rv, "Pmax_lv": Pmax_lv, "Pmax_rv": Pmax_rv, "Pmax_la": Pmax_la, "Pmax_ra": Pmax_ra,
        "V_rv": V_rv, "V_ra": V_ra, "V_lv": V_lv, "V_la": V_la, "VT_rv": VT_rv, "VT_ra": VT_ra,
        "VT_lv": VT_lv, "VT_la": VT_la, "P_pa": P_pa, "P_pp": P_pp, "P_pv": P_pv, "P_thor": P_thor,
        "V_vc": V_vc, "P_vc": P_vc, "Qi_lv": Qi_lv, "Qi_rv": Qi_rv, "V_pa": V_pa, "phi": phi,
        "phi_atr": phi_atr, "S": S, "V_pv": V_pv, "V_pp": V_pp, "P_amv": P_amv, "P_ev": P_ev,
        "V_u": V_u, "V_sv": V_sv, "V_rmv": V_rmv, "V_amv": V_amv, "V_bv": V_bv, "V_hv": V_hv,
        "P_sp": P_sp, "Q_sa": Q_sa, "Q_jp": Q_jp, "Q_vc": Q_vc, "VT_amv": VT_amv, "P_im": P_im,
        "Q_amv": Q_amv, "Q_sp": Q_sp, "Q_ep": Q_ep, "Q_pa": Q_pa, "P_abd": P_abd, "beta": beta
    }

    [updates[key].append(value) for key, value in data_to_append.items()]


    return [dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
            dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dVT_ev_dt, dP_sp_dt, dV_sa_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt, d_beta_dt]