import numpy as np
import math
from Parameters import Parameters as params

from Test_controller import source

from Activation_Functions import activation_U, activation_H, activation_Naghavi, g_function, activation_H_derivative, \
    activation_conduit, activation_S


def frac(x):
    return x - math.floor(x)

def cardiovascular_system(t, state, params, heart_control_inputs, resp_control_inputs, updates, num_removed, t_start, time_saved, i, BUFFER_LIMIT):
    """
    Pulmonary circulation state variables: VT_pa, VT_pp, VT_pv, Q_pa
    Cardiovascular system state variables: VT_la, VT_lv, VT_ra, VT_rv
    Systemic circulation state variables: VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, VT_ev, P_sp, P_sa, Q_sa, VT_vc

    Other inputs from heart controller: Vu_ev, Vu_amv, Vu_rmv, Vu_sv, R_ep, R_amp, R_rmp, R_sp, R_bp, R_hp, HR, Emax_lv,
                                        Emax_rv, I, beta, U
                       resp controller: BF, TI, VT

    """

    # State variables
    (VT_pa, VT_pp, VT_pv, Q_pa,
     VT_la, VT_lv, VT_ra, VT_rv,
     VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, VT_ev, P_sp, P_sa, Q_sa, VT_vc) = state


    (A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa, R_pa,
     R_pp, R_pv, Vu_pa, Vu_pp, Vu_pv, KE_lv, KE_rv, P0_lv, P0_rv, Vu_la, Vu_lv, Vu_ra, Vu_rv, Emax_la, P0_la, KE_la, Emax_ra,
     P0_ra, KE_ra, C_sa, L_sa, R_sa, Vu_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n, Vu_vc, Vvc_max, Vvc_min, C_ep, C_sp, C_bp,
     C_hp, C_rmp, C_amp, V_tot, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv, Vu_ep,
     Vu_sp, Vu_bp, Vu_hp, Vu_rmp, Vu_amp, kr_am, Vu_bv, Vu_hv) = (params[k] for k in ["A_im", "Tc", "T_im", "g_abd", "g_thor",
    "P_abdmax_n", "P_abdmin_n", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv",
    "Vu_pa", "Vu_pp", "Vu_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "Emax_la", "P0_la",
    "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa", "R_sa", "Vu_sa", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n",
    "Vu_vc", "Vvc_max", "Vvc_min", "C_ep", "C_sp", "C_bp", "C_hp", "C_rmp", "C_amp", "V_tot", "R_ev_n", "R_sv_n", "R_bv_n",
    "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv", "Vu_ep", "Vu_sp", "Vu_bp", "Vu_hp",
    "Vu_rmp", "Vu_amp", "kr_am", "Vu_bv", "Vu_hv"])

    # Determine the correct index based on t
    if t == t_start:
        heart_control_index = i % BUFFER_LIMIT
        time_since_beat = updates["time_since_beat_store"][i % BUFFER_LIMIT]
        resp_control_index = i % BUFFER_LIMIT
    else:
        heart_control_index = (i - num_removed - 1) % BUFFER_LIMIT
        resp_control_index = (i - num_removed - 1) % BUFFER_LIMIT
        time_since_beat = updates["time_since_beat_store"][(i - num_removed - 1) % BUFFER_LIMIT]

    # Muscle Pump
    # alp ranges between 0 (corresponding to the beginning of muscle contraction) and 1
    alp = (t % Tc) / Tc

    if (Tc / T_im) >= alp >= 0:
        psi = np.sin(np.pi * (T_im / Tc) * alp)
    elif (Tc / T_im) <= alp <= 1:
        psi = 0

    P_im = A_im * psi

    # p_im is 0 in resting conditions
    # P_im = 0



    ## Respiratory Pump
    # if t > 1000:
    #     P_thormax_n = -4
    #     P_thormin_n = -9

    # respiratory controller inputs
    T_resp = 1 / resp_control_inputs["BF_store"][resp_control_index]
    TI = resp_control_inputs["TI_store"][resp_control_index]
    VT = resp_control_inputs["VT_store"][resp_control_index]

    # inputs from the cardiovascular controller
    T = 1/heart_control_inputs["HR_store"][heart_control_index] # heart period
    Vu_ev = heart_control_inputs["Vu_ev_store"][heart_control_index]
    Vu_sv = heart_control_inputs["Vu_sv_store"][heart_control_index]
    Vu_rmv = heart_control_inputs["Vu_rmv_store"][heart_control_index]
    Vu_amv = heart_control_inputs["Vu_amv_store"][heart_control_index]
    Emax_lv = heart_control_inputs["Emax_lv_store"][heart_control_index]
    Emax_rv = heart_control_inputs["Emax_rv_store"][heart_control_index]

    R_ep = heart_control_inputs["R_ep_store"][heart_control_index]
    R_amp = heart_control_inputs["R_amp_store"][heart_control_index]
    R_rmp = heart_control_inputs["R_rmp_store"][heart_control_index]
    R_sp = heart_control_inputs["R_sp_store"][heart_control_index]
    R_bp = heart_control_inputs["R_bp_store"][heart_control_index]
    R_hp = heart_control_inputs["R_hp_store"][heart_control_index]
    I = heart_control_inputs["I_store"][heart_control_index]


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

    # added P_thor to only the pulmonary compartments
    if VT_pa > Vu_pa:
        V_pa = VT_pa - Vu_pa
    else:
        V_pa = 0

    P_pa = V_pa / C_pa + P_thor # 6-16mmHg

    if VT_pp > Vu_pp:
        V_pp = VT_pp - Vu_pp
    else:
        V_pp = 0

    P_pp = V_pp / C_pp + P_thor

    if VT_pv > Vu_pv:
        V_pv = VT_pv - Vu_pv
    else:
        V_pv = 0

    P_pv = V_pv / C_pv + P_thor



    ## The Heart

    if VT_la > Vu_la:
        V_la = VT_la - Vu_la
    else:
        V_la = 0

    # phi_atr = activation_U(beta, 1)
    phi_atr = activation_H(t - time_since_beat, 1, T)
    # phi_atr = activation_Naghavi(t, 1)
    # phi_atr = g_function(t, 1)
    # phi_cond = activation_conduit(t, T)

    # P_la = (V_la / C_la) + P_thor
    Pmax_la = phi_atr * Emax_la * (VT_la - Vu_la) + (1 - phi_atr) * P0_la * (np.exp(KE_la * VT_la) - 1) + P_thor
    ##############################################################

    if t > 1062.2:
        A = 2
    # V_lv can be growing but there should not be any flow (Q) into the ventricles?
    if VT_lv > Vu_lv:
        V_lv = VT_lv - Vu_lv
    else:
        V_lv = 0

    # phi = activation_U(beta, 0)
    phi = activation_H(t - time_since_beat, 0, T)
    # phi = activation_Naghavi(t, 0)
    # phi = g_function(t, 0)


    Pmax_lv = phi * Emax_lv * (VT_lv - Vu_lv) + (1 - phi) * P0_lv * (np.exp(KE_lv * VT_lv) - 1) + P_thor

    # if Q_lv > 0:
    #     d2theta_ao_dt2 = (Pmax_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt + Kb_ao * Q_lv * np.cos(theta_ao) - Kv_ao * Q_lv * np.sin(theta_ao)
    # else:
    #     theta_ao = 0.0872665 # theta_ao_min
    #     d2theta_ao_dt2 = 0
    #     # d2theta_ao_dt2 = (Pmax_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt + Kb_ao * Q_lv * np.cos(theta_ao)


    # aortic valve
    # parameters:
    # Kp_ao = 55
    # Kf_ao = 50
    # Kb_ao = 2
    # Kv_ao = 7
    # theta_ao_max = 1.309 # 75 degrees to radian

    # if t == 0:
    #     Q_lv = 0
    # else:
    #     Q_lv = updates["Q_lv"][-1]


    if Pmax_lv > P_sa:
        # if theta_ao > theta_ao_max:
        #     theta_ao = theta_ao_max
        # d2theta_ao_dt2 = (Pmax_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt + Kb_ao * Q_lv * np.cos(theta_ao) - Kv_ao * Q_lv * np.sin(theta_ao)
        # d2theta_ao_dt2 = 0
        # AR_ao = ((1 - np.cos(theta_ao)) ** 2) / ((1 - np.cos(theta_ao_max)) ** 2)
        AR_ao = 1

        Q_lv = (math.sqrt(Pmax_lv - P_sa) * AR_ao * 350)
        P_lv = Pmax_lv
    else:
        Q_lv = 0.0
        # theta_ao = 0.0872665  # theta_ao_min
        # d2theta_ao_dt2 = 0.0
        P_lv = Pmax_lv

        # AR_ao = ((1 - np.cos(theta_ao)) ** 2) / ((1 - np.cos(theta_ao_max)) ** 2)
        AR_ao = 1

    # P_lv = Pmax_lv - (KR_lv * Pmax_lv) * Q_lv


    ##################
    # P_lv already considers P_thor, no need to add again
    if Pmax_la > P_lv:
        Qi_lv = math.sqrt(Pmax_la - P_lv) * 350
        P_la = Pmax_la
    else:
        Qi_lv = 0
        P_la = Pmax_la

    # P_la = Pmax_la - R_la * Qi_lv
    ####################




    if VT_ra > Vu_ra:
        V_ra = VT_ra - Vu_ra
    else:
        V_ra = 0





    ##################################
    # P_ra = (V_ra / C_ra) + P_thor
    Pmax_ra = phi_atr * Emax_ra * (VT_ra - Vu_ra) + (1 - phi_atr) * P0_ra * (np.exp(KE_ra * VT_ra) - 1) + P_thor
    ##################################






    if VT_rv > Vu_rv:
        V_rv = VT_rv - Vu_rv
    else:
        V_rv = 0


    Pmax_rv = phi * Emax_rv * (VT_rv - Vu_rv) + (1 - phi) * P0_rv * (np.exp(KE_rv * VT_rv) - 1) + P_thor

    # P_pa already considers P_thor, no need to add again
    if Pmax_rv > P_pa:
        Q_rv = (math.sqrt(Pmax_rv - P_pa) * 350)
        P_rv = Pmax_rv
    else:
        Q_rv = 0
        P_rv = Pmax_rv

    # P_rv = Pmax_rv - (KR_rv * Pmax_rv) * Q_rv

    ####################################
    # P_rv already considers P_thor, no need to add again
    if Pmax_ra > P_rv:
        Qi_rv = math.sqrt(Pmax_ra - P_rv) * 350
        P_ra = Pmax_ra
    else:
        Qi_rv = 0
        P_ra = Pmax_ra

    # P_ra = Pmax_ra - R_ra * Qi_rv
    ####################################


    # P_pv and P_sa don't seem to stabilise
    ####### moved from the pulmonary circulation system here
    Q_la = (P_pv - P_la) / R_pv
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
        dV_lv_dt = 0.0           # Added this myself

    Wh_lv = (P_thor - P_lv) * dV_lv_dt


    if VT_vc > Vu_vc:
        V_vc = VT_vc - Vu_vc
    else:
        V_vc = 0

    # if t!=0:
    #     source_values = updates["source_values"][-1] + source(t) * (t - updates["time_history"][-1])
    #     A = source_values
    # else:
    #     source_values = 0
    #     A = 0

    # source_values = source(t)


    if V_vc < Vu_vc:
        if t != 0:
            P_vc = D2 + K2_vc * np.exp(V_vc / Vvc_min) + P_thor # + source_values
        else:
            P_vc = D2 + K2_vc * np.exp(V_vc / Vvc_min) + P_thor
    else:
        if t != 0:
            P_vc = D1 + K1_vc * (V_vc - Vu_vc) + P_thor # + source_values
        else:
            P_vc = D1 + K1_vc * (V_vc - Vu_vc) + P_thor


    if V_vc > 0:
        R_vc = Kr_vc * (Vvc_max / V_vc) ** 2 + Rvc_n
    else:
        R_vc = Rvc_n

    # removed Q_ra if statement so that back flow is possible into the vena cava
    Q_ra = (P_vc - P_ra) / R_vc

    dVT_rv_dt = Qi_rv - Q_rv
    dVT_ra_dt = Q_ra - Qi_rv

    if VT_rv > Vu_rv:
        dV_rv_dt = dVT_rv_dt  # Added this myself
    else:
        dV_rv_dt = 0.0  # Added this myself

    Wh_rv = (P_thor - P_rv) * dV_rv_dt



    ## systemic peripheral and venous circulation

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

    if t > 24.3:
        A = 2

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
    epsilon1 = 1e-6

    if P_vc < P_b:
        R_bv = R_bv_n * ((P_bv - P_vc) / (P_bv - P_b + epsilon1))
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
        R_hv = R_hv_n * ((P_hv - P_vc) / (P_hv - P_h + epsilon1))
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
        R_rmv = R_rmv_n * ((P_rmv - P_vc) / (P_rmv - P_rm + epsilon1))
    else:
        R_rmv = R_rmv_n

    if P_rmv >= P_vc:
        Q_rmv = (P_rmv - P_vc) / R_rmv
    else:
        Q_rmv = 0

    dVT_rmv_dt = Q_rmp - Q_rmv



    # active muscle
    V_amp = C_amp * P_sp

    P_0 = Vu_amv/ (C_amv * 10)

    if VT_amv >= Vu_amv:
        V_amv = VT_amv - Vu_amv
        P_amv = V_amv / C_amv + P_im
    else:
        V_amv = 0
        if VT_amv > 0:
            P_amv = P_im + P_0 * (1 - (VT_amv / Vu_amv) ** -1.5)
        else:
            P_amv = P_im + P_0
            VT_amv = 0
        # P_amv = P_0 + P_im

    Q_amp = (P_sp - P_amv) / R_amp


    P_am = 0

    if I > 0:
        R_amv = kr_am / VT_amv
    elif P_vc < P_am:
        R_amv = R_amv_n * ((P_amv - P_vc) / (P_amv - P_am + epsilon1))
    else:
        R_amv = R_amv_n


    if P_amv >= P_vc:
        Q_amv = (P_amv - P_vc) / R_amv
    else:
        Q_amv = 0

    dVT_amv_dt = Q_amp - Q_amv

    ## systemic peripheral and venous circulation
    # extrasplanchnic
    V_ep = C_ep * P_sp

    C_jp = C_ep + C_sp + C_bp + C_hp + C_rmp + C_amp
    Vu_jp = Vu_ep + Vu_sp + Vu_bp + Vu_hp + Vu_rmp + Vu_amp
    Vu_jv = Vu_ev + Vu_sv + Vu_bv + Vu_hv + Vu_rmv + Vu_amv

    V_u = Vu_sa + Vu_pa + Vu_pp + Vu_pv + Vu_ra + Vu_la + Vu_jp + Vu_jv

    V_sa = P_sa * C_sa
    multiplied = P_sp * C_jp

    left_over_volume = (V_tot - V_sa - V_ra - V_rv - V_la - V_lv - V_pa - V_pp - V_pv - V_sv - V_rmv - V_amv - V_bv
            - V_hv - V_vc - V_u - multiplied)

    # if left_over_volume < 0:
        # raise ValueError("Error: wrong")

    P_ev = left_over_volume / C_ev # + source_values

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
    dQ_sa_dt = (P_sa - P_thor - R_sa * Q_sa - P_sp) / L_sa
    # VT_sa = V_sa + Vu_sa
    # should be + ?, edit: removed P_thor from here. Ignore



    # if num_removed > 0:
        # keys = [
        #     "P_sa", "dP_sa_dt", "Q_bp", "Q_hp", "Q_rmp", "Q_amp", "Wh_lv", "Wh_rv", "time_since_beat", "Q_pp", "Q_la",
        #     # "Q_lv", "Q_ra", "Q_rv",
        #     # "P_ra", "P_la", "P_lv", "P_rv",
        #     # "Pmax_lv", "Pmax_rv", "Pmax_la", "Pmax_ra", "V_rv", "V_ra", "V_lv", "V_la",
        #     # "VT_rv", "VT_ra", "VT_lv", "VT_la", "P_pa", "P_pp", "P_pv", "P_thor",
        #     # "V_vc", "P_vc", "Qi_lv", "Qi_rv", "phi", "S", "V_pv", "V_pp", "V_pa",
        #     # "P_amv", "P_ev", "V_u", "V_sv", "V_rmv", "V_amv", "V_bv", "V_hv", "P_sp",
        #     # "Q_sa", "Q_jp", "Q_vc", "VT_amv", "P_im", "Q_amv", "Q_sp", "Q_pa",
        #     # "phi_atr", "P_abd", "Q_ep", "Pmax_la", "Pmax_ra", "V_sa", "P_bv", "R_bv",
        #     # "VT_ev", "Q_ev", "VT_pa", "VT_pp", "VT_pv",
        #     # "VT_sv", "VT_bv", "VT_hv", "VT_rmv", "VT_vc", "Q_bv"
        # ]
        # for key in keys:
        #     updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6) # Replace values with 1e6
        # i = i - num_removed

    # update after every heartbeat
    if t - time_since_beat > T:
        time_since_beat = t



    # update values needed in other systems
    for key, new_value in zip(
            [   # cardiac control inputs
                "P_sa_store", "dP_sa_dt_store", "Q_bp_store", "Q_hp_store",
                "Q_rmp_store", "Q_amp_store", "Wh_lv_store", "Wh_rv_store",
                "time_since_beat_store",
                # gas exchange inputs
                "Q_pp_store", "Q_la_store"],

            [   P_sa, dP_sa_dt, Q_bp, Q_hp, Q_rmp, Q_amp,
                Wh_lv, Wh_rv, time_since_beat, Q_pp, Q_la]
    ):
        updates[key][((i - num_removed) % BUFFER_LIMIT)] = new_value



    # # just for plotting purposes
    # if ((t % time_saved) < 0.001 or (time_saved - (t % time_saved)) < 0.001) and num_removed == 0:

    keys_and_values = zip(
        [
            # Cardio control inputs
            "P_sa", "dP_sa_dt", "Q_bp", "Q_hp", "Q_rmp", "Q_amp", "Wh_lv", "Wh_rv", "time_since_beat",

            # Gas exchange inputs
            "Q_pp", "Q_la",

            # For plot
            "Q_lv", "Q_ra", "Q_rv", "P_ra", "P_la", "P_lv", "P_rv", "Pmax_lv", "Pmax_rv",
            "Pmax_la", "Pmax_ra", "V_rv", "V_ra", "V_lv", "V_la", "VT_rv", "VT_ra",
            "VT_lv", "VT_la", "P_pa", "P_pp", "P_pv", "P_thor", "V_vc", "P_vc", "Qi_lv",
            "Qi_rv", "V_pa", "phi", "phi_atr", "S", "V_pv", "V_pp", "P_amv", "P_ev", "V_u",
            "V_sv", "V_rmv", "V_amv", "V_bv", "V_hv", "P_sp", "Q_sa", "Q_jp", "Q_vc", "VT_amv",
            "P_im", "Q_amv", "Q_sp", "Q_ep", "Q_pa", "P_abd", "V_sa", "P_bv", "Q_bv", "R_bv",
            "VT_ev", "Q_ev", "VT_pa", "VT_pp", "VT_pv", "VT_sv", "VT_bv", "VT_hv", "VT_rmv",
            "VT_vc", "P_0", "time_history"],

        [   # Corresponding values
            P_sa, dP_sa_dt, Q_bp, Q_hp, Q_rmp, Q_amp, Wh_lv, Wh_rv, time_since_beat,
            Q_pp, Q_la, Q_lv, Q_ra, Q_rv, P_ra, P_la, P_lv, P_rv, Pmax_lv, Pmax_rv,
            Pmax_la, Pmax_ra, V_rv, V_ra, V_lv, V_la, VT_rv, VT_ra,
            VT_lv, VT_la, P_pa, P_pp, P_pv, P_thor, V_vc, P_vc, Qi_lv,
            Qi_rv, V_pa, phi, phi_atr, S, V_pv, V_pp, P_amv, P_ev, V_u,
            V_sv, V_rmv, V_amv, V_bv, V_hv, P_sp, Q_sa, Q_jp, Q_vc, VT_amv,
            P_im, Q_amv, Q_sp, Q_ep, Q_pa, P_abd, V_sa, P_bv, Q_bv, R_bv,
            VT_ev, Q_ev, VT_pa, VT_pp, VT_pv, VT_sv, VT_bv, VT_hv, VT_rmv,
            VT_vc, P_0, t])

    for key, value in keys_and_values:
        updates[key][updates["j"].item()- num_removed] = value


    return [dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
            dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dVT_ev_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt]