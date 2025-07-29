import numpy as np
import math
from Parameters import Parameters
from Test_controller import source
from Activation_Functions import activation_U, activation_H, activation_Naghavi, g_function, activation_H_derivative, \
    activation_conduit, activation_S


def frac(x):
    return x - math.floor(x)


def accept_index_evals(idx_in_2, idx_in_1, buffer_limit, i):
    if idx_in_2 <= idx_in_1:
        indices = np.arange(idx_in_2, idx_in_1 + 1)
    else:
        indices = np.concatenate([np.arange(idx_in_2, buffer_limit), np.arange(0, idx_in_1 + 1)])

    # Take every 3rd step, rk23
    accepted_index = []
    for j in range(len(indices)):
        if (i - 2 - j) % 3 == 0 and (i - 2 - j) > 0:
            s = indices[len(indices) - 1 - j]
            accepted_index.append(s)

    # A = list(buffer)
    # AA = list(accepted_values)
    return accepted_index


def get_delayed_value(t, delay, t_start, all_time, heart_index, buffer_limit, history_array, default_value):
    delay_time = t - delay
    if delay_time >= t_start:
        if delay_time >= all_time[0]:
            # No wrap-around
            delay_index = np.searchsorted(all_time[:heart_index + 1], delay_time, side='right')
        else:
            # Wrap-around
            idx_in_sorted = np.searchsorted(all_time[heart_index + 1:], delay_time, side='right')
            delay_index = (idx_in_sorted + heart_index + 1) % buffer_limit
        return history_array[delay_index]
    else:
        return default_value



def model_derivatives(t, state, params, heart_inputs, heart_control_inputs, gas_exchange_inputs, resp_control_inputs, updates, num_removed, t_start, time_saved, i, all_time, BUFFER_LIMIT):

    # # State variables
    (# Cardio state variables
     VT_pa, VT_pp, VT_pv, Q_pa,
     VT_la, VT_lv, VT_ra, VT_rv,
     VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, VT_ev, P_sp, P_sa, Q_sa, VT_vc,
     theta_ao, dtheta_ao_dt, theta_po, dtheta_po_dt, theta_mi, dtheta_mi_dt, theta_tr, dtheta_tr_dt,

    # Cardio controller state variables
    theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv, theta_change_O2_sh,
    theta_change_CO2_sh, P_tilda, f_ac, f_ap, R_ep_change, R_sp_change,
    R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
    Emax_rv_change, Ts_change, Tv_change, xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met
    ) = state

    # # Parameters
    # Cardio parameters
    (A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa, R_pa,
     R_pp, R_pv, Vu_pa, Vu_pp, Vu_pv, KE_lv, KE_rv, P0_lv, P0_rv, Vu_la, Vu_lv, Vu_ra, Vu_rv, Emax_la, P0_la, KE_la, Emax_ra,
     P0_ra, KE_ra, C_sa, L_sa, R_sa, Vu_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n, Vu_vc, Vvc_max, Vvc_min, C_ep, C_sp, C_bp,
     C_hp, C_rmp, C_amp, V_tot, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv, Vu_ep,
     Vu_sp, Vu_bp, Vu_hp, Vu_rmp, Vu_amp, kr_am, Vu_bv, Vu_hv) = (params[k] if k in params else Parameters[k] for k in ["A_im", "Tc", "T_im", "g_abd", "g_thor",
    "P_abdmax_n", "P_abdmin_n", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv",
    "Vu_pa", "Vu_pp", "Vu_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "Emax_la", "P0_la",
    "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa", "R_sa", "Vu_sa", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n",
    "Vu_vc", "Vvc_max", "Vvc_min", "C_ep", "C_sp", "C_bp", "C_hp", "C_rmp", "C_amp", "V_tot", "R_ev_n", "R_sv_n", "R_bv_n",
    "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv", "Vu_ep", "Vu_sp", "Vu_bp", "Vu_hp",
    "Vu_rmp", "Vu_amp", "kr_am", "Vu_bv", "Vu_hv"])

    # Cardio controller parameters
    (MRBCO2, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v,
     Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv,
     Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp,
     GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, tau_Emax_lv, tau_Emax_rv, tau_Ramp,
     tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, AT, MRTCO2_basal,
     g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, tau_cc, tau_isc, theta_shn, theta_spn,
     theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, tau_p, tau_z, f_acCO2_n, f_ac_max, f_ac_min, k_ac,
     K_H, PaO2_ac_n, tau_ac, G_ap, tau_ap, DT_v, GT_s, GT_v, T0, tau_Ts, tau_Tv, A, B, C, D, Cvb_O2_n, gb_O2, MO2_bp, R_bpn,
     tau_CO2, tau_O2, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, tau_w, W_hn, Cvam_O2_n,
     gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min, tau_M, tau_met) = [params[k] if k in params else Parameters[k] for k in ["MRBCO2", "fab_o", "fes_o",
    "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv",
    "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh",
    "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv",
    "Wt_v", "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
    "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep",
    "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "AT",
    "MRTCO2_basal", "g_ccsh", "g_ccsp", "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "tau_cc",
    "tau_isc", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab",
    "P_n", "tau_p", "tau_z", "f_acCO2_n", "f_ac_max", "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "tau_ac", "G_ap", "tau_ap",
    "DT_v", "GT_s", "GT_v", "T0", "tau_Ts", "tau_Tv", "A", "B", "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "tau_CO2",
    "tau_O2", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "tau_w",
    "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet", "MO2_ampn", "phi_max", "phi_min", "tau_M", "tau_met"]]

    # Determine the correct index based on t
    if t == t_start:
        heart_index = i % BUFFER_LIMIT
        heart_control_index = i % BUFFER_LIMIT
        gas_index = i % BUFFER_LIMIT
        resp_control_index = i % BUFFER_LIMIT
    else:
        heart_index = (i - num_removed - 1) % BUFFER_LIMIT
        heart_control_index = (i - num_removed - 1) % BUFFER_LIMIT
        gas_index = (i - num_removed - 1) % BUFFER_LIMIT
        resp_control_index = (i - num_removed - 1) % BUFFER_LIMIT

    resp_control_index = 0
    gas_index = 0

    # AAAAAA = heart_control_inputs["f_sp_store"][:heart_index + 20]
    # AAAAAAA = heart_control_inputs["check_time"]
    # AAAAAAAA = heart_control_inputs["P_sa"][:heart_index + 20]
    # AAAAAAAAA = heart_control_inputs["all_time"]
    # AAAAAAAAAA = heart_control_inputs["HR_store"]
    # AAAAAAAAAAA = heart_control_inputs["Vu_ev_every_store"]

    # # Cardio inputs from other systems
    # resp
    T_resp = 1 / resp_control_inputs["BF_store"][resp_control_index]
    TI = resp_control_inputs["TI_store"][resp_control_index]
    VT = resp_control_inputs["VT_store"][resp_control_index]

    # cardio controller
    T = 1 / heart_control_inputs["HR_store"][heart_control_index]  # heart period

    R_ep = R_ep_change + R_ep0
    R_sp = R_sp_change + R_sp0

    R_amp_n = R_amp_n_change + R_amp0
    R_amp = R_amp_n / (1 + xam_O2 + x_met)

    R_rmp_n = R_rmp_n_change + R_rmp0
    R_rmp = R_rmp_n * (1 + xrm_CO2) / (1 + xrm_O2)

    G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
    R_bp = 1 / G_bp

    R_hp = R_hpn * (1 + xh_CO2) / (1 + xh_O2)

    # # Cardio controller inputs from other systems
    MRTCO2 = gas_exchange_inputs["MRTCO2_store"][gas_index]
    Pa_O2 = gas_exchange_inputs["Pa_O2_store"][gas_index]
    Pa_CO2 = gas_exchange_inputs["Pa_CO2_store"][gas_index]
    Ca_O2 = gas_exchange_inputs["Ca_O2_store"][gas_index]

    VE_integral = resp_control_inputs["VE_integral_store"][resp_control_index]

    # get the correct basal tissue CO2 production rate and exercise intensity from the inputs
    MRTCO2_basal = MRTCO2_basal - MRBCO2
    I = (MRTCO2 - MRTCO2_basal) / (AT - MRTCO2_basal)

    time_since_beat = updates["time_since_beat_store"][heart_index]
    # Update after every heartbeat
    if t - time_since_beat > T:
        if time_since_beat >= all_time[0]: # No wrap-around
            idx_in_2 = np.searchsorted(all_time[:heart_index + 1], time_since_beat, side='right')
        else:  # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[heart_index + 1:], time_since_beat, side='right')
            idx_in_2 = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT

        idx_in_1 = heart_index
        time_since_beat = time_since_beat + T

        accepted_indices = accept_index_evals(idx_in_2, idx_in_1, BUFFER_LIMIT, (i-num_removed))

        HR = np.mean(updates["HR_every_store"][accepted_indices])
        T = 1/HR
        Vu_ev = np.mean(updates["Vu_ev_every_store"][accepted_indices])
        Vu_sv = np.mean(updates["Vu_sv_every_store"][accepted_indices])
        Vu_rmv = np.mean(updates["Vu_rmv_every_store"][accepted_indices])
        Vu_amv = np.mean(updates["Vu_amv_every_store"][accepted_indices])
        Emax_lv = np.mean(updates["Emax_lv_every_store"][accepted_indices])
        Emax_rv = np.mean(updates["Emax_rv_every_store"][accepted_indices])

    else:
        HR = heart_control_inputs["HR_store"][heart_control_index]
        Vu_ev = heart_control_inputs["Vu_ev_store"][heart_control_index]  # previous mean value
        Vu_sv = heart_control_inputs["Vu_sv_store"][heart_control_index]  # previous mean value
        Vu_rmv = heart_control_inputs["Vu_rmv_store"][heart_control_index]  # previous mean value
        Vu_amv = heart_control_inputs["Vu_amv_store"][heart_control_index]  # previous mean value
        Emax_lv = heart_control_inputs["Emax_lv_store"][heart_control_index]  # previous mean value
        Emax_rv = heart_control_inputs["Emax_rv_store"][heart_control_index]  # previous mean value


    # # Cardiovascular System
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

    if 0 <= S < third:
        P_abd = P_abdmax - (P_abdmax - P_abdmin) * (T_resp / (TI / 2)) * S

    elif third <= S < first:
        P_abd = P_abdmin

    elif first <= S <= second:
        P_abd = P_abdmax - (P_abdmax - P_abdmin) * ((TI + TE - T_resp * S) / TE)


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
    if VT_la > Vu_la: # LA stressed volume is the total minus unstressed
        V_la = VT_la - Vu_la
    else:
        V_la = 0

    if VT_ra > Vu_ra: # RA stressed volume is the total minus unstressed
        V_ra = VT_ra - Vu_ra
    else:
        V_ra = 0

    if VT_rv > Vu_rv: # RV stressed volume is the total minus unstressed
        V_rv = VT_rv - Vu_rv
    else:
        V_rv = 0

    # V_lv can be growing but there should not be any flow (Q) into the ventricles?
    if VT_lv > Vu_lv: # LV stressed volume is the total minus unstressed
        V_lv = VT_lv - Vu_lv
    else:
        V_lv = 0


    # activation function for contraction of the ventricle and atria
    phi = activation_H(t - time_since_beat, 0, T)
    phi_atr = activation_H(t - time_since_beat, 1, T)

    Pmax_lv = phi * Emax_lv * (VT_lv - Vu_lv) + (1 - phi) * P0_lv * (np.exp(KE_lv * VT_lv) - 1) + P_thor
    Pmax_ra = phi_atr * Emax_ra * (VT_ra - Vu_ra) + (1 - phi_atr) * P0_ra * (np.exp(KE_ra * VT_ra) - 1) + P_thor
    Pmax_rv = phi * Emax_rv * (VT_rv - Vu_rv) + (1 - phi) * P0_rv * (np.exp(KE_rv * VT_rv) - 1) + P_thor
    Pmax_la = phi_atr * Emax_la * (VT_la - Vu_la) + (1 - phi_atr) * P0_la * (np.exp(KE_la * VT_la) - 1) + P_thor


    # aortic valve
    ####################################
    # parameters:
    Kp_ao = 800
    Kf_ao = 800
    Kb_ao = 2
    Kv_ao = 10
    theta_ao_max = 1.309  # 75 degrees to radian
    theta_ao_min = 0.0872665  # 5 degrees to radian

    if Pmax_lv - P_sa > 0:
        if theta_ao > theta_ao_max:
            theta_ao = theta_ao_max
        AR_ao = ((1 - np.cos(theta_ao)) ** 2) / ((1 - np.cos(theta_ao_max)) ** 2)
        # AR_ao = 1

        Q_lv = (math.sqrt(Pmax_lv - P_sa) * AR_ao * 350)

        d2theta_ao_dt2 = (Pmax_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt + Kb_ao * Q_lv * np.cos(
            theta_ao) - Kv_ao * Q_lv * np.sin(theta_ao)
        P_lv = Pmax_lv
    else:
        Q_lv = 0.0
        # if theta_ao < theta_ao_min:
        theta_ao = theta_ao_min
        # theta_ao = 0.0872665  # theta_ao_min
        # dtheta_ao_dt = 0.0
        d2theta_ao_dt2 = 0.0
        # d2theta_ao_dt2 = (Pmax_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt + Kb_ao * Q_lv * np.cos(theta_ao)
        P_lv = Pmax_lv
    ####################################

    ####################################
    Kp_mi = 100
    Kf_mi = 800
    Kb_mi = 2
    Kv_mi = 3.5
    theta_mi_max = 1.309  # 75 degrees to radian
    theta_mi_min = 0.0872665  # 5 degrees to radian

    if Pmax_la > P_lv:
        if theta_mi > theta_mi_max:
            theta_mi = theta_mi_max
        AR_mi = ((1 - np.cos(theta_mi)) ** 2) / ((1 - np.cos(theta_mi_max)) ** 2)
        Qi_lv = math.sqrt(Pmax_la - P_lv) * AR_mi * 350

        d2theta_mi_dt2 = (Pmax_la - P_lv) * Kp_mi * np.cos(theta_mi) - Kf_mi * dtheta_mi_dt + Kb_mi * Qi_lv * np.cos(
            theta_mi) - Kv_mi * Qi_lv * np.sin(theta_mi)
        P_la = Pmax_la
    else:
        Qi_lv = 0
        theta_mi = theta_mi_min
        d2theta_mi_dt2 = 0.0
        P_la = Pmax_la
    ####################################

    ####################################
    Kp_po = 800
    Kf_po = 800
    Kb_po = 2
    Kv_po = 7
    theta_po_max = 1.309 # 75 degrees to radian

    if Pmax_rv > P_pa:
        if theta_po > theta_po_max:
            theta_po = theta_po_max
        AR_po = ((1 - np.cos(theta_po)) ** 2) / ((1 - np.cos(theta_po_max)) ** 2)
        Q_rv = (math.sqrt(Pmax_rv - P_pa) * AR_po * 350)

        d2theta_po_dt2 = (Pmax_rv - P_pa) * Kp_po * np.cos(theta_po) - Kf_po * dtheta_po_dt + Kb_po * Q_rv * np.cos(theta_po) - Kv_po * Q_rv * np.sin(theta_po)
        P_rv = Pmax_rv
    else:
        Q_rv = 0
        theta_po = 0.0872665
        d2theta_po_dt2 = 0.0
        P_rv = Pmax_rv
    ####################################

    ####################################
    Kp_tr = 2000
    Kf_tr = 800
    Kb_tr = 2
    Kv_tr = 7
    theta_tr_max = 1.309  # 75 degrees to radian

    if Pmax_ra > P_rv:
        if theta_tr > theta_tr_max:
            theta_tr = theta_tr_max
        AR_tr = ((1 - np.cos(theta_tr)) ** 2) / ((1 - np.cos(theta_tr_max)) ** 2)
        Qi_rv = math.sqrt(Pmax_ra - P_rv) * AR_tr * 350

        d2theta_tr_dt2 = (Pmax_ra - P_rv) * Kp_tr * np.cos(theta_tr) - Kf_tr * dtheta_tr_dt + Kb_tr * Qi_rv * np.cos(
            theta_tr) - Kv_tr * Qi_rv * np.sin(theta_tr)
        P_ra = Pmax_ra
    else:
        Qi_rv = 0
        theta_tr = 0.0872665
        d2theta_tr_dt2 = 0.0
        P_ra = Pmax_ra
    ####################################


    Q_la = (P_pv - P_la) / R_pv
    Q_pp = (P_pp - P_pv) / R_pp

    dVT_pa_dt = Q_rv - Q_pa
    dVT_pp_dt = Q_pa - Q_pp
    dVT_pv_dt = Q_pp - Q_la
    dQ_pa_dt = (P_pa - R_pa * Q_pa - P_pp) / L_pa


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
    # else:
    #     source_values = 0

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
        R_amv = R_amv_n * ((P_amv - P_vc) / (P_amv - P_am))
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

    P_ev = left_over_volume / C_ev # + source_values

    Q_ep = (P_sp - P_ev) / R_ep


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








    # Cardiovascular Controller
    # deal with rejected steps
    if t - updates["finish_breath_time"][-1] < 0:
        updates["finish_breath_time"].pop()
        updates["Nd"] = updates["Nd"][:-5]

    a1, a2, tau, t1, t2 = updates["Nd"][-5:]
    prev_flat_bit = updates["prev_flat_bit_store"][gas_index]

    last_breath_time = t - updates["finish_breath_time"][-1]

    if last_breath_time % (t1 + t2) < t1:
        Nt = VE_integral - prev_flat_bit  # Take value minus previous flat bit
    else:
        Nt = 0  # Reset to zero
        prev_flat_bit = VE_integral

    ## CNS Ischemic Response
    w_sp = x_sp / (1 + np.exp((Pa_O2 - PO2_sp) / kisc_sp))
    theta_sp = theta_spn - theta_change_O2_sp - theta_change_CO2_sp
    dtheta_change_O2_sp_dt = (-theta_change_O2_sp + w_sp) / tau_isc
    dtheta_change_CO2_sp_dt = (-theta_change_CO2_sp + g_ccsp * (Pa_CO2 - PaCO2_n)) / tau_cc

    w_sv = x_sv / (1 + np.exp((Pa_O2 - PO2_sv) / kisc_sv))
    theta_sv = theta_svn - theta_change_O2_sv - theta_change_CO2_sv
    dtheta_change_O2_sv_dt = (-theta_change_O2_sv + w_sv) / tau_isc
    dtheta_change_CO2_sv_dt = (-theta_change_CO2_sv + g_ccsv * (Pa_CO2 - PaCO2_n)) / tau_cc

    w_sh = x_sh / (1 + np.exp((Pa_O2 - PO2_sh) / kisc_sh))
    theta_sh = theta_shn - theta_change_O2_sh - theta_change_CO2_sh
    dtheta_change_O2_sh_dt = (-theta_change_O2_sh + w_sh) / tau_isc
    dtheta_change_CO2_sh_dt = (-theta_change_CO2_sh + g_ccsh * (Pa_CO2 - PaCO2_n)) / tau_cc

    ## Afferent Pathways
    # exp_arg = np.clip((P_tilda - P_n) / k_ab, -40, 40)  # Prevent overflow
    exp_arg = (P_tilda - P_n) / k_ab
    f_ab = (f_ab_min + f_ab_max * np.exp(exp_arg)) / (1 + np.exp(exp_arg))
    dP_tilda_dt = (P_sa + tau_z * dP_sa_dt - P_tilda) / tau_p

    # afferent chemoreflex pathway constant parameters
    if Pa_O2 >= 80:
        K = K_H
    elif 40 <= Pa_O2 < 80:
        K = K_H - (1.2 * (Pa_O2 - 80) / 30)
    else:
        K = K_H - 1.6

    phi_ac = ((f_ac_max + f_ac_min * np.exp((Pa_O2 - PaO2_ac_n) / k_ac)) / (1 + np.exp((Pa_O2 - PaO2_ac_n) / k_ac)) *
              (K * np.log(Pa_CO2 / PaCO2_n) + f_acCO2_n))

    d_fac_dt = (phi_ac - f_ac) / tau_ac

    # afferent activity from Pulmonary Stretch Receptors constant parameters
    phi_ap = G_ap * VT
    df_ap_dt = (phi_ap - f_ap) / tau_ap

    ## Efferent Pathways constant parameters
    Y_sh = (Ysh_min + Ysh_max * np.exp((I - Io_sh) / kcc_sh)) / (1 + np.exp((I - Io_sh) / kcc_sh))
    f_ash = Wt_sh * Nt + Wb_sh * f_ab + Wc_sh * f_ac + Wp_sh * f_ap - theta_sh
    f_sh = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash) + Y_sh))

    Y_sp = (Ysp_min + Ysp_max * np.exp((I - Io_sp) / kcc_sp)) / (1 + np.exp((I - Io_sp) / kcc_sp))
    f_asp = Wt_sp * Nt + Wb_sp * f_ab + Wc_sp * f_ac + Wp_sp * f_ap - theta_sp
    f_sp = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp) + Y_sp))

    Y_sv = (Ysv_min + Ysv_max * np.exp((I - Io_sv) / kcc_sv)) / (1 + np.exp((I - Io_sv) / kcc_sv))
    f_asv = Wt_sv * Nt + Wb_sv * f_ab + Wc_sv * f_ac + Wp_sv * f_ap - theta_sv
    f_sv = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv) + Y_sv))

    Y_v = (Yv_min + Yv_max * np.exp((I - Io_v) / kcc_v)) / (1 + np.exp((I - Io_v) / kcc_v))
    first_term = (fev_o + fev_inf * np.exp((f_ab - fab_o) / kev)) / (1 + np.exp((f_ab - fab_o) / kev))
    f_v = first_term - Wt_v * Nt - Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v
    # f_v1 = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v # changed


    f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history = [updates[key] for key in
    ["f_sp_store", "f_sh_store", "f_v_store", "f_sv_store", "phi_met_store"]]

    # Fetch delayed values
    # f_sp_delay2 = get_delayed_value(t, 2, t_start, all_time, heart_index, BUFFER_LIMIT, f_sp_history, 3.97)
    # f_sh_delay2 = get_delayed_value(t, 2, t_start, all_time, heart_index, BUFFER_LIMIT, f_sh_history, 3.8576)
    # f_sv_delay5 = get_delayed_value(t, 5, t_start, all_time, heart_index, BUFFER_LIMIT, f_sv_history, 3.97)
    # f_v_delay0_2 = get_delayed_value(t, DT_v, t_start, all_time, heart_index, BUFFER_LIMIT, f_v_history, 4.2748)

    f_sp_delay2 = f_sp
    f_sh_delay2 = f_sh
    f_sv_delay5 = f_sv
    f_v_delay0_2 = f_v

    # heart period
    sigma_Ts = GT_s * np.log(max(f_sh_delay2, fes_min) - fes_min + 1)
    d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts

    sigma_Tv = GT_v * f_v_delay0_2
    d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv

    T = Ts_change + Tv_change + T0
    HR_every = 1 / T


    # continue with equations
    sigma_Rep = GR_ep * np.log(max(f_sp_delay2, fes_min) - fes_min + 1)
    sigma_Rsp = GR_sp * np.log(max(f_sp_delay2, fes_min) - fes_min + 1)
    sigma_Rrmp_n = GR_rmp * np.log(max(f_sp_delay2, fes_min) - fes_min + 1)
    sigma_Ramp_n = GR_amp * np.log(max(f_sp_delay2, fes_min) - fes_min + 1)

    sigma_Vu_ev = GV_ev * np.log(max(f_sv_delay5, fes_min) - fes_min + 1)
    sigma_Vu_sv = GV_sv * np.log(max(f_sv_delay5, fes_min) - fes_min + 1)
    sigma_Vu_rmv = GV_rmv * np.log(max(f_sv_delay5, fes_min) - fes_min + 1)
    sigma_Vu_amv = GV_amv * np.log(max(f_sv_delay5, fes_min) - fes_min + 1)

    sigma_Emax_lv = GEmax_lv * np.log(max(f_sh_delay2, fes_min) - fes_min + 1)
    sigma_Emax_rv = GEmax_rv * np.log(max(f_sh_delay2, fes_min) - fes_min + 1)

    dR_ep_change_dt = (- R_ep_change + sigma_Rep) / tau_Rep
    dR_sp_change_dt = (- R_sp_change + sigma_Rsp) / tau_Rsp
    dR_rmp_n_change_dt = (- R_rmp_n_change + sigma_Rrmp_n) / tau_Rrmp
    dR_amp_n_change_dt = (- R_amp_n_change + sigma_Ramp_n) / tau_Ramp

    dVu_ev_change_dt = (- Vu_ev_change + sigma_Vu_ev) / tau_Vev
    dVu_sv_change_dt = (- Vu_sv_change + sigma_Vu_sv) / tau_Vsv
    dVu_rmv_change_dt = (- Vu_rmv_change + sigma_Vu_rmv) / tau_Vrmv
    dVu_amv_change_dt = (- Vu_amv_change + sigma_Vu_amv) / tau_Vamv
    Vu_ev_every = max(Vu_ev_change + Vu_ev0, 0)
    Vu_sv_every = max(Vu_sv_change + Vu_sv0, 0)
    Vu_rmv_every = max(Vu_rmv_change + Vu_rmv0, 0)
    Vu_amv_every = max(Vu_amv_change + Vu_amv0, 0)

    dEmax_lv_change_dt = (- Emax_lv_change + sigma_Emax_lv) / tau_Emax_lv
    dEmax_rv_change_dt = (- Emax_rv_change + sigma_Emax_rv) / tau_Emax_rv
    Emax_lv_every = Emax_lv_change + Emax_lv0
    Emax_rv_every = Emax_rv_change + Emax_rv0


    ## Blood Flow Local Control
    Cvb_O2 = Ca_O2 - MO2_bp / Q_bp
    dxb_O2_dt = (- xb_O2 - gb_O2 * (Cvb_O2 - Cvb_O2_n)) / tau_O2

    numerator = A + (B / (1 + C * np.exp(D * np.log10(Pa_CO2))))
    denominator = A + (B / (1 + C * np.exp(D * np.log10(PaCO2_n))))
    phi_b = numerator / denominator - 1
    dxb_CO2_dt = (- xb_CO2 - phi_b) / tau_CO2

    # coronary
    MO2_hp = MO2_hpn * Wh / W_hn
    Cvh_O2 = Ca_O2 - MO2_hp / Q_hp
    dxh_O2_dt = (- xh_O2 - gh_O2 * (Cvh_O2 - Cvh_O2_n)) / tau_O2

    phi_h = (1 - np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2))
    dxh_CO2_dt = (- xh_CO2 + phi_h) / tau_CO2

    wh = Wh_lv + Wh_rv
    dWh_dt = (wh - Wh) / tau_w

    # resting muscle
    Cvrm_O2 = Ca_O2 - MO2_rmp / Q_rmp
    dxrm_O2_dt = (- xrm_O2 - grm_O2 * (Cvrm_O2 - Cvrm_O2_n)) / tau_O2

    phi_rm = (1 - np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2))
    dxrm_CO2_dt = (- xrm_CO2 + phi_rm) / tau_CO2

    # active muscle blood flow
    MO2_amp = MO2_ampn * (1 + xM)
    Cvam_O2 = Ca_O2 - MO2_amp / Q_amp
    dxam_O2_dt = (- xam_O2 - gam_O2 * (Cvam_O2 - Cvam_O2_n)) / tau_O2

    dxM_dt = (- xM + gM * I) / tau_M

    phi_met = (phi_min + phi_max * np.exp((I - Io_met) / kmet)) / (1 + np.exp((I - Io_met) / kmet))
    # phi_met_delay = get_delayed_value(t, 4, t_start, all_time, heart_index, BUFFER_LIMIT, phi_met_history, phi_met)
    phi_met_delay = phi_met

    dx_met_dt = (- x_met + phi_met_delay) / tau_met

    # Cardiovascular Controller
    # update values needed in other systems
    for key, new_value in zip(
            [  # Cardio inputs
                "time_since_beat_store",

                "HR_store", "Vu_ev_store", "Vu_sv_store", "Vu_rmv_store", "Vu_amv_store",
                "Emax_lv_store", "Emax_rv_store", "f_sp_store", "f_sh_store",
                "f_v_store", "f_sv_store", "phi_met_store", "HR_every_store", "Vu_ev_every_store",
                "Vu_sv_every_store", "Vu_rmv_every_store", "Vu_amv_every_store", "Emax_lv_every_store",
                "Emax_rv_every_store",

                # Needed in cardio controller
                "prev_flat_bit_store"],

            [time_since_beat,
             HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
             Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, phi_met, HR_every, Vu_ev_every, Vu_sv_every,
             Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,
             prev_flat_bit]
    ):
        updates[key][((i - num_removed) % BUFFER_LIMIT)] = new_value



        # # just for plotting purposes
    keys_and_values = zip(
        [
            # Cardio control inputs
            "P_sa", "Q_bp", "Q_hp", "Q_rmp", "Q_amp",

            # Gas exchange inputs
            "Q_pp", "Q_la",

            # For plot
            "Q_lv", "Q_ra", "Q_rv", "P_ra", "P_la", "P_lv", "P_rv", "Pmax_lv", "Pmax_rv",
            "Pmax_la", "Pmax_ra", "VT_rv", "VT_ra",
            "VT_lv", "VT_la", "P_pa", "P_pp", "P_pv", "P_thor", "P_vc", "Qi_lv",
            "Qi_rv", "phi", "phi_atr", "P_amv", "P_ev", "V_u",
            "P_sp", "Q_sa", "Q_vc", "VT_amv",
            "Q_amv", "Q_pa", "V_sa", "P_bv", "R_bv",
            "VT_ev", "Q_ev", "VT_pa", "VT_pp", "VT_pv", "VT_sv", "VT_bv", "VT_hv", "VT_rmv",
            "VT_vc", "time_history", "theta_ao", "theta_po", "theta_mi", "theta_tr",
            "Q_sv", "Q_bv", "Q_hv", "Q_rmv"],

        [  # Corresponding values
            P_sa, Q_bp, Q_hp, Q_rmp, Q_amp,
            Q_pp, Q_la, Q_lv, Q_ra, Q_rv, P_ra, P_la, P_lv, P_rv, Pmax_lv, Pmax_rv,
            Pmax_la, Pmax_ra, VT_rv, VT_ra,
            VT_lv, VT_la, P_pa, P_pp, P_pv, P_thor, P_vc, Qi_lv,
            Qi_rv, phi, phi_atr, P_amv, P_ev, V_u,
            P_sp, Q_sa, Q_vc, VT_amv,
            Q_amv, Q_pa, V_sa, P_bv, R_bv,
            VT_ev, Q_ev, VT_pa, VT_pp, VT_pv, VT_sv, VT_bv, VT_hv, VT_rmv,
            VT_vc, t, theta_ao, theta_po, theta_mi, theta_tr,
            Q_sv, Q_bv, Q_hv, Q_rmv])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value


    # just for plotting purposes
    keys_and_values = zip(
        [  # Cardio inputs
            "HR", "Vu_ev", "Vu_sv", "Vu_rmv", "Vu_amv", "Emax_lv", "Emax_rv",
            "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "I", "f_sp", "f_sh", "f_v", "f_sv", "Nt", "f_ab",
            "f_ac", "f_ap", "Tv_change", "Ts_change", "HR_check", "f_sh_delay2", "f_v_delay02", "sigma_Ts",
            "sigma_Tv"
        ],

        [  # Corresponding values
            HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv, Emax_lv, Emax_rv,
            R_ep, R_amp, R_rmp, R_sp, R_bp, R_hp, I, f_sp, f_sh, f_v, f_sv, Nt, f_ab, f_ac, f_ap, Tv_change,
            Ts_change, HR_every,
            f_sh_delay2, f_v_delay0_2, sigma_Ts, sigma_Tv])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value



    return [# cardio derivatives
            dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
            dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dVT_ev_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
            dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt, d2theta_tr_dt2,

            # cardio controller derivatives
            dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
            dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
            dxM_dt, dx_met_dt
    ]