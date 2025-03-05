import bisect
import math

import numpy as np

def cardiovascular_controller(t, state, params, time_history, exp_inputs, heart_inputs, resp_control_inputs, gas_exchange_inputs, updates, all_time, num_removed):
    """
    Afferent Pathways state variables:
    theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv,
    theta_change_O2_sh, theta_change_CO2_sh, P_tilda, f_ac, f_ap

    Effectors for reflex control state variables:
    R_ep_change, R_sp_change, R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change,
    Vu_amv_change, Emax_lv_change, Emax_rv_change, Ts_change, Tv_change

    Blood Flow Local Control state variables:
    xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met

    """
    (theta_change_O2_sp, theta_change_CO2_sp, theta_change_O2_sv, theta_change_CO2_sv, theta_change_O2_sh,
     theta_change_CO2_sh, P_tilda, f_ac, f_ap, R_ep_change, R_sp_change,
     R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
     Emax_rv_change, Ts_change, Tv_change, xb_O2, xb_CO2, xh_O2, xh_CO2, Wh, xrm_O2, xrm_CO2, xam_O2, xM, x_met) = state

    ## Metabolic regulation
    # constant parameters
    AT = params["AT"]
    MRTCO2_basal = params["MRTCO2_basal"]

    # Other inputs
    MRTCO2 = gas_exchange_inputs["MRTCO2"][-1]
    # T_resp = 1 / resp_control_inputs["BF"]
    previous_VE = exp_inputs["previous_VE"][-1]

    VE_integral = resp_control_inputs["VE_integral"][-1]


    I = (MRTCO2 - MRTCO2_basal)/(AT - MRTCO2_basal)



    ## Respiratory neuromuscular drive
    # if t < TI:
    #     RR = 1
    # elif TI <= t < T_resp:
    #     RR = 0, no need as Nt 0 outside of TI

    # Nt = VE_integral

    a0, a1, a2, tau, t1, t2 = exp_inputs["Nd"][-6:]
    prev_flat_bit = updates["prev_flat_bit"][-1]

    if t % (t1 + t2) < t1:
        Nt = VE_integral - prev_flat_bit  # Take value minus previous flat bit
    else:
        Nt = 0  # Reset to zero
        prev_flat_bit = VE_integral

    ## CNS Ischemic Response
    # constant parameters
    g_ccsh = params["gccsh"]
    g_ccsp = params["gccsp"]
    g_ccsv = params["gccsv"]
    kisc_sh = params["kisc_sh"]
    kisc_sp = params["kisc_sp"]
    kisc_sv = params["kisc_sv"]
    PO2_sh = params["PO2_sh"]
    PO2_sp = params["PO2_sp"]
    PO2_sv = params["PO2_sv"]
    tau_cc = params["tau_cc"]
    tau_isc = params["tau_isc"]
    theta_shn = params["theta_shn"]
    theta_spn = params["theta_spn"]
    theta_svn = params["theta_svn"]
    x_sh = params["x_sh"]
    x_sp = params["x_sp"]
    x_sv = params["x_sv"]

    PaCO2_n = params["PaCO2_n"]

    # Other inputs
    Pa_O2 = gas_exchange_inputs["Pa_O2"][-1]
    Pa_CO2 = gas_exchange_inputs["Pa_CO2"][-1]

    # cns response
    w_sp = x_sp / (1 + np.exp((Pa_O2 - PO2_sp)/kisc_sp))
    dtheta_change_O2_sp_dt = (-theta_change_O2_sp + w_sp) / tau_isc
    dtheta_change_CO2_sp_dt = (-theta_change_CO2_sp + g_ccsp * (Pa_CO2 - PaCO2_n))/tau_cc

    theta_sp = theta_spn - theta_change_O2_sp - theta_change_CO2_sp

    w_sv = x_sv / (1 + np.exp((Pa_O2 - PO2_sv) / kisc_sv))
    dtheta_change_O2_sv_dt = (-theta_change_O2_sv + w_sv) / tau_isc
    dtheta_change_CO2_sv_dt = (-theta_change_CO2_sv + g_ccsv * (Pa_CO2 - PaCO2_n)) / tau_cc

    theta_sv = theta_svn - theta_change_O2_sv - theta_change_CO2_sv

    w_sh = x_sh / (1 + np.exp((Pa_O2 - PO2_sh) / kisc_sh))
    dtheta_change_O2_sh_dt = (-theta_change_O2_sh + w_sh) / tau_isc
    dtheta_change_CO2_sh_dt = (-theta_change_CO2_sh + g_ccsh * (Pa_CO2 - PaCO2_n)) / tau_cc

    theta_sh = theta_shn - theta_change_O2_sh - theta_change_CO2_sh



    ## Afferent Pathways
    # afferent baroreflex constant parameters
    f_ab_max = params["f_ab_max"]
    f_ab_min = params["f_ab_min"]
    k_ab = params["k_ab"]
    P_n = params["P_n"]
    tau_p = params["tau_p"]
    tau_z = params["tau_z"]

    # Other inputs
    P_sa = heart_inputs["P_sa"][-2] # cardiovascular controller was run after cardio was run with states appended/updated so it must take the nonupdated version
    dP_sa_dt = heart_inputs["dP_sa_dt"][-2]

    f_ab = (f_ab_min + f_ab_max * np.exp((P_tilda - P_n)/k_ab)) / (1 + np.exp((P_tilda - P_n)/k_ab))
    dP_tilda_dt = (P_sa + tau_z * dP_sa_dt - P_tilda) / tau_p

    # afferent chemoreflex pathway constant parameters
    f_ac_IC = params["f_ac_IC"]
    f_acCO2_n = params["f_acCO2_n"]
    f_ac_max = params["f_ac_max"]
    f_ac_min = params["f_ac_min"]
    k_ac = params["k_ac"]
    K_H = params["K_H"]
    PaO2_ac_n = params["PaO2_ac_n"]
    PaCO2_n = params["PaCO2_n"]
    tau_ac = params["tau_ac"]

    if Pa_O2 >= 80:
        K = K_H
    elif 40 <= Pa_O2 < 80:
        K = K_H - (1.2 * (Pa_O2 - 80) / 30)
    else:
        K = K_H - 1.6

    phi_ac = ((f_ac_max + f_ac_min * np.exp((Pa_O2 - PaO2_ac_n)/k_ac))/(1 + np.exp((Pa_O2 - PaO2_ac_n)/k_ac)) *
              (K * np.log(Pa_CO2/PaCO2_n) + f_acCO2_n))

    d_fac_dt = (phi_ac - f_ac) / tau_ac

    # afferent activity from Pulmonary Stretch Receptors constant parameters
    f_ap_IC = params["f_ap_IC"]
    G_ap = params["G_ap"]
    tau_ap = params["tau_ap"]

    # Other inputs
    VT = resp_control_inputs["VT"][-1]

    phi_ap = G_ap * VT
    df_ap_dt = (phi_ap - f_ap)/tau_ap


    ## Efferent Pathways constant parameters
    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
        kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
        Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v) = [params[key] for key in
                                                  ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v",
        "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min",
        "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh",
        "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v"]]


    Y_sh = (Ysh_min + Ysh_max * np.exp((I - Io_sh)/kcc_sh)) / (1 + np.exp((I - Io_sh)/kcc_sh))
    f_ash = Wt_sh * Nt + Wb_sh * f_ab + Wc_sh * f_ac + Wp_sh * f_ap - theta_sh
    f_sh = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash) + Y_sh
    if f_sh > fes_max:
        f_sh = fes_max

    Y_sp = (Ysp_min + Ysp_max * np.exp((I - Io_sp) / kcc_sp)) / (1 + np.exp((I - Io_sp) / kcc_sp))
    f_asp = Wt_sp * Nt + Wb_sp * f_ab + Wc_sp * f_ac + Wp_sp * f_ap - theta_sp
    f_sp = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp) + Y_sp
    if f_sp > fes_max:
        f_sp = fes_max

    Y_sv = (Ysv_min + Ysv_max * np.exp((I - Io_sv) / kcc_sv)) / (1 + np.exp((I - Io_sv) / kcc_sv))
    f_asv = Wt_sv * Nt + Wb_sv * f_ab + Wc_sv * f_ac + Wp_sv * f_ap - theta_sv
    f_sv = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv) + Y_sv
    if f_sv > fes_max:
        f_sv = fes_max

    Y_v = (Yv_min + Yv_max * np.exp((I - Io_v) / kcc_v)) / (1 + np.exp((I - Io_v) / kcc_v))
    first_term = (fev_o + fev_inf * np.exp((f_ab - fab_o)/kev)) / (1 + np.exp((f_ab - fab_o)/kev))
    # f_v = first_term - Wt_v * Nt - Wc_v * f_ac - Wp_v * f_ap - theta_v + Y_v
    f_v = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v
    #
    ## Effectors for reflex control
    # resistances, unstressed volumes, and cardiac elastances.
    # DEmax_lv = params["DEmax_lv"]
    # DEmax_rv = params["DEmax_rv"]
    # DR_amp = params["DR_amp"]
    # DR_ep = params["DR_ep"]
    # DR_rmp = params["DR_rmp"]
    # DR_sp = params["DR_sp"]
    # DV_amv = params["DV_amv"]
    # DV_ev = params["DV_ev"]
    # DV_rmv = params["DV_rmv"]
    # DV_sv = params["DV_sv"]

    (Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0,
     R_ep0, R_rmp0, R_sp0, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv,
     tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0) = [params[key] for key in
        ["Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
         "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
         "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0",
         "Vu_sv0"]]

    f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history = [exp_inputs[key] for key in
                                                                              ["f_sp_history", "f_sh_history",
                                                                               "f_v_history", "f_sv_history",
                                                                               "phi_met_history"]]

    # added the below to get f_sp_delay from previous iterations.
    delay_time2 = t - 2
    if delay_time2 >= 0:
        # Find the index for delay_time in time_history
        index = bisect.bisect_right(time_history, delay_time2) - 1
        f_sp_delay2 = f_sp_history[index]
        f_sh_delay2 = f_sh_history[index]
    else:
        if t == 0:
            f_sp_delay2 = f_sp
            f_sh_delay2 = f_sh
        else:
            # f_sp_delay2 = np.mean(f_sp_history)
            f_sp_delay2 = 3.97
            f_sh_delay2 = 3.8576 #(f_shIC)

    delay_time5 = t - 5
    if delay_time5 >= 0:
        # Find the index for delay_time in time_history
        index = bisect.bisect_right(time_history, delay_time5) - 1
        f_sv_delay5 = f_sv_history[index]
    else:
        if t == 0:
            f_sv_delay5 = f_sv
        else:
            f_sv_delay5 = 3.97
            # f_sv_delay5 = np.mean(f_sv_history)

    # continue with equations
    if f_sp < fes_min:
        sigma_Rep = 0
        sigma_Rsp = 0
        sigma_Rrmp_n = 0
        sigma_Ramp_n = 0

    else:
        sigma_Rep = GR_ep * np.log(f_sp_delay2 - fes_min + 1)
        sigma_Rsp = GR_sp * np.log(f_sp_delay2 - fes_min + 1)
        sigma_Rrmp_n = GR_rmp * np.log(f_sp_delay2 - fes_min + 1)
        sigma_Ramp_n = GR_amp * np.log(f_sp_delay2 - fes_min + 1)

    if f_sv < fes_min:
        sigma_Vu_ev = 0
        sigma_Vu_sv = 0
        sigma_Vu_rmv = 0
        sigma_Vu_amv = 0
    else:
        sigma_Vu_ev = GV_ev * np.log(f_sv_delay5 - fes_min + 1)
        sigma_Vu_sv = GV_sv * np.log(f_sv_delay5 - fes_min + 1)
        sigma_Vu_rmv = GV_rmv * np.log(f_sv_delay5 - fes_min + 1)
        sigma_Vu_amv = GV_amv * np.log(f_sv_delay5 - fes_min + 1)

    if f_sh < fes_min:
        sigma_Emax_lv = 0
        sigma_Emax_rv = 0
    else:
        sigma_Emax_lv = GEmax_lv * np.log(f_sh_delay2 - fes_min + 1)
        sigma_Emax_rv = GEmax_rv * np.log(f_sh_delay2 - fes_min + 1)

    dR_ep_change_dt = (- R_ep_change + sigma_Rep) / tau_Rep
    dR_sp_change_dt = (- R_sp_change + sigma_Rsp) / tau_Rsp
    dR_rmp_n_change_dt = (- R_rmp_n_change + sigma_Rrmp_n) / tau_Rrmp
    dR_amp_n_change_dt = (- R_amp_n_change + sigma_Ramp_n) / tau_Ramp

    dVu_ev_change_dt = (- Vu_ev_change + sigma_Vu_ev) / tau_Vev
    dVu_sv_change_dt = (- Vu_sv_change + sigma_Vu_sv) / tau_Vsv
    dVu_rmv_change_dt = (- Vu_rmv_change + sigma_Vu_rmv) / tau_Vrmv
    dVu_amv_change_dt = (- Vu_amv_change + sigma_Vu_amv) / tau_Vamv

    # dVu_ev_change_dt = 0
    # dVu_sv_change_dt = 0
    # dVu_rmv_change_dt = 0
    # dVu_amv_change_dt = 0

    dEmax_lv_change_dt = (- Emax_lv_change + sigma_Emax_lv) / tau_Emax_lv
    dEmax_rv_change_dt = (- Emax_rv_change + sigma_Emax_rv) / tau_Emax_rv

    R_ep = R_ep_change + R_ep0
    R_sp = R_sp_change + R_sp0
    R_rmp_n = R_rmp_n_change + R_rmp0
    R_amp_n = R_amp_n_change + R_amp0

    Vu_ev1 = Vu_ev_change + Vu_ev0
    Vu_sv1 = Vu_sv_change + Vu_sv0
    Vu_rmv1 = Vu_rmv_change + Vu_rmv0
    Vu_amv1 = Vu_amv_change + Vu_amv0

    Emax_lv1 = Emax_lv_change + Emax_lv0
    Emax_rv1 = Emax_rv_change + Emax_rv0

    # heart period constants
    DT_s = params["DT_s"]
    DT_v = params["DT_v"]
    fsh_IC = params["fsh_IC"]
    fv_IC = params["fv_IC"]
    GT_s = params["GT_s"]
    GT_v = params["GT_v"]
    T0 = params["T0"]
    tau_Ts = params["tau_Ts"]
    tau_Tv = params["tau_Tv"]

    delay_time0_2 = t - DT_v
    if delay_time0_2 >= 0:
        # Find the index for delay_time in time_history
        index = bisect.bisect_right(time_history, delay_time0_2) - 1
        f_v_delay0_2 = f_v_history[index]
    else:
        if t == 0:
            f_v_delay0_2 = f_v
        else:
            f_v_delay0_2 = 4.2748 # np.mean(f_v_history), f_v_IC

    if f_sh < fes_min:
        sigma_Ts = 0
    else:
        sigma_Ts = GT_s * np.log(f_sh_delay2 - fes_min + 1)

    d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts
    # d_Ts_change_dt = 0

    sigma_Tv = GT_v * f_v_delay0_2
    d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv
    # d_Tv_change_dt = 0

    T = Tv_change + Ts_change + T0

    HR1 = 1 / T

    ## Blood Flow Local Control
    # Cerebral Blood Flow constant parameters
    A = params["A"]
    B = params["B"]
    C = params["C"]
    D = params["D"]
    Cvb_O2_n = params["Cvb_O2_n"]
    gb_O2 = params["gb_O2"]
    MO2_bp = params["MO2_bp"]
    R_bpn = params["R_bpn"]
    tau_CO2 = params["tau_CO2"]
    tau_O2 = params["tau_O2"]

    # other inputs
    Ca_O2 = gas_exchange_inputs["Ca_O2"][-1]
    Q_bp = heart_inputs["Q_bp"][-2]
    Q_hp = heart_inputs["Q_hp"][-2]
    Q_rmp = heart_inputs["Q_rmp"][-2]
    Q_amp = heart_inputs["Q_amp"][-2]

    G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
    R_bp = 1 / G_bp
    Cvb_O2 = Ca_O2 - MO2_bp / Q_bp
    dxb_O2_dt = (- xb_O2 - gb_O2 * (Cvb_O2 - Cvb_O2_n)) / tau_O2
    numerator = A + (B / (1 + C * np.exp(D * np.log10(Pa_CO2))))
    denominator = A + (B / (1 + C * np.exp(D * np.log10(PaCO2_n))))
    phi_b = numerator / denominator - 1

    dxb_CO2_dt = (- xb_CO2 - phi_b) / tau_CO2

    # Coronary and Resting Muscle Blood Flow constant parameters
    Cvh_O2_n = params["Cvh_O2_n"]
    Cvrm_O2_n = params["Cvrm_O2_n"]
    gh_O2 = params["gh_O2"]
    grm_O2 = params["grm_O2"]
    Kh_CO2 = params["Kh_CO2"]
    Krm_CO2 = params["Krm_CO2"]
    MO2_hpn = params["MO2_hpn"]
    MO2_rmp = params["MO2_rmp"]
    R_hpn = params["R_hpn"]
    tau_w = params["tau_w"]
    W_hn = params["W_hn"]

    # other inputs
    Wh_lv = heart_inputs["Wh_lv"][-2]
    Wh_rv = heart_inputs["Wh_rv"][-2]

    # coronary
    R_hp = R_hpn * (1 + xh_CO2) / (1 + xh_O2)

    MO2_hp = MO2_hpn * Wh / W_hn
    Cvh_O2 = Ca_O2 - MO2_hp / Q_hp

    dxh_O2_dt = (- xh_O2 - gh_O2 * (Cvh_O2 - Cvh_O2_n)) / tau_O2

    phi_h = (1 - np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Kh_CO2))

    dxh_CO2_dt = (- xh_CO2 + phi_h) / tau_CO2

    wh = Wh_lv + Wh_rv

    dWh_dt = (wh - Wh) / tau_w

    # resting muscle
    R_rmp = R_rmp_n * (1 + xrm_CO2) / (1 + xrm_O2)
    Cvrm_O2 = Ca_O2 - MO2_rmp / Q_rmp

    dxrm_O2_dt = (- xrm_O2 - grm_O2 * (Cvrm_O2 - Cvrm_O2_n)) / tau_O2

    phi_rm = (1 - np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2)) / (1 + np.exp((Pa_CO2 - PaCO2_n) / Krm_CO2))

    dxrm_CO2_dt = (- xrm_CO2 + phi_rm) / tau_CO2

    # active muscle blood flow
    Cvam_O2_n = params["Cvam_O2_n"]
    Dmet = params["Dmet"]
    gam_O2 = params["gam_O2"]
    gM = params["gM"]
    Io_met = params["Io_met"]
    kmet = params["kmet"]
    MO2_ampn = params["MO2_ampn"]
    phi_max = params["phi_max"]
    phi_min = params["phi_min"]
    tau_M = params["tau_M"]
    tau_met = params["tau_met"]

    R_amp = R_amp_n / (1 + xam_O2 + x_met)

    MO2_amp = MO2_ampn * (1 + xM)
    Cvam_O2 = Ca_O2 - MO2_amp / Q_amp

    dxam_O2_dt = (- xam_O2 - gam_O2 * (Cvam_O2 - Cvam_O2_n)) / tau_O2

    dxM_dt = (- xM + gM * I) / tau_M

    phi_met = (phi_min + phi_max * np.exp((I - Io_met) / kmet)) / (1 + np.exp((I - Io_met) / kmet))

    delay_time_met = t - Dmet
    if delay_time_met >= 0:
        # Find the index for delay_time in time_history
        index = bisect.bisect_right(time_history, delay_time_met) - 1
        phi_met_delay = phi_met_history[index]
    else:
        phi_met_delay = phi_met


    dx_met_dt = (- x_met + phi_met_delay) / tau_met

    if t == 0:
        HR = HR1
        Vu_ev = Vu_ev1
        Vu_sv = Vu_sv1
        Vu_rmv = Vu_rmv1
        Vu_amv = Vu_amv1
        Emax_lv = Emax_lv1
        Emax_rv = Emax_rv1
    else:
        HR = updates["HR"][-1]
        Vu_ev = updates["Vu_ev"][-1]
        Vu_sv = updates["Vu_sv"][-1]
        Vu_rmv = updates["Vu_rmv"][-1]
        Vu_amv = updates["Vu_amv"][-1]
        Emax_lv = updates["Emax_lv"][-1]
        Emax_rv = updates["Emax_rv"][-1]

    U2 = updates["U2"][-1]

    # update after every heartbeat
    if U2 < 0.01 and updates["U2"][-2] > 0.99:
        HR = np.mean(updates["HR1"])
        updates["HR1"].clear()

        Vu_ev = np.mean(updates["Vu_ev1"])
        updates["Vu_ev1"].clear()

        Vu_sv = np.mean(updates["Vu_sv1"])
        updates["Vu_sv1"].clear()

        Vu_rmv = np.mean(updates["Vu_rmv1"])
        updates["Vu_rmv1"].clear()

        Vu_amv = np.mean(updates["Vu_amv1"])
        updates["Vu_amv1"].clear()

        Emax_lv = np.mean(updates["Emax_lv1"])
        updates["Emax_lv1"].clear()

        Emax_rv = np.mean(updates["Emax_rv1"])
        updates["Emax_rv1"].clear()


    if t != 0:
        if t < all_time[-1]:
            for key in [
                "f_sp_history", "f_sh_history", "f_v_history", "phi_met_history", "f_sv_history",
                "Vu_ev", "Vu_amv", "Vu_rmv", "Vu_sv", "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "HR",
                "Emax_lv", "Emax_rv", "I", "phi_met", "Nt", "Vu_sv_change", "prev_flat_bit", "Pa_O2", "HR1", "Vu_ev1",
                "Vu_sv1", "Vu_rmv1", "Vu_amv1", "Emax_lv1", "Emax_rv1", "T", "xb_O2", "Cvb_O2", "xb_CO2"
            ]:
                del updates[key][-num_removed:]

    # t_eval = updates["t_eval2"][0]
    # check2 = t
    # check3 = np.abs(t - t_eval)
    #
    # tolerance = 1e-3
    # if np.abs(t - t_eval) < tolerance:


        # update history
    updates["HR1"].append(HR1)
    updates["Vu_ev1"].append(Vu_ev1)
    updates["Vu_sv1"].append(Vu_sv1)
    updates["Vu_rmv1"].append(Vu_rmv1)
    updates["Vu_amv1"].append(Vu_amv1)
    updates["Emax_lv1"].append(Emax_lv1)
    updates["Emax_rv1"].append(Emax_rv1)

    updates["f_sp_history"].append(f_sp)
    updates["f_sh_history"].append(f_sh)
    updates["f_v_history"].append(f_v)
    updates["phi_met_history"].append(phi_met)
    updates["f_sv_history"].append(f_sv)

    updates["Vu_ev"].append(Vu_ev)
    updates["Vu_amv"].append(Vu_amv)
    updates["Vu_rmv"].append(Vu_rmv)
    updates["Vu_sv"].append(Vu_sv)
    updates["R_ep"].append(R_ep)
    updates["R_amp"].append(R_amp)
    updates["R_rmp"].append(R_rmp)
    updates["R_sp"].append(R_sp)
    updates["R_bp"].append(R_bp)
    updates["R_hp"].append(R_hp)
    updates["Emax_lv"].append(Emax_lv)
    updates["Emax_rv"].append(Emax_rv)
    updates["I"].append(I)
    updates["phi_met"].append(phi_met)
    updates["Nt"].append(Nt)
    updates["Vu_sv_change"].append(Vu_sv_change)
    updates["prev_flat_bit"].append(prev_flat_bit)
    updates["Pa_O2"].append(Pa_O2)
    updates["HR"].append(HR)
    updates["xb_O2"].append(xb_O2)
    updates["T"].append(T)
    updates["Cvb_O2"].append(Cvb_O2)
    updates["xb_CO2"].append(xb_CO2)

    # updates["t_eval2"] = updates["t_eval2"][1:]

    return [dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
            dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
            dxM_dt, dx_met_dt]