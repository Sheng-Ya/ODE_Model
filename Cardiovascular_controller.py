import math
import numpy as np
from Parameters import Parameters


def frac(x):
    return x - math.floor(x)


def extract_mean(buffer, idx_in_2, idx_in_1):
    if idx_in_2 <= idx_in_1:
        values = buffer[idx_in_2:idx_in_1 + 1]
    else:
        values = np.concatenate([buffer[idx_in_2:], buffer[:idx_in_1 + 1]])
    return np.mean(values)


def cardiovascular_controller(t, state, params, all_time, exp_inputs, heart_inputs, resp_control_inputs, gas_exchange_inputs, updates, num_removed, t_start, time_saved, i, BUFFER_LIMIT):
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

    if t == t_start:
        heart_index = i % BUFFER_LIMIT
        gas_index = i % BUFFER_LIMIT
        resp_control_index = i % BUFFER_LIMIT
    else:
        heart_index = (i - num_removed - 1) % BUFFER_LIMIT
        gas_index = (i - num_removed - 1) % BUFFER_LIMIT
        resp_control_index = (i - num_removed - 1) % BUFFER_LIMIT

    gas_index = 0
    resp_control_index = 0


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

        # Other inputs
    MRTCO2 = gas_exchange_inputs["MRTCO2_store"][gas_index]
    Pa_O2 = gas_exchange_inputs["Pa_O2_store"][gas_index]
    Pa_CO2 = gas_exchange_inputs["Pa_CO2_store"][gas_index]
    Ca_O2 = gas_exchange_inputs["Ca_O2_store"][gas_index]

    MRTCO2_basal = MRTCO2_basal - MRBCO2

    VE_integral = resp_control_inputs["VE_integral_store"][resp_control_index]

    I = (MRTCO2 - MRTCO2_basal)/(AT - MRTCO2_basal)

    # deal with rejected steps
    if t - updates["finish_breath_time"][-1] < 0:
        updates["finish_breath_time"].pop()
        updates["Nd"] = updates["Nd"][:-5]

    a1, a2, tau, t1, t2 = exp_inputs["Nd"][-5:]
    prev_flat_bit = updates["prev_flat_bit_store"][gas_index]

    last_breath_time = t - updates["finish_breath_time"][-1]

    if last_breath_time % (t1 + t2) < t1:
        Nt = VE_integral - prev_flat_bit  # Take value minus previous flat bit
    else:
        Nt = 0  # Reset to zero
        prev_flat_bit = VE_integral

    ## CNS Ischemic Response

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

    # Other inputs
    P_sa = heart_inputs["P_sa_store"][heart_index] # cardiovascular controller was run after cardio was run with states appended/updated so it must take the nonupdated version
    dP_sa_dt = heart_inputs["dP_sa_dt_store"][heart_index]
    Q_bp = heart_inputs["Q_bp_store"][heart_index]
    Q_hp = heart_inputs["Q_hp_store"][heart_index]
    Q_rmp = heart_inputs["Q_rmp_store"][heart_index]
    Q_amp = heart_inputs["Q_amp_store"][heart_index]
    Wh_lv = heart_inputs["Wh_lv_store"][heart_index]
    Wh_rv = heart_inputs["Wh_rv_store"][heart_index]

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

    phi_ac = ((f_ac_max + f_ac_min * np.exp((Pa_O2 - PaO2_ac_n)/k_ac))/(1 + np.exp((Pa_O2 - PaO2_ac_n)/k_ac)) *
              (K * np.log(Pa_CO2/PaCO2_n) + f_acCO2_n))

    d_fac_dt = (phi_ac - f_ac) / tau_ac

    # afferent activity from Pulmonary Stretch Receptors constant parameters

    # Other inputs
    VT = resp_control_inputs["VT_store"][resp_control_index]

    phi_ap = G_ap * VT
    df_ap_dt = (phi_ap - f_ap)/tau_ap


    ## Efferent Pathways constant parameters


    Y_sh = (Ysh_min + Ysh_max * np.exp((I - Io_sh)/kcc_sh)) / (1 + np.exp((I - Io_sh)/kcc_sh))
    f_ash = Wt_sh * Nt + Wb_sh * f_ab + Wc_sh * f_ac + Wp_sh * f_ap - theta_sh
    f_sh = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash) + Y_sh))

    Y_sp = (Ysp_min + Ysp_max * np.exp((I - Io_sp) / kcc_sp)) / (1 + np.exp((I - Io_sp) / kcc_sp))
    f_asp = Wt_sp * Nt + Wb_sp * f_ab + Wc_sp * f_ac + Wp_sp * f_ap - theta_sp
    f_sp = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp) + Y_sp))

    Y_sv = (Ysv_min + Ysv_max * np.exp((I - Io_sv) / kcc_sv)) / (1 + np.exp((I - Io_sv) / kcc_sv))
    f_asv = Wt_sv * Nt + Wb_sv * f_ab + Wc_sv * f_ac + Wp_sv * f_ap - theta_sv
    f_sv = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv) + Y_sv))


    Y_v = (Yv_min + Yv_max * np.exp((I - Io_v) / kcc_v)) / (1 + np.exp((I - Io_v) / kcc_v))
    first_term = (fev_o + fev_inf * np.exp((f_ab - fab_o)/kev)) / (1 + np.exp((f_ab - fab_o)/kev))
    f_v = first_term - Wt_v * Nt - Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v
    # f_v1 = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v # changed


    f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history = [exp_inputs[key] for key in
                                                                              ["f_sp_store", "f_sh_store",
                                                                               "f_v_store", "f_sv_store",
                                                                               "phi_met_store"]]

    # added the below to get f_sp_delay from previous iterations.
    delay_time2 = t - 2
    if delay_time2 >= t_start:

        # Find index in wrapped all_time array
        if delay_time2 >= all_time[0]:
            # No wrap-around
            delay_index = np.searchsorted(all_time[:(heart_index + 1)], delay_time2, side='right') - 1
        else:
            # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[(heart_index + 1):], delay_time2, side='right') - 1
            delay_index = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT


        f_sp_delay2 = f_sp_history[delay_index]
        f_sh_delay2 = f_sh_history[delay_index]
    else:
        f_sp_delay2 = 3.97
        f_sh_delay2 = 3.8576 #(f_shIC)

    delay_time5 = t - 5
    if delay_time5 >= t_start:

        if delay_time5 >= all_time[0]:
            # No wrap-around
            delay_index = np.searchsorted(all_time[:heart_index + 1], delay_time5, side='right') - 1
        else:
            # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[heart_index + 1:], delay_time5, side='right') - 1
            delay_index = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT


        f_sv_delay5 = f_sv_history[delay_index]

    else:
        f_sv_delay5 = 3.97
        # f_sv_delay5 = np.mean(f_sv_history)

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

    Vu_ev_every = max(Vu_ev_change + Vu_ev0, 0)
    Vu_sv_every = max(Vu_sv_change + Vu_sv0, 0)
    Vu_rmv_every = max(Vu_rmv_change + Vu_rmv0, 0)
    Vu_amv_every = max(Vu_amv_change + Vu_amv0, 0)

    Emax_lv_every = Emax_lv_change + Emax_lv0
    Emax_rv_every = Emax_rv_change + Emax_rv0

    # heart period constants

    delay_time0_2 = t - DT_v
    if delay_time0_2 >= t_start:

        if delay_time0_2 >= all_time[0]:
            # No wrap-around
            delay_index = np.searchsorted(all_time[:heart_index + 1], delay_time0_2, side='right') - 1
        else:
            # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[heart_index + 1:], delay_time0_2, side='right') - 1
            delay_index = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT

        f_v_delay0_2 = f_v_history[delay_index]
    else:
        if t == 0:
            f_v_delay0_2 = f_v
        else:
            f_v_delay0_2 = 4.2748 # np.mean(f_v_history), f_v_IC


    sigma_Ts = GT_s * np.log(max(f_sh_delay2, fes_min) - fes_min + 1)

    d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts
    # d_Ts_change_dt = 0

    sigma_Tv = GT_v * f_v_delay0_2
    d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv
    # d_Tv_change_dt = 0

    T = Ts_change + Tv_change + T0

    HR_every = 1 / T

    ## Blood Flow Local Control
    # Cerebral Blood Flow constant parameters

    G_bp = (1 / R_bpn) * (1 + xb_O2 + xb_CO2)
    R_bp = 1 / G_bp
    Cvb_O2 = Ca_O2 - MO2_bp / Q_bp
    dxb_O2_dt = (- xb_O2 - gb_O2 * (Cvb_O2 - Cvb_O2_n)) / tau_O2
    numerator = A + (B / (1 + C * np.exp(D * np.log10(Pa_CO2))))
    denominator = A + (B / (1 + C * np.exp(D * np.log10(PaCO2_n))))
    phi_b = numerator / denominator - 1

    dxb_CO2_dt = (- xb_CO2 - phi_b) / tau_CO2

    # Coronary and Resting Muscle Blood Flow constant parameters

    # other inputs

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

    R_amp = R_amp_n / (1 + xam_O2 + x_met)

    MO2_amp = MO2_ampn * (1 + xM)
    Cvam_O2 = Ca_O2 - MO2_amp / Q_amp

    dxam_O2_dt = (- xam_O2 - gam_O2 * (Cvam_O2 - Cvam_O2_n)) / tau_O2

    dxM_dt = (- xM + gM * I) / tau_M

    phi_met = (phi_min + phi_max * np.exp((I - Io_met) / kmet)) / (1 + np.exp((I - Io_met) / kmet))

    delay_time_met = t - 4
    if delay_time_met >= t_start:

        # Find index in wrapped all_time array
        if delay_time_met >= all_time[0]:
            # No wrap-around
            delay_index = np.searchsorted(all_time[:heart_index + 1], delay_time_met, side='right') - 1
        else:
            # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[heart_index + 1:], delay_time_met, side='right') - 1
            delay_index = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT


        phi_met_delay = phi_met_history[delay_index]
    else:
        phi_met_delay = phi_met


    dx_met_dt = (- x_met + phi_met_delay) / tau_met

    if t == 0:
        HR = HR_every
        Vu_ev = Vu_ev_every
        Vu_sv = Vu_sv_every
        Vu_rmv = Vu_rmv_every
        Vu_amv = Vu_amv_every
        Emax_lv = Emax_lv_every
        Emax_rv = Emax_rv_every
        check = f_ab
    else:
        HR = updates["HR_store"][heart_index]
        Vu_ev = updates["Vu_ev_store"][heart_index]
        Vu_sv = updates["Vu_sv_store"][heart_index]
        Vu_rmv = updates["Vu_rmv_store"][heart_index]
        Vu_amv = updates["Vu_amv_store"][heart_index]
        Emax_lv = updates["Emax_lv_store"][heart_index]
        Emax_rv = updates["Emax_rv_store"][heart_index]
        check = updates["check_store"][heart_index]



    # if num_removed > 0:
    #     keys = [
    #         "HR", "Vu_ev", "Vu_sv", "Vu_rmv", "Vu_amv", "Emax_lv", "Emax_rv",
    #         "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "I", "prev_flat_bit",
    #         "f_sp", "f_sh", "f_v", "f_sv", "phi_met"
    #         # "f_ab", "f_ac", "Nt", "T", "Cvb_O2", "Wh", "xamO2", "Cvam_O2", "MO2_amp", "xM", "f_asv", "f_asp", "f_ash", "theta_change_O2_sp",
    #         # "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh", "theta_change_CO2_sh"
    #     ]
    #     keys2 = [
    #         "HR1", "Vu_ev1", "Vu_sv1", "Vu_rmv1", "Vu_amv1", "Emax_lv1", "Emax_rv1"
    #     ]
    #     for key in keys:
    #         updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)
    #     for key in keys2:
    #         del updates[key][-num_removed:]
    #
    #     i = i - num_removed


    if num_removed > 0:
        keys2 = [
            "HR1", "Vu_ev1", "Vu_sv1", "Vu_rmv1", "Vu_amv1", "Emax_lv1", "Emax_rv1"
        ]
        for key in keys2:
            del updates[key][-num_removed:]




    if heart_index <= 1:
        time_since_beat1 = updates["time_since_beat_store"][heart_index]
        time_since_beat2 = updates["time_since_beat_store"][heart_index]
    else:
        time_since_beat1 = updates["time_since_beat_store"][(heart_index + 1) % BUFFER_LIMIT]
        time_since_beat2 = updates["time_since_beat_store"][heart_index]


    # update after every heartbeat
    if time_since_beat1 != time_since_beat2:
        if time_since_beat2 >= all_time[0]:
            # No wrap-around
            idx_in_2 = np.searchsorted(all_time[:heart_index + 1], time_since_beat2, side='right') - 1
        else:
            # Wrap-around
            idx_in_sorted2 = np.searchsorted(all_time[heart_index + 1:], time_since_beat2, side='right') - 1
            idx_in_2 = (idx_in_sorted2 + heart_index + 1) % BUFFER_LIMIT

        idx_in_1 = heart_index

        HR = extract_mean(updates["HR_every_store"], idx_in_2, idx_in_1)
        Vu_ev = extract_mean(updates["Vu_ev_every_store"], idx_in_2, idx_in_1)
        Vu_sv = extract_mean(updates["Vu_sv_every_store"], idx_in_2, idx_in_1)
        Vu_rmv = extract_mean(updates["Vu_rmv_every_store"], idx_in_2, idx_in_1)
        Vu_amv = extract_mean(updates["Vu_amv_every_store"], idx_in_2, idx_in_1)
        Emax_lv = extract_mean(updates["Emax_lv_every_store"], idx_in_2, idx_in_1)
        Emax_rv = extract_mean(updates["Emax_rv_every_store"], idx_in_2, idx_in_1)
        check = extract_mean(updates["check_store_all"], idx_in_2, idx_in_1)





    # update values needed in other systems
    for key, new_value in zip(
            [  # Cardio inputs
                "HR_store", "Vu_ev_store", "Vu_sv_store", "Vu_rmv_store", "Vu_amv_store",
                "Emax_lv_store", "Emax_rv_store", "R_ep_store", "R_amp_store", "R_rmp_store",
                "R_sp_store", "R_bp_store", "R_hp_store", "I_store", "f_sp_store", "f_sh_store",
                "f_v_store", "f_sv_store", "phi_met_store", "HR_every_store", "Vu_ev_every_store",
                "Vu_sv_every_store", "Vu_rmv_every_store", "Vu_amv_every_store", "Emax_lv_every_store",
                "Emax_rv_every_store",

                # Needed in cardio controller
                "prev_flat_bit_store",
                "check_store_all", "check_store"],

            [HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
             Emax_lv, Emax_rv, R_ep, R_amp, R_rmp,
             R_sp, R_bp, R_hp, I, f_sp, f_sh, f_v, f_sv, phi_met, HR_every, Vu_ev_every, Vu_sv_every,
             Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,
             prev_flat_bit, f_ab, check]
    ):
        updates[key][((i - num_removed) % BUFFER_LIMIT)] = new_value





    #         # just for plotting purposes
    # if ((t % time_saved) < 0.001 or (time_saved - (t % time_saved)) < 0.001) and num_removed == 0:
    keys_and_values = zip(
        [   # Cardio inputs
            "HR", "Vu_ev", "Vu_sv", "Vu_rmv", "Vu_amv", "Emax_lv", "Emax_rv",
            "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "I", "f_sp", "f_sh", "f_v", "f_sv", "Nt", "f_ab",
            "f_ac", "f_ap", "Tv_change", "Ts_change", "HR_check", "f_sh_delay2", "f_v_delay02", "sigma_Ts", "sigma_Tv",
            "theta_sp", "theta_sh", "theta_sv", "theta_v", "check_store_all_time"
            ],

        [   # Corresponding values
            HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv, Emax_lv, Emax_rv,
            R_ep, R_amp, R_rmp, R_sp, R_bp, R_hp, I, f_sp, f_sh, f_v, f_sv, Nt, f_ab, f_ac, f_ap, Tv_change, Ts_change, HR_every,
            f_sh_delay2, f_v_delay0_2, sigma_Ts, sigma_Tv, P_sa, exp_arg, f_sv, f_v, check])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value


    return [dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
            dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
            dxM_dt, dx_met_dt]