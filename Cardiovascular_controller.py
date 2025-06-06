import bisect
import math
import numpy as np
from Parameters import Parameters as params



def frac(x):
    return x - math.floor(x)


def cardiovascular_controller(t, state, params, time_history, exp_inputs, heart_inputs, resp_control_inputs, gas_exchange_inputs, updates, num_removed, i, t_start, previous_Selected_Conditions):
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
        heart_index = i
        gas_index = i
        # resp_control_index = 0
        resp_control_index = i
        # gas_index = 0
    # elif num_removed > 0:
    #     heart_index = i - num_removed - 1
    #     # gas exchange variables have not been removed yet
    #     gas_index = i - 1
    #     # resp_control_index = 0
    #     resp_control_index = i - 1
    #     # gas_index = 0
    else:
        heart_index = i - 1 - num_removed
        gas_index = i - 1 - num_removed
        # resp_control_index = 0
        resp_control_index = i - 1 - num_removed
        # gas_index = 0


    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v,
     Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv,
     Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp,
     GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, tau_Emax_lv, tau_Emax_rv, tau_Ramp,
     tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, AT, MRTCO2_basal,
     g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, tau_cc, tau_isc, theta_shn, theta_spn,
     theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, tau_p, tau_z, f_acCO2_n, f_ac_max, f_ac_min, k_ac,
     K_H, PaO2_ac_n, tau_ac, G_ap, tau_ap, DT_v, GT_s, GT_v, T0, tau_Ts, tau_Tv, A, B, C, D, Cvb_O2_n, gb_O2, MO2_bp, R_bpn,
     tau_CO2, tau_O2, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, tau_w, W_hn, Cvam_O2_n,
     gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min, tau_M, tau_met) = [params[k] for k in ["fab_o", "fes_o",
    "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv",
    "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh",
    "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv",
    "Wt_v", "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
    "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep",
    "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "AT",
    "MRTCO2_basal", "gccsh", "gccsp", "gccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "tau_cc",
    "tau_isc", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab",
    "P_n", "tau_p", "tau_z", "f_acCO2_n", "f_ac_max", "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "tau_ac", "G_ap", "tau_ap",
    "DT_v", "GT_s", "GT_v", "T0", "tau_Ts", "tau_Tv", "A", "B", "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "tau_CO2",
    "tau_O2", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "tau_w",
    "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet", "MO2_ampn", "phi_max", "phi_min", "tau_M", "tau_met"]]

        # Other inputs
    MRTCO2 = gas_exchange_inputs["MRTCO2"][gas_index]
    Pa_O2 = gas_exchange_inputs["Pa_O2"][gas_index]
    Pa_CO2 = gas_exchange_inputs["Pa_CO2"][gas_index]
    Ca_O2 = gas_exchange_inputs["Ca_O2"][gas_index]

    MRTCO2_basal = MRTCO2_basal - params["MRBCO2"]

    VE_integral = resp_control_inputs["VE_integral"][resp_control_index]

    I = (MRTCO2 - MRTCO2_basal)/(AT - MRTCO2_basal)

    # deal with rejected steps
    if t - updates["finish_breath_time"][-1] < 0:
        updates["finish_breath_time"].pop()
        updates["Nd"] = updates["Nd"][:-5]

    a1, a2, tau, t1, t2 = exp_inputs["Nd"][-5:]
    prev_flat_bit = updates["prev_flat_bit"][gas_index]

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

    # if t > 0.845:
    #     AAA = list(gas_exchange_inputs["theta_change_O2_sp"])

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
    P_sa = heart_inputs["P_sa"][heart_index] # cardiovascular controller was run after cardio was run with states appended/updated so it must take the nonupdated version
    dP_sa_dt = heart_inputs["dP_sa_dt"][heart_index]
    Q_bp = heart_inputs["Q_bp"][heart_index]
    Q_hp = heart_inputs["Q_hp"][heart_index]
    Q_rmp = heart_inputs["Q_rmp"][heart_index]
    Q_amp = heart_inputs["Q_amp"][heart_index]
    Wh_lv = heart_inputs["Wh_lv"][heart_index]
    Wh_rv = heart_inputs["Wh_rv"][heart_index]

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
    VT = resp_control_inputs["VT"][resp_control_index]

    phi_ap = G_ap * VT
    df_ap_dt = (phi_ap - f_ap)/tau_ap


    ## Efferent Pathways constant parameters


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
    f_v = first_term - Wt_v * Nt - Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v
    # f_v1 = first_term - Wt_v * Nt + Wc_v * f_ac + Wp_v * f_ap - theta_v + Y_v # changed


    f_sp_history, f_sh_history, f_v_history, f_sv_history, phi_met_history = [exp_inputs[key] for key in
                                                                              ["f_sp", "f_sh",
                                                                               "f_v", "f_sv",
                                                                               "phi_met"]]

    # added the below to get f_sp_delay from previous iterations.
    delay_time2 = t - 2
    if delay_time2 >= t_start:
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, delay_time2) - 1
        f_sp_delay2 = f_sp_history[delay_index]
        f_sh_delay2 = f_sh_history[delay_index]
    else:
        if t == 0:
            f_sp_delay2 = f_sp
            f_sh_delay2 = f_sh
        else:
            if t_start != 0: # this is for the previous run with delays recorded to be used for the current run
                delay_index = bisect.bisect_right(previous_Selected_Conditions["time_history"], delay_time2) - 1
                f_sp_delay2 = previous_Selected_Conditions["f_sp"][delay_index]
                f_sh_delay2 = previous_Selected_Conditions["f_sh"][delay_index]
            else:
                # f_sp_delay2 = np.mean(f_sp_history)
                f_sp_delay2 = 3.97
                f_sh_delay2 = 3.8576 #(f_shIC)

    delay_time5 = t - 5
    if delay_time5 >= t_start:
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, delay_time5) - 1
        f_sv_delay5 = f_sv_history[delay_index]
    else:
        if t == 0:
            f_sv_delay5 = f_sv
        else:
            if t_start != 0:
                delay_index = bisect.bisect_right(previous_Selected_Conditions["time_history"], delay_time5) - 1
                f_sv_delay5 = previous_Selected_Conditions["f_sv"][delay_index]
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

    Vu_ev1 = max(Vu_ev_change + Vu_ev0, 0)
    Vu_sv1 = max(Vu_sv_change + Vu_sv0, 0)
    Vu_rmv1 = max(Vu_rmv_change + Vu_rmv0, 0)
    Vu_amv1 = max(Vu_amv_change + Vu_amv0, 0)

    Emax_lv1 = Emax_lv_change + Emax_lv0
    Emax_rv1 = Emax_rv_change + Emax_rv0

    # heart period constants

    delay_time0_2 = t - DT_v
    if delay_time0_2 >= t_start:
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, delay_time0_2) - 1
        f_v_delay0_2 = f_v_history[delay_index]
    else:
        if t == 0:
            f_v_delay0_2 = f_v
        else:
            if t_start != 0:
                delay_index = bisect.bisect_right(previous_Selected_Conditions["time_history"], delay_time0_2) - 1
                f_v_delay0_2 = previous_Selected_Conditions["f_v"][delay_index]
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
        # Find the index for delay_time in time_history
        delay_index = bisect.bisect_right(time_history, delay_time_met) - 1
        phi_met_delay = phi_met_history[delay_index]
    elif t_start != 0:
        delay_index = bisect.bisect_right(previous_Selected_Conditions["time_history"], delay_time_met) - 1
        phi_met_delay = previous_Selected_Conditions["phi_met"][delay_index]
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
        HR = updates["HR"][heart_index]
        Vu_ev = updates["Vu_ev"][heart_index]
        Vu_sv = updates["Vu_sv"][heart_index]
        Vu_rmv = updates["Vu_rmv"][heart_index]
        Vu_amv = updates["Vu_amv"][heart_index]
        Emax_lv = updates["Emax_lv"][heart_index]
        Emax_rv = updates["Emax_rv"][heart_index]




    if num_removed > 0:
        keys = [
            "HR", "Vu_ev", "Vu_sv", "Vu_rmv", "Vu_amv", "Emax_lv", "Emax_rv",
            "R_ep", "R_amp", "R_rmp", "R_sp", "R_bp", "R_hp", "I", "prev_flat_bit",
            "f_sp", "f_sh", "f_v", "f_sv", "phi_met"
            # "f_ab", "f_ac", "Nt", "T", "Cvb_O2", "Wh", "xamO2", "Cvam_O2", "MO2_amp", "xM", "f_asv", "f_asp", "f_ash", "theta_change_O2_sp",
            # "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh", "theta_change_CO2_sh"
        ]
        keys2 = [
            "HR1", "Vu_ev1", "Vu_sv1", "Vu_rmv1", "Vu_amv1", "Emax_lv1", "Emax_rv1"
        ]
        for key in keys:
            updates[key][(i - num_removed): (i + 1)] = np.full((num_removed + 1,), 1e6)
        for key in keys2:
            del updates[key][-num_removed:]

        i = i - num_removed


    if heart_index <= 1:
        time_since_beat1 = updates["time_since_beat"][heart_index]
        time_since_beat2 = updates["time_since_beat"][heart_index]
    else:
        time_since_beat1 = updates["time_since_beat"][heart_index]
        time_since_beat2 = updates["time_since_beat"][heart_index - 1]

    # update after every heartbeat
    if time_since_beat1 != time_since_beat2 and num_removed == 0:
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


        # update history
    # updates["HR1"] = np.append(updates["HR1"], HR1)
    # updates["Vu_ev1"] = np.append(updates["Vu_ev1"], Vu_ev1)
    # updates["Vu_sv1"] = np.append(updates["Vu_sv1"], Vu_sv1)
    # updates["Vu_rmv1"] = np.append(updates["Vu_rmv1"], Vu_rmv1)
    # updates["Vu_amv1"] = np.append(updates["Vu_amv1"], Vu_amv1)
    # updates["Emax_lv1"] = np.append(updates["Emax_lv1"], Emax_lv1)
    # updates["Emax_rv1"] = np.append(updates["Emax_rv1"], Emax_rv1)
    updates["HR1"].append(HR1)
    updates["Vu_ev1"].append(Vu_ev1)
    updates["Vu_sv1"].append(Vu_sv1)
    updates["Vu_rmv1"].append(Vu_rmv1)
    updates["Vu_amv1"].append(Vu_amv1)
    updates["Emax_lv1"].append(Emax_lv1)
    updates["Emax_rv1"].append(Emax_rv1)

    # cardio inputs
    updates["HR"][i] = HR
    updates["Vu_ev"][i] = Vu_ev
    updates["Vu_sv"][i] = Vu_sv
    updates["Vu_rmv"][i] = Vu_rmv
    updates["Vu_amv"][i] = Vu_amv
    updates["Emax_lv"][i] = Emax_lv
    updates["Emax_rv"][i] = Emax_rv

    updates["R_ep"][i] = R_ep
    updates["R_amp"][i] = R_amp
    updates["R_rmp"][i] = R_rmp
    updates["R_sp"][i] = R_sp
    updates["R_bp"][i] = R_bp
    updates["R_hp"][i] = R_hp
    updates["I"][i] = I

    # needed in cardio controller
    updates["prev_flat_bit"][i] = prev_flat_bit

    # save for delay
    updates["f_sp"][i] = f_sp
    updates["f_sh"][i] = f_sh
    updates["f_v"][i] = f_v
    updates["f_sv"][i] = f_sv
    updates["phi_met"][i] = phi_met


    # just for plotting purposes
    updates["f_ac"][i] = f_ac
    updates["f_ab"][i] = f_ab
    updates["f_ap"][i] = f_ap
    updates["Nt"][i] = Nt
    updates["Cvb_O2"][i] = Cvb_O2
    updates["Wh"][i]= Wh
    updates["xamO2"][i] = xam_O2
    updates["Cvam_O2"][i] = Cvam_O2
    updates["MO2_amp"][i] = MO2_amp
    updates["xM"][i] = xM
    updates["f_asv"][i] = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv)
    updates["f_asp"][i] = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp)
    updates["f_ash"][i] = fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash)

    updates["theta_change_O2_sp"][i] = theta_change_O2_sp
    updates["theta_change_CO2_sp"][i] = theta_change_CO2_sp
    updates["theta_change_O2_sv"][i] = theta_change_O2_sv
    updates["theta_change_CO2_sv"][i] = theta_change_CO2_sv
    updates["theta_change_O2_sh"][i] = theta_change_O2_sh
    updates["theta_change_CO2_sh"][i] = theta_change_CO2_sh

    updates["sigma_Tv"][i] = sigma_Tv
    updates["sigma_Ts"][i] = sigma_Ts
    updates["Y_v"][i] = Y_v


    return [dtheta_change_O2_sp_dt, dtheta_change_CO2_sp_dt, dtheta_change_O2_sv_dt, dtheta_change_CO2_sv_dt,
            dtheta_change_O2_sh_dt, dtheta_change_CO2_sh_dt, dP_tilda_dt, d_fac_dt, df_ap_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dxb_O2_dt, dxb_CO2_dt, dxh_O2_dt, dxh_CO2_dt, dWh_dt, dxrm_O2_dt, dxrm_CO2_dt, dxam_O2_dt,
            dxM_dt, dx_met_dt]