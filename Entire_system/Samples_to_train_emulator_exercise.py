import os
import copy
import signal
import torch
from scipy.stats import qmc

import numpy as np
from SALib import ProblemSpec
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective

from tqdm import tqdm
import tqdm_joblib

from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from All_derivatives_njit import model_derivatives
from fixed_params import Parameters as Old_Parameters

from Initial_Conditions_after_running_again import Initial_Conditions
from All_Next_Conditions import Next_Conditions


target_values = np.arange(0, 10000, 10)
BUFFER_LIMIT = 20000

max_time = 400 # Maximum time limit to avoid infinite loops

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1:  # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            # num_removed = 6
            index = -1  # Set a default value for safety

            # Iterating through the buffer in circular order
            for j in range(BUFFER_LIMIT):
                logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
                if all_time[logical_index] < t:
                    index = (logical_index + 1) % BUFFER_LIMIT
                    break

            num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (
                        actual_index - index)

            for j in range(num_removed):
                all_time[(index + j) % BUFFER_LIMIT] = 0

            # if num_removed != 6:
            #     print(f"num_removed should be 6, got {num_removed}")
            # raise ValueError(f"num_removed should be 6, got {num_removed}")
        else:
            num_removed = 0
    else:
        num_removed = 0

    # Indices for slicing
    idx_resp_contr = num_cardio + num_cardio_control + num_gas + num_resp_control

    # Extract each subsystem's state variables
    resp_contr_state = Initial_Conditions_numpy[:idx_resp_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    derivatives_all = model_derivatives(t, resp_contr_state, Initial_Conditions_dict, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters)
    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

    # AA = list(Initial_Conditions_dict["all_time"])
    # AAAAAAA = list(Initial_Conditions_dict["check_time"])

    # # Debugging check for progress
    # if t != 0:
    #     diff = np.abs(t - target_values)
    #     if np.any(diff < 0.0001):
    #         print(t)

    return derivatives_all

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
                         "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt"]
IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)
num_cardio = len(required_cardio_keys)

# cardiovascular controller
required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv",
                         "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change",
                         "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                         "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                         "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2", "xam_O2",
                         "xM", "x_met", "P_n_current"]

IC_cardio_contr = np.array([Initial_Conditions[key] for key in required_cardio_control_keys], dtype=float)
num_cardio_control = len(required_cardio_control_keys)

# resp control ventilation
required_resp_control_keys = ["VE_integral"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)


# # resp mechanics
# required_resp_mech_keys = ["Vflow_ua"]
# IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
# num_resp_mech = len(required_resp_mech_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))

def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
    try:
        dt = 0.001
        bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
        tolerance = 0.0001

        VAflow_vals = np.linspace(0.06, 1, 200)
        VAflow_repeated = np.repeat(VAflow_vals, 3)

        VD = GV_dead * VAflow_repeated + V0_dead

        optimal_t1 = []
        optimal_t2 = []
        initial_guess = [t1, t2]

        for idx, VAflow in enumerate(VAflow_repeated):
            VD_volume = VD[idx]
            required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]

            res = minimize(objective, x0= np.array(initial_guess[-2:]),
                           args=(required_params, VAflow, VD_volume, dt, tolerance), method='COBYLA', bounds=bounds)
            t1_opt, t2_opt = res.x
            optimal_t1.append(t1_opt)
            optimal_t2.append(t2_opt)
            initial_guess.extend(res.x)


        # Convert to arrays for indexing
        VAflow_clean = np.array(VAflow_repeated)
        t1_clean = np.array(optimal_t1)
        t2_clean = np.array(optimal_t2)

        # Fit a polynomial (or linear)
        t1_poly = np.poly1d(np.polyfit(VAflow_clean, t1_clean, deg=6))
        t2_poly = np.poly1d(np.polyfit(VAflow_clean, t2_clean, deg=6))

        c0, c1, c2, c3, c4, c5, c6 = t1_poly.c[0], t1_poly.c[1], t1_poly.c[2], t1_poly.c[3], t1_poly.c[4], t1_poly.c[5], t1_poly.c[6]
        d0, d1, d2, d3, d4, d5, d6 = t2_poly.c[0], t2_poly.c[1], t2_poly.c[2], t2_poly.c[3], t2_poly.c[4], t2_poly.c[5], t2_poly.c[6]

        # print("Best fit equation for t1:", t1_poly)
        # print("Best fit equation for t2:", t2_poly)
    except:
        return 0,0,0,0,0,0,0,0,0,0,0,0,0,0

    return c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6


def simulate_cpu(Current_Parameters, local_updates, old_parameters):
    # local_updates = {key: copy.deepcopy(value) for key, value in storage.items()}

    IC_current = IC_overall.copy()
    t_span = [0, max_time]


    # Cardio parameters
    (A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n,
    C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    kr_am) = (
    Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
    ["A_im", "Tc", "T_im", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa",
     "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv",
     "Emax_la", "P0_la", "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa",
     "R_sa", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n", "C_jp",
     "R_ev_n", "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv",
     "kr_am"])

    # Cardio controller parameters
    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
    kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
    Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv,
    GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv,
    theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max,
    f_acCO2_n, f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D,
    Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
    MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min) = \
    [Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
     ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o",
      "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max",
      "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp",
      "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
      "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
      "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp", "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh",
      "PO2_sp", "PO2_sv", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp", "x_sv",
      "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max", "f_acCO2_n", "f_ac_max", "f_ac_min",
      "k_ac", "K_H", "PaO2_ac_n", "G_ap", "DT_v", "GT_s", "GT_v", "T0", "A", "B", "C", "D",
      "Cvb_O2_n", "gb_O2", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
      "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met",
      "kmet", "MO2_ampn", "phi_max", "phi_min"]]

    # Gas exchange and mixing
    (a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm,
     P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp, MRTCO2_basal, MRTO2_basal,
     MRCO2, MRO2, s) = (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in [
    "a2", "alpha2", "beta2", "C2", "K2", "PACO2_Delay_IC",
    "PAO2_Delay_IC", "P_atm", "P_ws", "Z", "dc", "KCCO2", "MRBCO2",
    "MO2_bp", "MRTCO2_basal", "MRTO2_basal", "MRCO2", "MRO2", "s"])

    # Resp control
    (GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
   "V0_dead", "VA_rest", "lambda1", "lambda2", "n", "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"])

    # added params
    (Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
    Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
    R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vvc_max, Vvc_min, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
     scale_param1, scale_param2, scale_param3, scale_param4, scale_param5, scale_param6, scale_param7, scale_param8,
     shift_param1, shift_param2, shift_param3, shift_param4, Pa_O2_lower, rise_time_atr, fall_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P
     ) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["Kp_ao", "Kf_ao", "Kb_ao",
    "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po", "Kf_po", "Kb_po", "Kv_po",
    "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2", "R_po", "R_mi", "R_tr", "R_ao",
    "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal", "Vu_sa", "V_tot", "Vu_jp",
    "Vu_bv", "Vu_hv", "Vu_vc", "Vvc_max", "Vvc_min", "Vu_pa", "Vu_pp", "Vu_pv",
    "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp",
    "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc",
    "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met",
    "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v",
    "Dmet", "Fi_CO2", "Fi_O2", "Ta", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
    "scale_param1", "scale_param2", "scale_param3", "scale_param4", "scale_param5", "scale_param6", "scale_param7",
    "scale_param8", "shift_param1", "shift_param2", "shift_param3", "shift_param4", "Pa_O2_lower", "rise_time_atr",
    "fall_time_atr", "rise_time_ven", "fall_time_ven", "ahead1", "theta_min", "delta_P"])

    # determine the correct breathing profile
    # c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6 = (56.68997590915653, -202.59647354823105, 288.8670155008632, -209.7703017034145, 82.28269589051426, -17.368480186780154, 2.5893052287384397, 89.14188202682894, -308.69610281589763, 429.74939918039985, -308.8054292147809, 122.49308640665272, -26.978019539657186, 4.001791662703984)
    c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6 = minimise_breathing(1.5,
    1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao)

    if all(x == 0 for x in [c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6]):
        # Integration failed or early termination
        print("fail_breathing")
        return [0.0]

    Input_Parameters = [A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa,
     C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra, C_sa,
     L_sa, R_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv,
     C_bv, C_hv, C_rmv, C_amv, kr_am, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv,
     Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v,
     Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0,
     Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0,
     R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, theta_shn, theta_spn,
     theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max, f_acCO2_n, f_ac_max, f_ac_min,
     k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D, Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2,
     grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max,
     phi_min, a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm, P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp,
     MRTCO2_basal, MRTO2_basal, MRCO2, MRO2, s, GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2,
     n, Pmax, Pmax_dot, E_rs, R_rs, P_ao, c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6,
     # added params
     Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
     Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
     R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
     Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vvc_max, Vvc_min, Vu_pa, Vu_pp,
     Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
     tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
     tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
     DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
     scale_param1, scale_param2, scale_param3, scale_param4, scale_param5, scale_param6, scale_param7, scale_param8,
     shift_param1, shift_param2, shift_param3, shift_param4, Pa_O2_lower, rise_time_atr, fall_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P]

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        t_span,
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters)
    )


    if ODE_solution.status == -1:
        # Integration failed or early termination
        print("fail")
        return [0.0]

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT
    # change
    # P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
    # peaks, _ = find_peaks(P_sa, distance=int(500))
    # # # troughs, _ = find_peaks(-P_sa, distance=int(500))
    # # #
    # # # last_10_troughs_P_sa = troughs[-10:-1]
    # # # last_10_min_P_sa = P_sa[last_10_troughs_P_sa]
    # # #
    # last_10_peaks_P_sa = peaks[-10:-1]
    # last_10_max_P_sa = P_sa[last_10_peaks_P_sa]
    #
    # V_lv = np.concatenate((local_updates["V_lv_store"][i_buffer:], local_updates["V_lv_store"][:i_buffer]))
    # peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    # troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)
    #
    # last_10_troughs_V_lv = troughs[-10:-1]
    # last_10_min_V_lv = V_lv[last_10_troughs_V_lv]
    #
    # last_10_peaks_V_lv = peaks[-10:-1]
    # last_10_max_V_lv = V_lv[last_10_peaks_V_lv]
    #
    # V_rv = np.concatenate((local_updates["V_rv_store"][i_buffer:], local_updates["V_rv_store"][:i_buffer]))
    # peaks, _ = find_peaks(V_rv, distance=int(500), prominence=1)
    # troughs, _ = find_peaks(-V_rv, distance=int(500), prominence=1)
    #
    # last_10_troughs_V_rv = troughs[-10:-1]
    # last_10_min_V_rv = V_rv[last_10_troughs_V_rv]
    #
    # last_10_peaks_V_rv = peaks[-10:-1]
    # last_10_max_V_rv = V_rv[last_10_peaks_V_rv]
    #
    P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
    peaks, _ = find_peaks(P_rv, distance=int(500), prominence=1)
    # troughs, _ = find_peaks(-P_rv, distance=int(500), prominence=1)
    #
    # last_10_troughs_P_rv = troughs[-10:-1]
    # last_10_min_P_rv = P_rv[last_10_troughs_P_rv]
    #
    last_10_peaks_P_rv = peaks[-10:-1]
    last_10_max_P_rv = P_rv[last_10_peaks_P_rv]
    #
    #
    # Get past 10 HR
    # HR = np.concatenate((local_updates["HR_store"][i_buffer:], local_updates["HR_store"][:i_buffer]))
    #
    # past_10_flat_segments = []
    # # Start from the end and track the current segment value
    # prev_value = None
    # for j in range(len(HR) - 1, -1, -1):
    #     current_value = HR[j]
    #     if current_value != prev_value:
    #         # New segment found
    #         past_10_flat_segments.append(current_value)
    #         prev_value = current_value
    #         if len(past_10_flat_segments) == 10:
    #             break
    #
    # # left atria
    # V_la = np.concatenate((local_updates["V_la_store"][i_buffer:], local_updates["V_la_store"][:i_buffer]))
    # peaks, _ = find_peaks(V_la, distance=int(1000), prominence=1)
    # troughs, _ = find_peaks(-V_la, distance=int(1000), prominence=1)
    #
    # last_10_troughs_V_la = troughs[-10:-1]
    # last_10_min_V_la = V_la[last_10_troughs_V_la]
    #
    # last_10_peaks_V_la = peaks[-10:-1]
    # last_10_max_V_la = V_la[last_10_peaks_V_la]
    #
    # P_la = np.concatenate((local_updates["P_la_store"][i_buffer:], local_updates["P_la_store"][:i_buffer]))
    # peaks, _ = find_peaks(P_la, distance=int(2000), prominence=1)
    # troughs, _ = find_peaks(-P_la, distance=int(2000), prominence=1)
    #
    # last_10_troughs_P_la = troughs[-10:-1]
    # last_10_min_P_la = P_la[last_10_troughs_P_la]
    #
    # last_10_peaks_P_la = peaks[-10:-1]
    # last_10_max_P_la = P_la[last_10_peaks_P_la]
    #
    # # right atria
    # V_ra = np.concatenate((local_updates["V_ra_store"][i_buffer:], local_updates["V_ra_store"][:i_buffer]))
    # peaks, _ = find_peaks(V_ra, distance=int(1000), prominence=1)
    # troughs, _ = find_peaks(-V_ra, distance=int(1000), prominence=1)
    #
    # last_10_troughs_V_ra = troughs[-10:-1]
    # last_10_min_V_ra = V_ra[last_10_troughs_V_ra]
    #
    # last_10_peaks_V_ra = peaks[-10:-1]
    # last_10_max_V_ra = V_ra[last_10_peaks_V_ra]

    # P_ra = np.concatenate((local_updates["P_ra_store"][i_buffer:], local_updates["P_ra_store"][:i_buffer]))
    # peaks, _ = find_peaks(P_ra, distance=int(2000), prominence=1)
    # # troughs, _ = find_peaks(-P_ra, distance=int(2000), prominence=1)
    #
    # # last_10_troughs_P_ra = troughs[-10:-1]
    # # last_10_min_P_ra = P_ra[last_10_troughs_P_ra]
    #
    # last_10_peaks_P_ra = peaks[-10:-1]
    # last_10_max_P_ra = P_ra[last_10_peaks_P_ra]

    # # get volume before atrial contraction
    # phi_atr = np.concatenate((local_updates["phi_atr_store"][i_buffer:], local_updates["phi_atr_store"][:i_buffer]))
    # # Find transitions: where phi_atr goes from 0 to >0
    # starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    # local_mins = starts[-10:]
    # last_10_b4_LA_atrial_contract = V_la[local_mins]
    # last_10_b4_RA_atrial_contract = V_ra[local_mins]
    #
    # # maximum ventricular pressure derivative
    # P_lv = np.concatenate((local_updates["P_lv_store"][i_buffer:], local_updates["P_lv_store"][:i_buffer]))
    # all_time = np.concatenate((local_updates["all_time"][i_buffer:], local_updates["all_time"][:i_buffer]))
    # dPmax_lv_dt1 = np.gradient(P_lv, all_time)
    # dPmax_lv_dt = savgol_filter(dPmax_lv_dt1, window_length=11, polyorder=3)
    # peaks, _ = find_peaks(dPmax_lv_dt, distance=int(1000), prominence=10)
    # last_10 = peaks[-10:-1]
    # last_10_max_P_lv_deriv = dPmax_lv_dt[last_10]
    #
    # P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
    #
    # dPmax_rv_dt1 = np.gradient(P_rv, all_time)
    # dPmax_rv_dt = savgol_filter(dPmax_rv_dt1, window_length=11, polyorder=3)
    # peaks, _ = find_peaks(dPmax_rv_dt, distance=int(1000), prominence=10)
    # last_10 = peaks[-10:-1]
    # last_10_max_P_rv_deriv = dPmax_rv_dt[last_10]
    #
    # tidal = np.concatenate((local_updates["tidal_store"][i_buffer:], local_updates["tidal_store"][:i_buffer]))
    # peaks, _ = find_peaks(tidal, distance=int(1000))
    # last_10_peaks_tidal = peaks[-1]
    # max_tidal = tidal[last_10_peaks_tidal]
    #
    # VAflow = np.concatenate((local_updates["VAflow_store"][i_buffer:], local_updates["VAflow_store"][:i_buffer]))
    # t1 = np.concatenate((local_updates["t1_store"][i_buffer:], local_updates["t1_store"][:i_buffer]))
    # t2 = np.concatenate((local_updates["t2_store"][i_buffer:], local_updates["t2_store"][:i_buffer]))
    # VD = GV_dead * VAflow[-1] + V0_dead
    # VDflow = (1 / (t1[-1] + t2[-1])) * VD
    # Minute_Ventilation = (VAflow[-1] + VDflow) * 60
    #
    # cardiac_output = np.mean(local_updates["Q_pp_store"])
    # Pa_O2 = np.mean(local_updates["Pa_O2_every_store"])
    # Pa_CO2 = np.mean(local_updates["Pa_CO2_every_store"])
    # change
    print(np.mean(last_10_max_P_rv))

    # A = IC_overall.copy()

    # IC_current = ODE_solution.y[:, -1]

    # return ([np.mean(past_10_flat_segments), np.mean(last_10_max_P_sa), np.mean(last_10_min_P_sa),
    #       np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(last_10_max_V_rv), np.mean(last_10_min_V_rv),
    #       np.mean(last_10_max_P_rv), np.mean(last_10_min_P_rv),
    #       np.mean(last_10_min_V_ra), np.mean(last_10_max_V_ra), np.mean(last_10_min_P_ra), np.mean(last_10_max_P_ra),
    #       np.mean(last_10_min_V_la), np.mean(last_10_max_V_la), np.mean(last_10_min_P_la), np.mean(last_10_max_P_la),
    #       np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
    #       np.mean(last_10_max_P_lv_deriv), np.mean(last_10_max_P_rv_deriv), max_tidal, Minute_Ventilation,
    #          cardiac_output, Pa_O2, Pa_CO2])#, IC_current, local_updates)
    # change
    return [np.mean(last_10_max_P_rv)]#, IC_current, local_updates)


#
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timeout")

def safe_simulate_cpu(params, storage, old_parameters, timeout=400):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        result = simulate_cpu(params, storage, old_parameters)
        signal.alarm(0)  # Cancel timeout
        return result
    except Exception:
        print("timeout")
        signal.alarm(0)  # Cancel timeout
        return ([0.0])


def parallel_simulations(param_samples, storage, n_jobs, chunk_size=2500, save_path='Result_lhcs.npy'):
    results_all = []

    # If file exists from previous run, remove it to start fresh
    if os.path.exists(save_path):
        os.remove(save_path)

    for i, chunk in enumerate(chunked(param_samples, chunk_size)):
        with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i + 1) * chunk_size}", total=len(chunk))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(safe_simulate_cpu)(params, copy.deepcopy(storage), Old_Parameters) for params in chunk)

        results_all.extend(results)

        # Optional: also accumulate in a single array
        np.save(save_path, np.array(results_all))  # full file overwritten

    return results_all


# def parallel_simulations(param_samples, storage, chunk_size=10, save_path='Result_DGSM_chunked.npy'):
#     results_all = []
#
#     # If file exists from previous run, remove it to start fresh
#     if os.path.exists(save_path):
#         os.remove(save_path)
#
#     for i, chunk in enumerate(chunked(param_samples, chunk_size)):
#         results = []
#         for params in tqdm(chunk, desc=f"Sim {i * chunk_size}-{(i+1)*chunk_size}"):
#             res = simulate_cpu(params, copy.deepcopy(storage), Old_Parameters)
#             results.append(res)
#
#         results_all.extend(results)
#
#         # Save progressively (overwrites with accumulated results)
#         np.save(save_path, np.array(results_all))
#
#     return results_all

def sample_inputs_from_spec(
        spec: dict, n_samples: int, random_seed: int | None = None, method: str = "lhs"
) -> torch.Tensor:
    """
    Generate samples from a ProblemSpec-style dictionary.

    Parameters
    ----------
    spec : dict
        Must contain 'names' and 'bounds'.
    n_samples : int
        Number of samples to generate.
    random_seed : int | None
        For reproducibility.
    method : str
        "lhs" or "sobol".

    Returns
    -------
    torch.Tensor
        Samples of shape (n_samples, n_parameters)
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    param_names = spec['names']
    param_bounds = spec['bounds']
    in_dim = len(param_names)

    # Check if any bounds are fixed (min == max)
    constant_params = {i: b[0] for i, b in enumerate(param_bounds) if b[0] == b[1]}
    sample_param_bounds = [b for b in param_bounds if b[0] != b[1]]

    # Handle case all parameters are constant
    if len(sample_param_bounds) == 0:
        const_vals = torch.tensor(list(constant_params.values()))
        return const_vals.repeat(n_samples, 1)

    # Create sampler
    if method.lower() == "lhs":
        sampler = qmc.LatinHypercube(d=len(sample_param_bounds))
    elif method.lower() == "sobol":
        sampler = qmc.Sobol(d=len(sample_param_bounds))
    else:
        raise ValueError(f"Invalid method {method}, choose 'lhs' or 'sobol'.")

    samples = sampler.random(n=n_samples)
    # scale samples to bounds
    scaled_samples = qmc.scale(
        samples,
        [b[0] for b in sample_param_bounds],
        [b[1] for b in sample_param_bounds]
    )
    scaled_samples = torch.tensor(scaled_samples, dtype=torch.float32)

    # Insert constant parameters at the correct indices
    full_samples = torch.empty((n_samples, in_dim), dtype=torch.float32)
    sample_idx = 0
    for idx in range(in_dim):
        if idx in constant_params:
            full_samples[:, idx] = constant_params[idx]
        else:
            full_samples[:, idx] = scaled_samples[:, sample_idx]
            sample_idx += 1

    return full_samples




if __name__ == "__main__":

    lower = 0.8
    upper = 1.2

    sp = ProblemSpec({
        'outputs': ["HR"],

        'names': [
            "beta2", "C2", "K2", "a2", "alpha2", "dc", "KCCO2",
            # "MRBCO2",
            "GV_dead",
            # "Kbg",
            "KcCO2", "KcMRV", "KpCO2", "KpO2", "V0_dead", "VA_rest", "Pmax",
            "Pmax_dot", "E_rs", "R_rs",
            # cardio
            "C_jp", "C_sa", "L_sa", "R_sa", "C_amv", "C_bv",
            "C_ev", "C_hv", "C_rmv", "C_sv", "kr_am", "P_0", "R_amv_n", "R_bv_n",
            "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "K1_vc", "Kr_vc", "Rvc_n",
            "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
            "P0_ra", "KE_la", "KE_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n",
            "P_thormax_n", "P_thormin_n",
            "VT_n", "A_im", "Tc", "T_im", "s",
            # cardio control
            "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
            "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max",
            "Ysh_min", "Ysp_max", "Ysp_min",
            "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
            "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
            "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
            "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
            "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn",
            "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max",
            "f_acCO2_n", "f_ac_max",
            "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v", "T0", "A", "B",
            "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
            "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met", "kmet",
            "MO2_ampn", "phi_max", "phi_min",
            # added params
            "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
            "Kp_po",
            "Kf_po", "Kb_po", "Kv_po", "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2",
            "R_po", "R_mi", "R_tr", "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal",
            "Vu_sa", "V_tot", "Vu_bv", "Vu_hv", "Vu_jp", "Vu_vc",
            "Vvc_max", "Vvc_min", "Vu_pa", "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv",
            "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv",
            "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc", "tau_p", "tau_z", "tau_ac", "tau_ap",
            "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met", "DEmax_lv", "DEmax_rv", "DR_amp",
            "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v", "Dmet", "Fi_CO2",
            "Fi_O2", "Ta", "KE_lv", "KE_rv", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2",
            "tau_MRV",
            "scale_param1", "scale_param2", "scale_param3", "scale_param4",
            "scale_param5", "scale_param6", "scale_param7", "scale_param8",
            "shift_param1", "shift_param2", "shift_param3", "shift_param4",
            "Pa_O2_lower", "rise_time_atr", "fall_time_atr", "rise_time_ven",
            "fall_time_ven", "ahead1", "theta_min", "delta_P"
        ],

        'bounds': [
            # gas
            [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1],
            [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
            [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper],
            [346000 * lower, 346000 * upper],
            # [0.0009 * lower, 0.0009 * upper],
            # resp control
            [0.1698 * lower, 0.1698 * upper],
            # [17.4 * lower, 17.4 * upper],
            [0.2332 * lower, 0.2332 * upper],
            [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
            [0.1587 * lower, 0.1587 * upper], [0.067 * lower, 0.067 * upper], [100 * lower, 100 * upper],
            [1000 * lower, 1000 * upper], [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
            # cardio
            [3.72 * lower, 3.72 * upper],
            [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
            [4.4 * lower, 4.4 * upper],
            [5.71 * lower, 5.71 * upper], [10 * lower, 10 * upper],
            [1.57 * lower, 1.57 * upper],
            [3.28 * lower, 3.28 * upper], [31.11 * lower, 31.11 * upper], [24.17 * lower, 24.17 * upper],
            [3.93 * lower, 3.93 * upper],
            [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
            [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
            [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
            [0.001 * lower, 0.001 * upper], [0.05 * lower, 0.05 * upper],
            [0.76 * lower, 0.76 * upper], [15.8 * lower, 15.8 * upper], [25.37 * lower, 25.37 * upper],
            [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
            [0.1 * lower, 0.1 * upper], [0.35 * lower, 0.35 * upper], [0.55 * lower, 0.55 * upper],
            [0.35 * lower, 0.35 * upper], [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper],
            [0.05 * lower, 0.05 * upper], [1.5 * lower, 1.5 * upper],
            [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
            [-1 * upper, -1 * lower], [-2.5 * upper, -2.5 * lower],
            [-2 * upper, -2 * lower],
            [-6 * upper, -6 * lower],
            [0.73 * lower, 0.73 * upper], [30 * lower, 30 * upper],
            [0.7 * lower, 0.7 * upper], [1.1 * lower, 1.1 * upper], [0.04 * lower, 0.04 * upper],
            # cardio control
            [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper],
            [80 * lower, 80 * upper], [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper],
            [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper], [0.658 * lower, 0.658 * upper],
            [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper],
            [0.22 * lower, 0.22 * upper], [0.114 * lower, 0.114 * upper],
            [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
            [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper],
            [-0.037 * upper, -0.037 * lower], [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower],
            [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower], [-0.68 * upper, -0.68 * lower],
            [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
            [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper],
            [0.2 * lower, 0.2 * upper], [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower],
            [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower], [0.4 * lower, 0.4 * upper],
            [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
            [1.4 * lower, 1.4 * upper], [0.7 * lower, 0.7 * upper], [2.66 * lower, 2.66 * upper],
            [0.475 * lower, 0.475 * upper], [0.282 * lower, 0.282 * upper], [4.47 * lower, 4.47 * upper],
            [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper], [0.695 * lower, 0.695 * upper],
            [-28.29 * upper, -28.29 * lower], [-74.21 * upper, -74.21 * lower], [-28.29 * upper, -28.29 * lower],
            [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper],
            [5.27 * lower, 5.27 * upper], [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper],
            [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper], [0.2 * lower, 0.2 * upper],
            [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper],
            [3.6 * lower, 3.6 * upper], [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper],
            [53 * lower, 53 * upper], [6 * lower, 6 * upper], [6 * lower, 6 * upper],
            [40 * 0.9, 40 * 1.1], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
            [11.76 * lower, 11.76 * upper], [92 * lower, 92 * upper], [112 * lower, 112 * upper],
            [1.4 * lower, 1.4 * upper],
            [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper],
            [3 * lower, 3 * upper], [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper],
            [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper], [0.58 * lower, 0.58 * upper],
            [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
            [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper],
            [0.925 * lower, 0.925 * upper], [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper],
            [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper], [30 * lower, 30 * upper],
            [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
            [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper],
            [0.1555 * lower, 0.1555 * upper], [30 * lower, 30 * upper], [40 * lower, 40 * upper],
            [0.4266 * lower, 0.4266 * upper],
            [0.18 * lower, 0.18 * upper], [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper],
            [-1.87 * upper, -1.87 * lower],
            # added params
            [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper],
            [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper], [100 * lower, 100 * upper],
            [500 * lower, 500 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper],
            [1.309 * lower, 1.309 * upper], [3000 * lower, 3000 * upper], [2000 * lower, 2000 * upper],
            [5 * lower, 5 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
            [100 * lower, 100 * upper], [500 * lower, 500 * upper], [2 * lower, 2 * upper],
            [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper], [0.0000317 * lower, 0.0000317 * upper],
            [350 * lower, 350 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper],
            [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper],
            [2.6 * lower, 2.6 * upper], [3.03e-5 * lower, 3.03e-5 * upper], [104 * lower, 104 * upper],
            [1 * lower, 1 * upper], [5027.6 * 0.8, 5027.6 * 1.2], [279.49 * lower, 279.49 * upper],
            [93.16 * lower, 93.16 * upper],
            [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper], [350 * lower, 350 * upper],
            [50 * lower, 50 * upper], [1 * lower, 1 * upper], [116.6775 * lower, 116.6775 * upper],
            [114 * lower, 114 * upper], [4 * lower, 4 * upper], [15.908 * lower, 15.908 * upper],
            [4 * lower, 4 * upper], [38.703 * lower, 38.703 * upper], [8 * lower, 8 * upper],
            [8 * lower, 8 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
            [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper],
            [286.4 * lower, 286.4 * upper], [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper],
            [1361.6 * lower, 1361.6 * upper], [20 * lower, 20 * upper], [30 * lower, 30 * upper],
            [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper],
            [20 * lower, 20 * upper], [10 * lower, 10 * upper], [5 * lower, 5 * upper],
            [40 * lower, 40 * upper], [10 * lower, 10 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper],
            [5 * lower, 5 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
            [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper], [4 * lower, 4 * upper],
            [0.0421 * lower, 0.0421 * upper], [21.0379 * lower, 21.0379 * upper], [5 * lower, 5 * upper],
            [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
            [1 * lower, 1 * upper], [2 * lower, 2 * upper], [3 * lower, 3 * upper],
            [2.5 * lower, 2.5 * upper], [20 * lower, 20 * upper], [0.9 * lower, 0.9 * upper],
            [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper], [0.25 * lower, 0.25 * upper],
            [50 * lower, 50 * upper],
            # further added params
            [4.9 * lower, 4.9 * upper], [1.5 * lower, 1.5 * upper], [0.3 * lower, 0.3 * upper],
            [26.6 * lower, 26.6 * upper], [0.5 * lower, 0.5 * upper], [1.2 * lower, 1.2 * upper],
            [30 * lower, 30 * upper], [1.6 * lower, 1.6 * upper], [4 * lower, 4 * upper],
            [0.3 * lower, 0.3 * upper], [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper],
            [80 * lower, 80 * upper], [0.05 * lower, 0.05 * upper], [0.1 * lower, 0.1 * upper],
            [0.15 * lower, 0.15 * upper], [0.3 * lower, 0.3 * upper], [0.9 * 0.95, 0.9 * 1.05],
            [0.0872665 * lower, 0.0872665 * upper], [0.3 * lower, 0.3 * upper]]
    })


    # # filtered set from DGSM
    # subset_vars = {'k_ac', 'Wp_sv', 'ahead1', 'theta_min', 'delta_P', 'G_ap', 'Cvh_O2_n', 'T_im', 'K1_vc',
    #                'theta_mi_max', 'P0_rv', 'GT_s', 'theta_svn', 'Emax_lv0', 'f_ac_min',
    #                'kmet', 'R_sa', 'R_bpn', 'Io_sv', 'phi_max', 'R_po', 'f_acCO2_n', 'Kv_tr', 'Emax_rv0', 'V_tot',
    #                'kes', 'Io_met', 'Cvam_O2_n', 'fev_inf', 'theta_spn', 'theta_tr_max', 'Wb_sh',
    #                'C_pp', 'Vu_hv', 'g_ccsp', 'R_mi', 'f_ab_max', 'Wb_sv', 'Tc', 'GEmax_rv', 'GEmax_lv',
    #                'Vu_bv', 'KE_lv', 'Wc_sp', 'scale_param2', 'KE_ra', 'GR_ep', 'Vu_rv', 'fes_min', 'Ysv_min',
    #                'k_ab', 'R_pv', 'grm_O2', 'KE_la', 'fes_o', 'Vu_vc', 'GR_sp',
    #                'Cvrm_O2_n', 'C_jp', 'Wc_v', 'C_pv', 'g_ccsh', 'C_sv', 'MO2_rmp', 'rise_time_ven',
    #                'Ysv_max', 'Vu_amv0', 'KE_rv', 'Vu_lv', 'Vu_ev0', 'GT_v', 'R_amp0', 'D', 'gb_O2',
    #                'f_ac_max', 'theta_v', 'theta_shn', 'kcc_sv', 'kev', 'fes_inf', 'MO2_ampn', 'Vu_rmv0', 'Wb_sp',
    #                'Vu_jp', 'Cvb_O2_n', 'Kv_po', 'Wp_v', 'Rvc_n', 'R_rmp0', 'R_sp0',
    #                'kcc_sh', 'Kp_tr', 'GV_sv', 'T0', 'fev_o', 'R_tr', 'theta_po_max', 'f_ab_min',
    #                'R_hv_n', 'R_pa', 'P_thormin_n', 'Vu_sv0', 'fab_o', 'phi_min', 'fall_time_ven', 'Wp_sh', 'Kp_po',
    #                'P0_lv', 'R_ep0', 'Vu_pv', 'C_ev', 'MO2_bp', 'Wc_sh', 'P_n', 'Vu_pp', 'R_pp'
    #                # removed the below to just focus on the cardiovascular variables
    #                'beta2', 'C2', 'K2', 'a2', 'alpha2', 'GV_dead', 'KcCO2', 'KpCO2', 'Fi_O2', 'V0_dead'
    #                'VA_rest', 'E_rs', 'R_rs', 'PaCO2_n', 'C_O2_param1', 'C_O2_param2', 'PaO2_ac_n',
    #                'scale_param3', 'scale_param4', 'K_H'
    #                }

 # change
 #    # HR: 17 parameters contribute 90 % sensitivity
 #    subset_vars = {'T0', 'V_tot', 'P_n', 'fev_o', 'GT_v', 'GT_s', 'C2', 'C_O2_param1', 'Fi_O2',
 # 'Vu_sv0', 'fes_o', 'fab_o', 'kes', 'Wb_sh', 'K2', 'k_ab', 'f_acCO2_n'}
 #    HR: Heart Rate: 15 parameters contribute 90% sensitivity
 #    subset_vars = {'T0', 'GT_s', 'GT_v', 'fev_o', 'Fi_O2', 'AT', 'V_tot', 'Yv_max', 'Io_sh', 'R_rs',
 #     'E_rs', 'Wp_v', 'G_ap', 'P_n_max', 'Ysh_max'}

 #    Max RA Pressure: 89 parameters contribute 90 % sensitivity
 #    subset_vars = {'a2', 'Vu_sv0', 'MO2_bp', 'PaCO2_n', 'C2', 'G_ap', 'Wp_v', 'R_rs', 'kes',
 #     'V0_dead', 'GT_v', 'GV_dead', 'theta_v', 'K2', 'PaO2_ac_n', 'beta2', 'V_tot',
 #     'E_rs', 'Fi_O2', 'kev', 'fev_o', 'GV_sv', 'Wb_sh', 'T0', 'f_ab_max', 'fes_o',
 #     'Wc_v', 'f_acCO2_n', 'P_n', 'alpha2', 'C_O2_param1', 'k_ab', 'GT_s',
 #     'scale_param4', 'Cvb_O2_n', 'KcCO2', 'C_pv', 'fev_inf', 'Vu_ev0', 'Kv_mi',
 #     'fes_min', 'Vu_jp', 'fab_o', 'f_ac_max', 'theta_mi_max', 'f_ab_min', 'KE_ra',
 #     'C_pp', 'k_ac', 'theta_shn', 'Vu_bv', 'VA_rest', 'Cvrm_O2_n', 'f_ac_min',
 #     'R_bpn', 'Io_met', 'Cvam_O2_n', 'grm_O2', 'KE_lv', 'fall_time_ven', 'Emax_ra',
 #     'Kv_tr', 'C_sv', 'KE_rv', 'Wb_sp', 'kmet', 'P0_rv', 'theta_svn', 'g_ccsh',
 #     'C_O2_param2', 'Wc_sh', 'Wp_sp', 'KpCO2', 'C_jp', 'Vu_rmv0', 'R_pv', 'Kp_mi',
 #     'fes_inf', 'Io_sh', 'theta_tr_max', 'Vu_vc', 'Io_sp', 'R_sp0', 'kcc_sh', 'C_sa',
 #     'C_bv', 'MO2_hpn', 'Vu_rv', 'Vu_pp'}

# #   EDP: 83 parameters contribute 90% sensitivity
#     subset_vars = {'Wp_v', 'fab_o', 'G_ap', 'theta_v', 'Fi_O2', 'a2', 'Vu_ev0', 'C2', 'C_O2_param1',
#                    'C_pp', 'Vu_jp', 'R_bpn', 'T0', 'fes_o', 'C_pv', 'PaCO2_n', 'R_sp0', 'V_tot',
#                    'GV_sv', 'kes', 'K2', 'P_n', 'k_ab', 'Wb_sh', 'Kp_tr', 'Cvrm_O2_n', 'fev_inf',
#                    'theta_mi_max', 'Kv_mi', 'kev', 'fev_o', 'KE_lv', 'Emax_lv0', 'MO2_bp', 'R_pv',
#                    'f_acCO2_n', 'GT_s', 'Kp_mi', 'theta_spn', 'theta_shn', 'Wc_v', 'kcc_sv',
#                    'Vu_bv', 'Vu_sv0', 'E_rs', 'Kv_tr', 'Cvb_O2_n', 'C_sv', 'PaO2_ac_n', 'C_jp',
#                    'Wb_sp', 'f_ac_max', 'Io_met', 'GT_v', 'f_ab_min', 'Io_sv', 'V0_dead', 'Vu_vc',
#                    'GR_ep', 'fall_time_ven', 'f_ab_max', 'KcCO2', 'Cvam_O2_n', 'k_ac',
#                    'theta_tr_max', 'Wb_sv', 'phi_min', 'kmet', 'Vu_rmv0', 'VA_rest', 'KE_rv',
#                    'C_O2_param2', 'P0_lv', 'Vu_amv0', 'R_ep0', 'Rvc_n', 'fes_inf', 'g_ccsh',
#                    'theta_svn', 'fes_min', 'GV_dead', 'R_mi', 'MO2_rmp'}

# #   Systolic Pressure: 28 parameters contribute 90% sensitivity
#     subset_vars = {'V_tot', 'Vu_sv0', 'P_n', 'C2', 'PaCO2_n', 'kes', 'a2', 'V0_dead', 'fes_o', 'R_rs',
#                    'E_rs', 'GV_dead', 'Vu_ev0', 'K2', 'Vu_jp', 'C_pv', 'fes_min', 'R_pv', 'R_sa',
#                    'Fi_O2', 'Cvrm_O2_n', 'C_O2_param1', 'fab_o', 'rise_time_ven', 'fall_time_ven',
#                    'GV_sv', 'C_pp', 'Kv_tr'}
#     subset_vars = {
#         'V_tot', 'Vu_sv0', 'GV_sv', 'R_rs', 'G_ap', 'R_sa', 'fes_o', 'P_n', 'Fi_O2',
#         'E_rs', 'fab_o', 'C_pv', 'rise_time_ven', 'GT_v', 'Vu_ev0', 'f_acCO2_n', 'C_sv',
#         'Vu_jp', 'fall_time_ven', 'T0', 'Wc_v', 'C_O2_param1', 'Kv_mi', 'k_ab',
#         'V0_dead', 'C_pp', 'Kp_mi', 'GV_dead', 'Wb_sh', 'fev_inf', 'Kv_tr', 'fev_o',
#         'Wp_v', 'Ysh_max', 'PaO2_ac_n', 'kev', 'theta_v', 'AT', 'tauMR', 'VA_rest',
#         'P_n_max', 'GT_s', 'R_pv', 'f_ab_max', 'k_ac', 'GR_amp', 'f_ac_max', 'Yv_max',
#         'Io_met', 'theta_mi_max', 'KE_lv', 'Kp_tr', 'Io_sh', 'MO2_bp', 'KcCO2', 'Tc',
#         'Vu_amv0', 'theta_tr_max', 'phi_max', 'Vu_bv', 'kes', 'PaCO2_n', 'f_ac_min'
#     }

# # Max RV Pressure: 46 parameters contribute 90% sensitivity
#     subset_vars = {'V_tot', 'PaCO2_n', 'C2', 'R_rs', 'a2', 'V0_dead', 'E_rs', 'K2', 'Vu_sv0',
#                    'GV_dead', 'C_O2_param1', 'alpha2', 'Vu_ev0', 'Vu_jp', 'P_n', 'rise_time_ven',
#                    'KcCO2', 'Fi_O2', 'Wb_sh', 'C_pv', 'Kv_tr', 'kes', 'fes_o', 'MO2_bp', 'fab_o',
#                    'theta_v', 'GT_s', 'VA_rest', 'G_ap', 'Wp_v', 'beta2', 'fev_inf', 'k_ab', 'C_pp',
#                    'fev_o', 'kev', 'T0', 'f_acCO2_n', 'GV_sv', 'Kp_tr', 'R_bpn', 'KE_rv', 'k_ac',
#                    'KE_lv', 'theta_tr_max', 'Wc_v'}

    subset_vars = {
        'V_tot', 'Vu_sv0', 'E_rs', 'R_rs', 'GV_sv', 'GV_dead', 'V0_dead', 'theta_v',
        'C_O2_param1', 'Wp_v', 'G_ap', 'C_pv', 'VA_rest', 'Wc_v', 'Wb_sh', 'AT', 'Vu_jp',
        'rise_time_ven', 'k_ab', 'Vu_ev0', 'k_ac', 'f_acCO2_n', 'Io_sh', 'fab_o',
        'Kv_tr', 'Yv_max', 'tauMR', 'kev', 'PaO2_ac_n', 'P_n_max', 'fev_o', 'Fi_O2',
        'C_pp', 'fes_o', 'GT_v', 'P_n', 'C_sv', 'KcCO2', 'fev_inf', 'GT_s', 'PaCO2_n',
        'C2', 'MO2_bp', 'T0', 'Ysh_max', 'f_ac_max', 'Tc', 'Kp_tr', 'f_ab_max',
        'fall_time_ven', 'theta_tr_max', 'R_po', 'a2', 'Kv_mi', 'KE_lv', 'kes', 'Io_sv',
        'Kp_mi', 'GR_amp', 'Io_met', 'R_pv', 'KE_rv', 'f_ac_min', 'K2', 'Cvb_O2_n',
        'phi_max', 'f_ab_min', 'Vu_bv', 'R_bpn', 'theta_mi_max', 'scale_param4',
        'kcc_sh', 'Rvc_n'
    }

# PaO2: 2 parameters contribute 90% sensitivity
#     subset_vars = {'Fi_O2', 'PaCO2_n'}

# # EDV: 12 parameters contribute 90% sensitivity
#     subset_vars = {'V_tot', 'Vu_sv0', 'Emax_lv0', 'T0', 'Vu_ev0', 'Vu_jp', 'fall_time_ven', 'C_pv',
#                     'C_O2_param1', 'Kv_tr', 'C_pp', 'KE_lv'}

# Minute Ventilation: 7 parameters contribute 90% sensitivity
#     subset_vars = {'R_rs', 'PaCO2_n', 'E_rs', 'C2', 'V0_dead', 'GV_dead', 'V_tot'}

    # subset_vars = {
    #     'R_rs', 'E_rs', 'GV_dead', 'V0_dead', 'PaCO2_n', 'VA_rest', 'KcCO2',
    #     'V_tot', 'C_O2_param1', 'C2', 'MO2_bp', 'KcMRV'
    # }
    #
    # Filter the names and bounds
    filtered_names = []
    filtered_bounds = []

    A = len(subset_vars)

    for name, bound in zip(sp["names"], sp["bounds"]):
        filtered_names.append(name)
        if name in subset_vars:
            filtered_bounds.append(bound)
            # filtered_bounds.append([np.mean(bound), np.mean(bound)])
        else:
            # take nominal (mean of lower/upper)
            filtered_bounds.append([np.mean(bound), np.mean(bound)])

    # Create a new ProblemSpec with only the filtered variables
    sp_filtered = ProblemSpec({
        "outputs": sp["outputs"],
        "names": filtered_names,
        "bounds": filtered_bounds
    })

    param_keys = list(sp_filtered["names"])

    # AA = np.load("Result_DGSM_chunked.npy")
    # change
    # X = sample_inputs_from_spec(sp_filtered, n_samples=10000, random_seed=42, method="lhs")
    # X = X.cpu().numpy() if X.is_cuda else X.numpy()
    # np.save('Max_RV_LHCS_10000_X_sample_exercise_20.npy', X)
    X = np.load('Max_RV_LHCS_10000_X_sample_exercise_20.npy')

    mask = np.ptp(X, axis=0) != 0
    print(np.sum(mask))

    param_samples = [dict(zip(param_keys, row)) for row in X]

    # A = [{'A': 20.9, 'AT': 1/60, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 87.0, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.6, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 10.0, 'C_hv': 1.57, 'C_jp': 3.72, 'C_pa': 0.76, 'C_pp': 15.8, 'C_pv': 25.37, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 31.11, 'Cvam_O2_n': 0.1555, 'Cvb_O2_n': 0.14, 'Cvh_O2_n': 0.11, 'Cvrm_O2_n': 0.155, 'D': -5.251, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 21.9, 'Emax_la': 0.35, 'Emax_lv0': 1.4, 'Emax_ra': 0.35, 'Emax_rv0': 0.7, 'Fi_CO2': 0.0421, 'Fi_O2': 21.0379, 'GEmax_lv': 0.475, 'GEmax_rv': 0.282, 'GR_amp': 4.47, 'GR_ep': 1.94, 'GR_rmp': 2.47, 'GR_sp': 0.695, 'GT_s': -0.13, 'GT_v': 0.09, 'GV_amv': -28.29, 'GV_dead': 0.1698, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -265.4, 'G_ap': 11.76, 'Io_met': 0.4266, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.45, 'Io_v': 0.22, 'K1_vc': 0.15, 'K2': 194.4, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.05, 'KE_lv': 0.014, 'KE_ra': 0.05, 'KE_rv': 0.011, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.2332, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.2025, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 3000.0, 'Kp_tr': 100.0, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 10.0, 'Kv_tr': 7.0, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.516, 'MO2_bp': 0.925, 'MO2_hpn': 0.4, 'MO2_rmp': 0.86, 'P0_la': 0.55, 'P0_lv': 1.5, 'P0_ra': 0.55, 'P0_rv': 1.5, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 92.0, 'P_n_max': 112.0, 'P_thormax_n': -4.0, 'P_thormin_n': -4.0, 'PaCO2_n': 40.0, 'PaO2_ac_n': 45.0, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 3.51, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 6.57, 'R_bv_n': 0.075, 'R_ep0': 1.655, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.224, 'R_mi': 350.0, 'R_pa': 0.023, 'R_po': 350.0, 'R_pp': 0.0894, 'R_pv': 0.1, 'R_rmp0': 5.27, 'R_rmv_n': 0.125, 'R_rs': 3.02, 'R_sa': 0.06, 'R_sp0': 2.49, 'R_sv_n': 0.038, 'R_tr': 350.0, 'Rvc_n': 0.05, 'T0': 0.58, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.1, 'Ta': 5.0, 'Tc': 0.7, 'V0_dead': 0.1587, 'VA_rest': 0.067, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 5027.6, 'Vu_amv0': 286.4, 'Vu_bv': 279.49, 'Vu_ev0': 607.8, 'Vu_hv': 93.16, 'Vu_jp': 579.76, 'Vu_la': 4.0, 'Vu_lv': 15.908, 'Vu_pa': 1.0, 'Vu_pp': 116.6775, 'Vu_pv': 114.0, 'Vu_ra': 4.0, 'Vu_rmv0': 190.95, 'Vu_rv': 38.703, 'Vu_sa': 1.0, 'Vu_sv0': 1361.6, 'Vu_vc': 123.0, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.75, 'Wb_sp': -1.1375, 'Wb_sv': -1.1375, 'Wc_sh': 1.0, 'Wc_sp': 1.716, 'Wc_sv': 1.716, 'Wc_v': 0.2, 'Wp_sh': -0.2, 'Wp_sp': -0.3997, 'Wp_sv': -0.3997, 'Wp_v': -0.103, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 64.9, 'Ysv_min': -0.437, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.819, 'ahead1': 0.9, 'alpha2': 0.05591, 'alpha_O2': 3.17e-05, 'beta2': 0.03255, 'dc': 0.015, 'delta_P': 0.3, 'f_ab_max': 47.78, 'f_ab_min': 2.52, 'f_acCO2_n': 1.4, 'f_ac_max': 12.3, 'f_ac_min': 0.835, 'fab_o': 25.0, 'fall_time_atr': 0.1, 'fall_time_ven': 0.3, 'fes_inf': 2.1, 'fes_max': 80.0, 'fes_min': 2.66, 'fes_o': 16.11, 'fev_inf': 6.3, 'fev_o': 3.2, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 1.0, 'g_ccsp': 1.5, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 10.0, 'gh_O2': 35.0, 'grm_O2': 30.0, 'k_ab': 11.76, 'k_ac': 29.27, 'kcc_sh': 0.114, 'kcc_sp': 0.13, 'kcc_sv': 0.09, 'kcc_v': 0.0162, 'kes': 0.0675, 'kev': 7.06, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.18, 'kr_am': 24.17, 'phi_max': 20.0, 'phi_min': -1.87, 'rise_time_atr': 0.05, 'rise_time_ven': 0.15, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.5, 'scale_param3': 0.3, 'scale_param4': 26.6, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.309, 'theta_min': 0.0872665, 'theta_po_max': 1.309, 'theta_shn': 3.6, 'theta_spn': 13.32, 'theta_svn': 13.32, 'theta_tr_max': 1.309, 'theta_v': -0.68, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0},]

    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=100)
    # print(Result)
    # change
    np.save('Max_RV_Result_exercise_10000_20.npy', Result)

    # 515 is from 0_10000
    # 390 is from 10000_20000
    # 515 is from 20000_30000
    # 392 is from 30000_40000









