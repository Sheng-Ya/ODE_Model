import os
import copy
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
from scipy.signal import find_peaks
from All_derivatives_njit import model_derivatives
from fixed_params import Parameters as Old_Parameters

from Initial_Conditions_after_running_again import Initial_Conditions
from All_Next_Conditions import Next_Conditions


target_values = np.arange(0, 10000, 10)
BUFFER_LIMIT = 20000

max_time = 200 # Maximum time limit to avoid infinite loops

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


def simulate_cpu(Current_Parameters, storage, old_parameters):
    local_updates = {key: copy.deepcopy(value) for key, value in storage.items()}

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
     fall_time_ven, ahead1
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
    "fall_time_atr", "rise_time_ven", "fall_time_ven", "ahead1"])

    # determine the correct breathing profile
    c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6 = minimise_breathing(1.5,
    1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao)

    if all(x == 0 for x in [c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6]):
        # Integration failed or early termination
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, None, None

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
     fall_time_ven, ahead1]

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
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None, None, None

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT

    P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
    peaks, _ = find_peaks(P_sa, distance=int(500))
    troughs, _ = find_peaks(-P_sa, distance=int(500))

    last_10_troughs_P_sa = troughs[-10:-1]
    last_10_min_P_sa = P_sa[last_10_troughs_P_sa]

    last_10_peaks_P_sa = peaks[-10:-1]
    last_10_max_P_sa = P_sa[last_10_peaks_P_sa]

    V_lv = np.concatenate((local_updates["V_lv_store"][i_buffer:], local_updates["V_lv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)

    last_10_troughs_V_lv = troughs[-8:-1]
    last_10_min_V_lv = V_lv[last_10_troughs_V_lv]

    last_10_peaks_V_lv = peaks[-8:-1]
    last_10_max_V_lv = V_lv[last_10_peaks_V_lv]

    V_rv = np.concatenate((local_updates["V_rv_store"][i_buffer:], local_updates["V_rv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_rv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_rv, distance=int(500), prominence=1)

    last_10_troughs_V_rv = troughs[-10:-1]
    last_10_min_V_rv = V_rv[last_10_troughs_V_rv]

    last_10_peaks_V_rv = peaks[-10:-1]
    last_10_max_V_rv = V_rv[last_10_peaks_V_rv]

    P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
    peaks, _ = find_peaks(P_rv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-P_rv, distance=int(500), prominence=1)

    last_10_troughs_P_rv = troughs[-10:-1]
    last_10_min_P_rv = P_rv[last_10_troughs_P_rv]

    last_10_peaks_P_rv = peaks[-10:-1]
    last_10_max_P_rv = P_rv[last_10_peaks_P_rv]


    # Get past 10 HR
    HR = np.concatenate((local_updates["HR_store"][i_buffer:], local_updates["HR_store"][:i_buffer]))

    past_10_flat_segments = []
    # Start from the end and track the current segment value
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            # New segment found
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break
    print(np.mean(past_10_flat_segments), np.mean(last_10_max_P_sa), np.mean(last_10_max_V_lv), np.mean(last_10_max_V_rv), np.mean(last_10_max_P_rv))

    # A = IC_overall.copy()

    IC_current = ODE_solution.y[:, -1]

    return ([np.mean(past_10_flat_segments), np.mean(last_10_max_P_sa), np.mean(last_10_min_P_sa),
            np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(last_10_max_V_rv), np.mean(last_10_min_V_rv),
            np.mean(last_10_max_P_rv), np.mean(last_10_min_P_rv)], IC_current, local_updates,
            [c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6])


#
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def parallel_simulations(param_samples, storage, n_jobs, chunk_size=1000, save_path='Result_DGSM_chunked.npy'):
    chunk_array = None

    # If file exists from previous run, remove it to start fresh
    if os.path.exists(save_path):
        os.remove(save_path)

    for i, chunk in enumerate(chunked(param_samples, chunk_size)):
        with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i+1)*chunk_size}", total=len(chunk))):
            results = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, copy.deepcopy(storage), Old_Parameters) for params in chunk)

        results_chunk = [res[0] for res in results]
        # Initialize chunk_array if first iteration
        if chunk_array is None:
            chunk_array = np.zeros_like(results_chunk)

        # Overwrite chunk_array with current results
        np.copyto(chunk_array, results_chunk)

        # Optional: also accumulate in a single array
        np.save(save_path, chunk_array)  # full file overwritten

    return chunk_array


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
#         results_chunk = [res[0] for res in results]
#         results_all.extend(results_chunk)
#
#         # Save progressively (overwrites with accumulated results)
#         np.save(save_path, np.array(results_all))
#
#     return results_all

# def sample_inputs_from_spec(
#         spec: dict, n_samples: int, random_seed: int | None = None, method: str = "lhs"
# ) -> torch.Tensor:
#     """
#     Generate samples from a ProblemSpec-style dictionary.
#
#     Parameters
#     ----------
#     spec : dict
#         Must contain 'names' and 'bounds'.
#     n_samples : int
#         Number of samples to generate.
#     random_seed : int | None
#         For reproducibility.
#     method : str
#         "lhs" or "sobol".
#
#     Returns
#     -------
#     torch.Tensor
#         Samples of shape (n_samples, n_parameters)
#     """
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#
#     param_names = spec['names']
#     param_bounds = spec['bounds']
#     in_dim = len(param_names)
#
#     # Check if any bounds are fixed (min == max)
#     constant_params = {i: b[0] for i, b in enumerate(param_bounds) if b[0] == b[1]}
#     sample_param_bounds = [b for b in param_bounds if b[0] != b[1]]
#
#     # Handle case all parameters are constant
#     if len(sample_param_bounds) == 0:
#         const_vals = torch.tensor(list(constant_params.values()))
#         return const_vals.repeat(n_samples, 1)
#
#     # Create sampler
#     if method.lower() == "lhs":
#         sampler = qmc.LatinHypercube(d=len(sample_param_bounds))
#     elif method.lower() == "sobol":
#         sampler = qmc.Sobol(d=len(sample_param_bounds))
#     else:
#         raise ValueError(f"Invalid method {method}, choose 'lhs' or 'sobol'.")
#
#     samples = sampler.random(n=n_samples)
#     # scale samples to bounds
#     scaled_samples = qmc.scale(
#         samples,
#         [b[0] for b in sample_param_bounds],
#         [b[1] for b in sample_param_bounds]
#     )
#     scaled_samples = torch.tensor(scaled_samples, dtype=torch.float32)
#
#     # Insert constant parameters at the correct indices
#     full_samples = torch.empty((n_samples, in_dim), dtype=torch.float32)
#     sample_idx = 0
#     for idx in range(in_dim):
#         if idx in constant_params:
#             full_samples[:, idx] = constant_params[idx]
#         else:
#             full_samples[:, idx] = scaled_samples[:, sample_idx]
#             sample_idx += 1
#
#     return full_samples




if __name__ == "__main__":

    lower = 0.5
    upper = 1.5

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
            "fall_time_ven", "ahead1"
        ],

        'bounds': [
            # gas
            [0.03255 * lower, 0.03255 * upper], [87 * 0.9, 87 * 1.1],
            [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
            [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],
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
            [0.05 * lower, 0.05 * upper],
            [0.35 * lower, 0.35 * upper], [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper],
            [1.5 * lower, 1.5 * upper],
            [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
            [-1 * upper, -1 * lower], [-2.5 * upper, -2.5 * lower],
            [-4 * upper, -4 * lower],
            [-9 * upper, -9 * lower],
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
            [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [0.1 * lower, 0.1 * upper],
            [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper], [600 * lower, 600 * upper],
            [800 * lower, 800 * upper], [1 * lower, 1 * upper], [10 * lower, 10 * upper],
            [1.309 * lower, 1.309 * upper], [800 * lower, 800 * upper], [800 * lower, 800 * upper],
            [1 * lower, 1 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
            [600 * lower, 600 * upper], [800 * lower, 800 * upper], [1 * lower, 1 * upper],
            [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper], [0.0000317 * lower, 0.0000317 * upper],
            [350 * lower, 350 * upper], [40 * lower, 40 * upper], [40 * lower, 40 * upper],
            [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper],
            [2.6 * lower, 2.6 * upper], [3.03e-5 * lower, 3.03e-5 * upper], [104 * lower, 104 * upper],
            [1 * lower, 1 * upper], [5027.6 * lower, 5027.6 * upper], [279.49 * lower, 279.49 * upper],
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
            [0.15 * lower, 0.15 * upper], [0.3 * lower, 0.3 * upper], [0.9 * lower, 0.9 * upper]]
    })


    # filtered set from DGSM
    subset_vars = {'k_ac', 'C_O2_param2', 'Wp_sv', 'ahead1', 'G_ap', 'PaO2_ac_n', 'Cvh_O2_n', 'T_im', 'K1_vc',
                   'theta_mi_max', 'P0_rv', 'KcCO2', 'R_rs', 'GT_s', 'theta_svn', 'Emax_lv0', 'C_O2_param1', 'f_ac_min',
                   'kmet', 'R_sa', 'R_bpn', 'Io_sv', 'phi_max', 'R_po', 'f_acCO2_n', 'Kv_tr', 'Emax_rv0', 'V_tot',
                   'kes', 'Io_met', 'Cvam_O2_n', 'fev_inf', 'theta_spn', 'theta_tr_max', 'E_rs', 'C2', 'Wb_sh',
                   'C_pp', 'Vu_hv', 'g_ccsp', 'R_mi', 'f_ab_max', 'Wb_sv', 'K2', 'Tc', 'GEmax_rv', 'GEmax_lv', 'a2',
                   'Vu_bv', 'KE_lv', 'Wc_sp', 'scale_param2', 'KE_ra', 'GR_ep', 'Vu_rv', 'fes_min', 'Ysv_min',
                   'VA_rest', 'k_ab', 'R_pv', 'scale_param3', 'KpCO2', 'grm_O2', 'KE_la', 'fes_o', 'Vu_vc', 'GR_sp',
                   'Cvrm_O2_n', 'C_jp', 'Wc_v', 'C_pv', 'g_ccsh', 'Fi_O2', 'C_sv', 'MO2_rmp', 'rise_time_ven',
                   'Ysv_max', 'Vu_amv0', 'KE_rv', 'Vu_lv', 'beta2', 'Vu_ev0', 'GT_v', 'R_amp0', 'D', 'gb_O2',
                   'f_ac_max', 'theta_v', 'theta_shn', 'kcc_sv', 'kev', 'fes_inf', 'MO2_ampn', 'Vu_rmv0', 'Wb_sp',
                   'Vu_jp', 'Cvb_O2_n', 'Kv_po', 'Wp_v', 'PaCO2_n', 'Rvc_n', 'scale_param4', 'R_rmp0', 'R_sp0',
                   'alpha2', 'kcc_sh', 'Kp_tr', 'GV_sv', 'T0', 'V0_dead', 'fev_o', 'R_tr', 'theta_po_max', 'f_ab_min',
                   'R_hv_n', 'R_pa', 'P_thormin_n', 'Vu_sv0', 'fab_o', 'phi_min', 'fall_time_ven', 'Wp_sh', 'Kp_po',
                   'K_H', 'P0_lv', 'R_ep0', 'Vu_pv', 'C_ev', 'MO2_bp', 'GV_dead', 'Wc_sh', 'P_n', 'Vu_pp', 'R_pp'}
    # Filter the names and bounds
    filtered_names = []
    filtered_bounds = []

    for name, bound in zip(sp["names"], sp["bounds"]):
        filtered_names.append(name)
        if name in subset_vars:
            filtered_bounds.append(bound)
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

    # X = sample_inputs_from_spec(sp_filtered, n_samples=500000, random_seed=42, method="lhs")
    # X = X.cpu().numpy() if X.is_cuda else X.numpy()
    # np.save('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy', X)
    X = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[:500000,:]

    param_samples = [dict(zip(param_keys, row)) for row in X]

    A = param_samples[0]

    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=-1)
    # print(Result)

    np.save('DGSM_filtered_LHCS_300000_500000_Result_HR_Plv_Prv_Vlv_Vrv_rest.npy', Result)










