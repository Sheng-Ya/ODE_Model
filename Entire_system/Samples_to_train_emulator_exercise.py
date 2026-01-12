import os
import copy
import signal
import torch
from scipy.stats import qmc
from scipy.interpolate import CubicSpline, interp1d
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
BUFFER_LIMIT = 40000

max_time = 60 # Maximum time limit to avoid infinite loops

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1:  # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            # # num_removed = 6
            # index = -1
            #
            # # Iterating through the buffer in circular order
            # for j in range(BUFFER_LIMIT):
            #     logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
            #     if all_time[logical_index] < t:
            #         index = (logical_index + 1) % BUFFER_LIMIT
            #         break
            #
            # num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (
            #             actual_index - index)

            num_removed = 3
            index = (actual_index - 3) % BUFFER_LIMIT
            for j in range(num_removed):
                all_time[(index + j) % BUFFER_LIMIT] = 0

        else:
            num_removed = 0
    else:
        num_removed = 0

    # Indices for slicing
    idx_resp_contr = num_cardio + num_cardio_control + num_gas + num_resp_control

    # Extract each subsystem's state variables
    resp_contr_state = Initial_Conditions_numpy[:idx_resp_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    derivatives_all = model_derivatives(t, resp_contr_state, Initial_Conditions_dict, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2)
    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

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

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))

def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
    dt = 0.001 # must edit in Resp_Control_Breath_Optimiser too
    bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
    tolerance = 0.0001

    VAflow_vals = np.linspace(0.01, 1, 200)
    VAflow_repeated = np.repeat(VAflow_vals, 3)

    VD = GV_dead * VAflow_repeated + V0_dead

    optimal_t1 = []
    optimal_t2 = []
    initial_guess = [t1, t2]
    required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]

    for idx, VAflow in enumerate(VAflow_repeated):
        VD_volume = VD[idx]

        res = minimize(objective, x0= np.array(initial_guess[-2:]),
                       args=(required_params, VAflow, VD_volume, dt, tolerance), method='nelder-mead', bounds=bounds)
        t1_opt, t2_opt = res.x
        optimal_t1.append(t1_opt)
        optimal_t2.append(t2_opt)
        initial_guess.extend(res.x)


    # Convert to arrays for indexing
    VAflow_clean = np.array(VAflow_repeated)
    t1_clean = np.array(optimal_t1)
    t2_clean = np.array(optimal_t2)

    t1_mean = np.array([np.nanmean(t1_clean[VAflow_clean == v]) for v in VAflow_vals])
    t2_mean = np.array([np.nanmean(t2_clean[VAflow_clean == v]) for v in VAflow_vals])

    cs_t1 = CubicSpline(VAflow_vals, t1_mean, bc_type="natural")
    cs_t2 = CubicSpline(VAflow_vals, t2_mean, bc_type="natural")

    coeffs_t1 = cs_t1.c  # shape (4, N-1)
    coeffs_t2 = cs_t2.c  # shape (4, N-1)

    knots_1 = cs_t1.x
    knots_2 = cs_t2.x

    return coeffs_t1, coeffs_t2, knots_1, knots_2


def simulate_cpu(Current_Parameters, local_updates, old_parameters):
    # local_updates = {key: copy.deepcopy(value) for key, value in storage.items()}

    IC_current = IC_overall.copy()
    t_span = [0, max_time]

    # Cardio parameters
    (g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, D1, K1_vc, Kr_vc, Rvc_n,
    C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    kr_am) = (
    Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
    ["g_thor", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa",
     "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv",
     "Emax_la", "P0_la", "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa",
     "R_sa", "D1", "K1_vc", "Kr_vc", "Rvc_n", "C_jp",
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
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vvc_max, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
     scale_param1, scale_param2, scale_param3, scale_param4, scale_param5, scale_param6, scale_param7,
     Pa_O2_lower, rise_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale
     ) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["Kp_ao", "Kf_ao", "Kb_ao",
    "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po", "Kf_po", "Kb_po", "Kv_po",
    "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2", "R_po", "R_mi", "R_tr", "R_ao",
    "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal", "Vu_sa", "V_tot", "Vu_jp",
    "Vu_bv", "Vu_hv", "Vu_vc", "Vvc_max", "Vu_pa", "Vu_pp", "Vu_pv",
    "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp",
    "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc",
    "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met",
    "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v",
    "Dmet", "Fi_CO2", "Fi_O2", "Ta", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
    "scale_param1", "scale_param2", "scale_param3", "scale_param4", "scale_param5", "scale_param6", "scale_param7",
    "Pa_O2_lower", "rise_time_atr",
    "rise_time_ven", "fall_time_ven", "ahead1", "theta_min", "delta_P", "r", "l", "V_nominal", "V_scale"])

    # determine the correct breathing profile
    cs_t1, cs_t2, knots_1, knots_2 = (minimise_breathing(1.5,1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax,
                                                             Pmax_dot, E_rs, R_rs, P_ao))

    Input_Parameters = [g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa,
     C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra, C_sa,
     L_sa, R_sa, D1, K1_vc, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv,
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
     n, Pmax, Pmax_dot, E_rs, R_rs, P_ao,
     # added params
     Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
     Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
     R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
     Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vvc_max, Vu_pa, Vu_pp,
     Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
     tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
     tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
     DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
     scale_param1, scale_param2, scale_param3, scale_param4, scale_param5, scale_param6, scale_param7,
     Pa_O2_lower, rise_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale]

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        t_span,
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2)
    )


    if ODE_solution.status == -1:
        # Integration failed or early termination
        print("fail")
        return [0.0]*31, [0.0]*82

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT

    all_time = np.concatenate((local_updates["all_time"][i_buffer:], local_updates["all_time"][:i_buffer]))
    time_since_beat_store = np.concatenate((local_updates["time_since_beat_store"][i_buffer:], local_updates["time_since_beat_store"][:i_buffer]))
    finish_breath_time = np.concatenate((local_updates["finish_breath_time"][i_buffer:], local_updates["finish_breath_time"][:i_buffer]))

    dtb = np.diff(time_since_beat_store)
    dtr = np.diff(finish_breath_time)
    beat_idx = np.where(dtb > 0)[0] + 1
    breath_idx = np.where(dtr > 0)[0] + 1
    beat_idx = beat_idx[-1]
    breath_idx = breath_idx[-1]
    last_beat_t = all_time[beat_idx]
    last_breath_t = all_time[breath_idx]

    interp = interp1d(
        ODE_solution.t,
        ODE_solution.y,
        axis=1,
        kind="linear",
        fill_value="extrapolate"
    )

    state_last_beat = interp(last_beat_t)
    state_last_breath = interp(last_breath_t)
    combined = np.concatenate((state_last_beat[:57], state_last_breath[57:]))

    P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
    peaks, _ = find_peaks(P_sa, distance=int(1000))

    last_10_peaks_P_sa = peaks[-11:-1]
    last_10_max_P_sa = P_sa[last_10_peaks_P_sa]

    theta_ao = np.concatenate((local_updates["theta_ao_store"][i_buffer:], local_updates["theta_ao_store"][:i_buffer]))
    theta_po = np.concatenate((local_updates["theta_po_store"][i_buffer:], local_updates["theta_po_store"][:i_buffer]))
    theta_mi = np.concatenate((local_updates["theta_mi_store"][i_buffer:], local_updates["theta_mi_store"][:i_buffer]))
    theta_tr = np.concatenate((local_updates["theta_tr_store"][i_buffer:], local_updates["theta_tr_store"][:i_buffer]))

    V_rv = np.concatenate((local_updates["V_rv_store"][i_buffer:], local_updates["V_rv_store"][:i_buffer]))
    V_ra = np.concatenate((local_updates["V_ra_store"][i_buffer:], local_updates["V_ra_store"][:i_buffer]))
    V_la = np.concatenate((local_updates["V_la_store"][i_buffer:], local_updates["V_la_store"][:i_buffer]))

    N = 100  # number of consecutive closed samples required

    is_open = theta_ao > theta_min
    open_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_open[k] and not np.any(is_open[k - N:k]):
            open_idx1.append(k)
    open_idx1 = np.array(open_idx1)[-11:-1]

    # is_closed_ao = theta_ao <= theta_min
    # close_idx1 = []
    # for k in range(N, len(theta_ao)):
    #     if is_closed_ao[k] and not np.any(is_closed_ao[k - N:k]):
    #         close_idx1.append(k)
    # close_idx1 = np.array(close_idx1)[-11:-1]

    is_open_po = theta_po > theta_min
    open_idx2 = []
    for k in range(N, len(theta_po)):
        if is_open_po[k] and not np.any(is_open_po[k - N:k]):
            open_idx2.append(k)
    open_idx2 = np.array(open_idx2)

    is_closed_po = theta_po <= theta_min
    close_idx2 = []
    for k in range(N, len(theta_po)):
        if is_closed_po[k] and not np.any(is_closed_po[k - N:k]):
            close_idx2.append(k)
    close_idx2 = np.array(close_idx2)

    is_open_mi = theta_mi > theta_min
    open_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_open_mi[k] and not np.any(is_open_mi[k - N:k]):
            open_idx3.append(k)
    open_idx3 = np.array(open_idx3)

    is_closed_mi = theta_mi <= theta_min
    close_idx3 = []
    for k in range(N, len(theta_mi)):
        if is_closed_mi[k] and not np.any(is_closed_mi[k - N:k]):
            close_idx3.append(k)
    close_idx3 = np.array(close_idx3)

    is_open_tr = theta_tr > theta_min
    open_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_open_tr[k] and not np.any(is_open_tr[k - N:k]):
            open_idx4.append(k)
    open_idx4 = np.array(open_idx4)

    is_closed_tr = theta_tr <= theta_min
    close_idx4 = []
    for k in range(N, len(theta_tr)):
        if is_closed_tr[k] and not np.any(is_closed_tr[k - N:k]):
            close_idx4.append(k)
    close_idx4 = np.array(close_idx4)

    pairs_po = np.array([
        (o, close_idx2[(close_idx2 > o) & (close_idx2 < o_next)][-1])
        for o, o_next in zip(open_idx2[:-1], open_idx2[1:])
        if np.any((close_idx2 > o) & (close_idx2 < o_next))])

    pairs_mi = np.array([
        (o, close_idx3[(close_idx3 > o) & (close_idx3 < o_next)][-1])
        for o, o_next in zip(open_idx3[:-1], open_idx3[1:])
        if np.any((close_idx3 > o) & (close_idx3 < o_next))])

    pairs_tr = np.array([
        (o, close_idx4[(close_idx4 > o) & (close_idx4 < o_next)][-1])
        for o, o_next in zip(open_idx4[:-1], open_idx4[1:])
        if np.any((close_idx4 > o) & (close_idx4 < o_next))])

    pairs_po = pairs_po[-11:-1]
    pairs_mi = pairs_mi[-11:-1]
    pairs_tr = pairs_tr[-11:-1]

    # Max pressure during atrial contraction takes the max p between phi_atr = 0 & 1
    phi_atr = np.concatenate((local_updates["phi_atr_store"][i_buffer:], local_updates["phi_atr_store"][:i_buffer]))

    dphi = np.diff(phi_atr, prepend=phi_atr[0])
    is_rising = dphi > 0
    edges = np.diff(is_rising.astype(int))
    start_idx = np.where(edges == 1)[0] + 1
    end_idx = np.where(edges == -1)[0] + 1

    n_pairs = min(len(start_idx), len(end_idx))
    # If first end comes before first start, skip that end
    if len(end_idx) > 0 and len(start_idx) > 0 and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]
        n_pairs = min(len(start_idx), len(end_idx))

    # Truncate to matching pairs
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    P_la = np.concatenate((local_updates["P_la_store"][i_buffer:], local_updates["P_la_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_la_max_idx = np.array([s + np.argmax(P_la[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    # period of V descent when mitral valve is open -> get second min la P
    P_la_descent2_idx = np.array([o + np.argmin(P_la[o:c]) for o, c in pairs_mi])
    P_la_descent1_idx = np.array(
        [c + np.argmin(P_la[c:o_next]) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])])

    P_ra = np.concatenate((local_updates["P_ra_store"][i_buffer:], local_updates["P_ra_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    # period of V descent when tricuspid valve is open -> get second min la P
    P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs_tr])
    P_ra_descent1_idx = np.array(
        [c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])])

    V_lv = np.concatenate((local_updates["V_lv_store"][i_buffer:], local_updates["V_lv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)

    last_10_troughs_V_lv = troughs[-11:-1]
    last_10_min_V_lv = V_lv[last_10_troughs_V_lv]

    last_10_peaks_V_lv = peaks[-11:-1]
    last_10_max_V_lv = V_lv[last_10_peaks_V_lv]

    P_rv = np.concatenate((local_updates["P_rv_store"][i_buffer:], local_updates["P_rv_store"][:i_buffer]))
    P_rv_max_idx = np.array([o + np.argmax(P_rv[o:c]) for o, c in pairs_po])
    P_rv_min_idx = np.array([c + np.argmin(P_rv[c:o_next]) for (_, c), (o_next, _) in zip(pairs_po[:-1], pairs_po[1:])])

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

    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_mins = starts[-11:-1]
    last_10_b4_LA_atrial_contract = V_la[local_mins]
    last_10_b4_RA_atrial_contract = V_ra[local_mins]

    tidal = np.concatenate((local_updates["tidal_store"][i_buffer:], local_updates["tidal_store"][:i_buffer]))
    peaks, _ = find_peaks(tidal, distance=int(1000))
    last_10_peaks_tidal = peaks[-1]
    max_tidal = tidal[last_10_peaks_tidal]

    VAflow = np.concatenate((local_updates["VAflow_store"][i_buffer:], local_updates["VAflow_store"][:i_buffer]))
    t1 = np.concatenate((local_updates["t1_store"][i_buffer:], local_updates["t1_store"][:i_buffer]))
    t2 = np.concatenate((local_updates["t2_store"][i_buffer:], local_updates["t2_store"][:i_buffer]))
    VD = GV_dead * VAflow[-1] + V0_dead
    VDflow = (1 / (t1[-1] + t2[-1])) * VD
    Minute_Ventilation = (VAflow[-1] + VDflow) * 60

    cardiac_output = np.mean(local_updates["Q_pp_store"])
    Pa_O2 = np.mean(local_updates["Pa_O2_every_store"])
    Pa_CO2 = np.mean(local_updates["Pa_CO2_every_store"])

    Total_Volume = V_ra + V_rv + V_lv + V_la
    is_active = phi_atr > 0.0  # atrial contraction window
    edges = np.diff(is_active.astype(int))

    start_idx = np.where(edges == 1)[0] + 1  # 0 → active
    end_idx = np.where(edges == -1)[0] + 1  # active → 0

    if len(start_idx) and len(end_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    n_pairs = min(len(start_idx), len(end_idx))
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    Total_Vol_min_idx = np.array([s + np.argmin(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]
    Total_Vol_max_idx = np.array([s + np.argmax(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    mean_min_Total_Volume = np.mean(Total_Volume[Total_Vol_min_idx])
    mean_max_Total_Volume = np.mean(Total_Volume[Total_Vol_max_idx])
    Pericardial_Volume_difference = mean_max_Total_Volume - mean_min_Total_Volume
    Vol_percentage_change = Pericardial_Volume_difference / mean_max_Total_Volume

    dP_lv_dt_store = np.concatenate(
        (local_updates["dP_lv_dt_store"][i_buffer:], local_updates["dP_lv_dt_store"][:i_buffer]))
    dP_lv_dt_idx = np.array([s + np.argmax(dP_lv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    dP_rv_dt_store = np.concatenate(
        (local_updates["dP_rv_dt_store"][i_buffer:], local_updates["dP_rv_dt_store"][:i_buffer]))
    dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    print(np.mean(P_sa[open_idx1]), np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_min_idx]), np.mean(P_la[P_la_descent1_idx]), Vol_percentage_change)

    return ([np.mean(past_10_flat_segments), np.mean(last_10_max_P_sa), np.mean(P_sa[open_idx1]),
            np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(V_rv[pairs_po[:, 0]]), np.mean(V_rv[pairs_po[:, 1]]),
            np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_min_idx]),
            np.mean(V_ra[pairs_tr[:, 1]]), np.mean(V_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent1_idx]),
            np.mean(P_ra[P_ra_max_idx]), np.mean(P_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent2_idx]),
            np.mean(V_la[pairs_mi[:, 1]]), np.mean(V_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent1_idx]),
            np.mean(P_la[P_la_max_idx]), np.mean(P_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent2_idx]),
            np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
            np.mean(dP_lv_dt_store[dP_lv_dt_idx]), np.mean(dP_rv_dt_store[dP_rv_dt_idx]), max_tidal, Minute_Ventilation,
            cardiac_output, Pa_O2, Pa_CO2, Vol_percentage_change], combined)


#
def chunked(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def timeout_handler(signum, frame):
    raise TimeoutError("Simulation timeout")

def safe_simulate_cpu(params, storage, old_parameters, timeout=300):
    try:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        result = simulate_cpu(params, storage, old_parameters)
        signal.alarm(0)  # Cancel timeout
        return result
    except Exception:
        signal.alarm(0)  # Cancel timeout
        print("too slow")
        return ([0.0] * 31, [0.0] * 82)


def parallel_simulations(param_samples, storage, n_jobs, chunk_size=2500, save_path_results='Result_chunked.npy', save_path_states='States_chunked.npy'):
    results_all = []
    final_states = []

    # If file exists from previous run, remove it to start fresh
    if os.path.exists(save_path_results):
        os.remove(save_path_results)
    if os.path.exists(save_path_states):
        os.remove(save_path_states)

    for i, chunk in enumerate(chunked(param_samples, chunk_size)):
        with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i + 1) * chunk_size}", total=len(chunk))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(safe_simulate_cpu)(params, copy.deepcopy(storage), Old_Parameters) for params in chunk)

        results_block = [res[0] for res in results]
        final_states_block = [res[1] for res in results]
        results_all.extend(results_block)
        final_states.extend(final_states_block)

        # Optional: also accumulate in a single array
        np.save(save_path_results, np.array(results_all))  # full file overwritten
        np.save(save_path_states, np.array(final_states))  # full file overwritten

    return results_all


# def parallel_simulations(param_samples, storage, chunk_size=3, save_path_results='Result_chunked123.npy', save_path_states='States_chunked123.npy'):
#     results_all = []
#     final_states_all = []
#
#     # If file exists from previous run, remove it to start fresh
#     if os.path.exists(save_path_results):
#         os.remove(save_path_results)
#     if os.path.exists(save_path_states):
#         os.remove(save_path_states)
#
#     for i, chunk in enumerate(chunked(param_samples, chunk_size)):
#         results = []
#         final_states = []
#         for params in tqdm(chunk, desc=f"Sim {i * chunk_size}-{(i+1)*chunk_size}"):
#             res = simulate_cpu(params, copy.deepcopy(storage), Old_Parameters)
#             results.append(res[0])
#             final_states.append(res[1])
#
#         results_all.extend(results)
#         final_states_all.extend(final_states)
#
#         # Save progressively (overwrites with accumulated results)
#         np.save(save_path_results, np.array(results_all))
#         np.save(save_path_states, np.array(final_states_all))
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

    lower = 0.5
    upper = 1.5

    sp = ProblemSpec({
        'names': [
            # gas
            "beta2", "C2", "K2", "a2",
            "alpha2", "dc", "KCCO2", "GV_dead",
            # resp control
            "KcCO2", "KcMRV", "KpCO2", "KpO2",
            "V0_dead", "VA_rest", "Pmax", "Pmax_dot",
            "E_rs", "R_rs",
            # cardio
            "C_jp", "C_sa", "L_sa", "R_sa",
            "C_amv", "C_bv", "C_ev", "C_hv",
            "C_rmv", "C_sv", "kr_am", "P_0",
            "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
            "R_rmv_n", "R_sv_n", "D1", "K1_vc",
            "Kr_vc", "Rvc_n", "C_pa", "C_pp",
            "C_pv", "L_pa", "R_pa", "R_pp",
            "R_pv", "Emax_la", "P0_la", "Emax_ra",
            "P0_ra", "KE_la", "KE_ra", "P0_lv",
            "P0_rv", #"g_thor", "P_thormax_n", "P_thormin_n",
            "VT_n", "s",
            # cardio control
            "fab_o", "fes_o", "fes_inf", "fes_max",
            "fev_o", "fev_inf", "kes", "kev",
            "Io_sh", "Io_sp", "Io_sv", "Io_v",
            "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v",
            "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
            "Ysv_max", "Ysv_min", "Yv_max", "Yv_min",
            "theta_v", "Wb_sh", "Wb_sp", "Wb_sv",
            "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v",
            "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v",
            "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
            "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
            "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
            "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
            "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
            #
            "R_sp0", "AT", "g_ccsh", "g_ccsp",
            "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv",
            "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
            "theta_spn", "theta_svn", "x_sh", "x_sp",
            "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
            "k_ab", "P_n", "P_n_max","f_acCO2_n",
            "f_ac_max", "f_ac_min", "k_ac", "K_H",
            "PaO2_ac_n", "G_ap", "GT_s", "GT_v",
            "T0", "A", "B", "C",
            "D", "Cvb_O2_n", "gb_O2", "MO2_bp",
            "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2",
            "grm_O2", "Kh_CO2", "Krm_CO2", "MO2_hpn",
            "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n",
            "gam_O2", "gM", "Io_met", "kmet",
            "MO2_ampn", "phi_max", "phi_min",
            # added params
            "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max",
            "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max",
            "Kp_po", "Kf_po", "Kb_po", "Kv_po", "theta_po_max",
            "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max",
            "alpha_O2", "R_po", "R_mi", "R_tr",
            "R_ao", "C_O2_param1", "C_O2_param2", "C_O2_param3",
            "PAMO2_nominal", "Vu_sa", "Vu_bv", "Vu_hv",
            "Vu_jp", "Vu_vc", "Vvc_max", "Vu_pa",
            "Vu_pp", "Vu_pv", "Vu_la", "Vu_lv",
            "Vu_ra", "Vu_rv",

            # "V_tot",
            "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
            "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv",
            "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0",
            "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc",
            "tau_isc", "tau_p", "tau_z", "tau_ac",
            "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2",
            "tau_O2", "tau_w", "tau_M", "tau_met",
            "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep",
            "DR_rmp", "DR_sp", "DV_amv", "DV_ev",
            "DV_rmv", "DV_sv", "DT_s", "DT_v",
            "Dmet", "Ta", "KE_lv", "KE_rv",
            "T1", "T2", "VL_CO2", "VL_O2",
            "KCSFCO2", "VB", "tauMR", "VTCO2",
            "VTO2", "tau_MRV",

            # further added
            "scale_param1", "scale_param2", "scale_param3", "scale_param4",
            "scale_param5", "scale_param6", "scale_param7", "Pa_O2_lower",
            "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
            "theta_min", "r", "l", "V_nominal", "V_scale"
        ],

        'bounds': [
            # gas
            [0.03255 * 0.9, 0.03255 * 1.1], [87 * 0.9, 87 * 1.1], [194.4 * 0.9, 194.4 * 1.1], [1.819 * 0.9, 1.819 * 1.1],
            [0.05591 * 0.9, 0.05591 * 1.1], [0.015 * lower, 0.015 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
            # resp control
            [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
            [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper], [100 * lower, 100 * upper], [1000 * lower, 1000 * upper],
            [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
            # cardio
            [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.2 * lower, 0.2 * upper],
            [4.4 * lower, 4.4 * upper], [5.71 * lower, 5.71 * upper], [10 * lower, 10 * upper], [1.57 * lower, 1.57 * upper],
            [3.28 * lower, 3.28 * upper], [31.11 * lower, 31.11 * upper], [24.17 * lower, 24.17 * upper], [3.93 * lower, 3.93 * upper],
            [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
            [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
            [0.0001 * lower, 0.0001 * upper], [0.0025 * lower, 0.0025 * upper], [0.76 * lower, 0.76 * upper], [15.8 * lower, 15.8 * upper],
            [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
            [0.0056 * lower, 0.0056 * upper], [0.30 * lower, 0.30 * upper], [0.55 * lower, 0.55 * upper], [0.34 * lower, 0.34 * upper],
            [0.55 * lower, 0.55 * upper], [0.05 * lower, 0.05 * upper], [0.09 * lower, 0.09 * upper], [1.5 * lower, 1.5 * upper],
            [1.5 * lower, 1.5 * upper], # [6.8 * lower, 6.8 * upper], [-2 * 1.5, -2 * 0.5], [-6 * 1.5, -6 * 0.5],
            [0.73 * lower, 0.73 * upper], [0.04 * lower, 0.04 * upper],
            # cardio control
            [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
            [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper],
            [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.70 * lower, 0.70 * upper], [0.22 * lower, 0.22 * upper],
            [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
            [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper], [-0.037 * upper, -0.037 * lower],
            [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower],
            [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
            [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper], [0.2 * lower, 0.2 * upper],
            [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
            [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
            [1.392 * lower, 1.392 * upper], [0.8 * lower, 0.8 * upper], [2.66 * lower, 2.66 * upper], [0.475 * lower, 0.475 * upper],
            [0.282 * lower, 0.282 * upper], [4.47 * lower, 4.47 * upper], [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper],
            [0.695 * lower, 0.695 * upper], [-28.29 * upper, -28.29 * lower], [-74.21 * upper, -74.21 * lower], [-28.29 * upper, -28.29 * lower],
            [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper], [5.27 * lower, 5.27 * upper],
            #
            [2.49 * lower, 2.49 * upper], [(1 / 60) * lower, (1 / 60) * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
            [0.2 * lower, 0.2 * upper], [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
            [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper], [6 * lower, 6 * upper],
            [6 * lower, 6 * upper], [40 * 0.9, 40 * 1.1], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
            [11.76 * lower, 11.76 * upper], [80 * lower, 80 * upper], [112 * 0.9, 112 * upper], [1.4 * lower, 1.4 * upper],
            [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper], [3 * lower, 3 * upper],
            [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper],
            [0.8 * lower, 0.8 * upper],  [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
            [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper], [0.925 * lower, 0.925 * upper],
            [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper],
            [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
            [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper], [0.1555 * lower, 0.1555 * upper],
            [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper], [0.18 * lower, 0.18 * upper],
            [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
            # added params
            [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [1.309 * lower, 1.309 * upper],
            [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
            [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [5 * lower, 5 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
            [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [10 * lower, 10 * upper], [1.309 * lower, 1.309 * upper],
            [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper], [350 * lower, 350 * upper],
            [350 * lower, 350 * upper], [0.00134 * 0.9, 0.00134 * 1.1], [2.6 * 0.9, 2.6 * 1.1], [3.03e-5 * 0.9, 3.03e-5 * 1.1],
            [104 * lower, 104 * upper], [1 * lower, 1 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
            [879.76 * lower, 879.76 * upper], [123 * lower, 123 * upper], [350 * lower, 350 * upper], [1.0 * lower, 1.0 * upper],
            [116.6775 * lower, 116.6775 * upper], [214 * lower, 214 * upper], [20 * lower, 20 * upper], [60 * lower, 60 * upper],
            [50 * lower, 50 * upper], [80 * lower, 80 * upper],

            # [5027.6 * 0.8, 5027.6 * 1.2],
            [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
            [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
            [807.8 * lower, 807.8 * upper], [190.95 * lower, 190.95 * upper], [1661.6 * lower, 1661.6 * upper], [20 * lower, 20 * upper],
            [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
            [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
            [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
            [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper], [0.015 * lower, 0.015 * upper],
            [0.1 * lower, 0.1 * upper], [0.1 * lower, 0.1 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
            [20 * lower, 20 * upper], [0.01 * lower, 0.01 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
            [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],

            # further added params
            [4.9 * lower, 4.9 * upper], [1.5 * lower, 1.5 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
            [0.5 * lower, 0.5 * upper], [1.2 * lower, 1.2 * upper], [30 * lower, 30 * upper], [80 * lower, 80 * upper],
            [0.05 * lower, 0.05 * upper], [0.15 * lower, 0.15 * upper], [0.3 * 0.8, 0.3 * 1.2], [0.9 * 0.95, 0.9 * 1.05],
            [0.0872665 * lower, 0.0872665 * upper], [1.5 * 0.67, 1.5 * upper], [2.0 * 0.5, 2.0 * upper], [380 * lower, 380 * upper], [40 * lower, 40 * upper]]
    })


# No P_thor: 52 parameters contribute 90% sensitivity for 30 targets
    subset_vars = {'C_jp','C_pa','C_pp','C_pv','C_sa','Cvam_O2_n','Cvrm_O2_n',
    'Emax_la','Emax_lv0','Emax_ra','Emax_rv0','GT_s','GT_v','G_ap','Io_met','Io_sv',
    'K1_vc','K2','KE_la','KE_lv','KE_ra','KE_rv', 'Kp_po','Kp_tr','Kv_po','Kv_tr',
    'P0_la','P0_lv','P0_ra','P0_rv','P_n','R_ep0','R_pa','R_po','R_pp','R_sa','R_sp0',
    'T0','V_nominal','V_scale','Vu_amv0','Vu_bv','Vu_ev0','Vu_jp','Vu_pv','Vu_rmv0','Vu_sv0',
    'Wb_sh','Wb_sp','Wb_sv','Wp_v','a2','f_ab_max','fab_o','fall_time_ven','fes_inf','fes_min','fes_o',
    'fev_inf','fev_o','k_ab','kcc_sv','kes','kmet','l','r','rise_time_atr','rise_time_ven',
    'theta_spn','theta_svn','theta_tr_max','theta_v'
}

    # Filter the names and bounds
    filtered_names = []
    filtered_bounds = []

    for name, bound in zip(sp["names"], sp["bounds"]):
        filtered_names.append(name)
        if name in subset_vars:
            filtered_bounds.append([round(bound[0], 12), round(bound[1], 12)])
        else:
            # take nominal (mean of lower/upper)
            filtered_bounds.append([np.mean(bound), np.mean(bound)])

    # Create a new ProblemSpec with only the filtered variables
    sp_filtered = ProblemSpec({
        "names": filtered_names,
        "bounds": filtered_bounds
    })

    param_keys = list(sp_filtered["names"])

    # change
    X = sample_inputs_from_spec(sp_filtered, n_samples=200000, random_seed=42, method="lhs")
    X = X.cpu().numpy() if X.is_cuda else X.numpy()
    np.save('LHCS_just_check.npy', X)
    # X = np.load('LHCS_just_check.npy')

    # # make sure only the set number of parameters are adjusted
    # mask = np.ptp(X, axis=0) != 0
    # print(np.sum(mask))

    param_samples = [dict(zip(param_keys, row)) for row in X]
    # param_samples[0] = {"beta2":0.03255,"C2":87.0,"K2":194.4,"a2":1.819,"alpha2":0.05591,"dc":0.015,"KCCO2":346000.0,"GV_dead":0.1698,"KcCO2":0.2332,"KcMRV":1.0,"KpCO2":0.2025,"KpO2":4.72e-09,"V0_dead":0.1587,"VA_rest":0.0673,"Pmax":100.0,"Pmax_dot":1000.0,"E_rs":21.9,"R_rs":3.02,"C_jp":3.72,"C_sa":0.28,"L_sa":0.00022,"R_sa":0.2,"C_amv":4.4,"C_bv":5.71,"C_ev":10.0,"C_hv":1.57,"C_rmv":3.28,"C_sv":31.11,"kr_am":24.17,"P_0":3.93,"R_amv_n":0.0833,"R_bv_n":0.075,"R_ev_n":0.04,"R_hv_n":0.224,"R_rmv_n":0.125,"R_sv_n":0.038,"D1":0.3855,"K1_vc":0.15,"Kr_vc":0.0001,"Rvc_n":0.0025,"C_pa":0.76,"C_pp":15.8,"C_pv":25.37,"L_pa":0.00018,"R_pa":0.023,"R_pp":0.0894,"R_pv":0.0056,"Emax_la":0.34,"P0_la":0.55,"Emax_ra":0.34,"P0_ra":0.55,"KE_la":0.05,"KE_ra":0.07,"P0_lv":1.5,"P0_rv":1.5,"g_thor":6.8,"P_thormax_n":-0.0,"P_thormin_n":-0.0,"VT_n":0.73,"s":0.04,"fab_o":25.0,"fes_o":16.11,"fes_inf":2.1,"fes_max":80.0,"fev_o":3.2,"fev_inf":6.3,"kes":0.0675,"kev":7.06,"Io_sh":0.658,"Io_sp":0.65,"Io_sv":0.45,"Io_v":0.22,"kcc_sh":0.114,"kcc_sp":0.13,"kcc_sv":0.09,"kcc_v":0.0162,"Ysh_max":20.0,"Ysh_min":-0.0283,"Ysp_max":5.5,"Ysp_min":-0.037,"Ysv_max":64.9,"Ysv_min":-0.437,"Yv_max":1.9,"Yv_min":-0.0008,"theta_v":-0.68,"Wb_sh":-1.75,"Wb_sp":-1.1375,"Wb_sv":-1.1375,"Wc_sh":1.0,"Wc_sp":1.716,"Wc_sv":1.716,"Wc_v":0.2,"Wp_sh":-0.2,"Wp_sp":-0.3997,"Wp_sv":-0.3997,"Wp_v":-0.103,"Wt_sh":0.4,"Wt_sp":0.4,"Wt_sv":0.4,"Wt_v":0.4,"Emax_lv0":2.392,"Emax_rv0":1.412,"fes_min":2.66,"GEmax_lv":0.475,"GEmax_rv":0.282,"GR_amp":4.47,"GR_ep":1.94,"GR_rmp":2.47,"GR_sp":0.695,"GV_amv":-28.29,"GV_ev":-74.21,"GV_rmv":-28.29,"GV_sv":-265.4,"R_amp0":3.51,"R_ep0":1.655,"R_rmp0":5.27,"R_sp0":2.49,"AT":0.016666666667,"g_ccsh":1.0,"g_ccsp":1.5,"g_ccsv":0.2,"kisc_sh":6.0,"kisc_sp":2.0,"kisc_sv":2.0,"PO2_sh":45.0,"PO2_sp":30.0,"PO2_sv":30.0,"theta_shn":3.6,"theta_spn":13.32,"theta_svn":13.32,"x_sh":53.0,"x_sp":6.0,"x_sv":6.0,"PaCO2_n":40.0,"f_ab_max":47.78,"f_ab_min":2.52,"k_ab":11.76,"P_n":92.0,"P_n_max":112.0,"f_acCO2_n":1.4,"f_ac_max":12.3,"f_ac_min":0.835,"k_ac":29.27,"K_H":3.0,"PaO2_ac_n":45.0,"G_ap":11.76,"GT_s":-0.13,"GT_v":0.09,"T0":0.58,"A":20.9,"B":92.8,"C":10570.0,"D":-5.251,"Cvb_O2_n":0.14,"gb_O2":10.0,"MO2_bp":0.925,"R_bpn":6.57,"Cvh_O2_n":0.11,"Cvrm_O2_n":0.155,"gh_O2":35.0,"grm_O2":30.0,"Kh_CO2":11.11,"Krm_CO2":142.8,"MO2_hpn":0.4,"MO2_rmp":0.86,"R_hpn":19.71,"W_hn":12660.0,"Cvam_O2_n":0.1555,"gam_O2":30.0,"gM":40.0,"Io_met":0.4266,"kmet":0.18,"MO2_ampn":0.516,"phi_max":20.0,"phi_min":-1.87,"Kp_ao":1000.0,"Kf_ao":5000.0,"Kb_ao":2.0,"Kv_ao":5.0,"theta_ao_max":1.309,"Kp_mi":2000.0,"Kf_mi":200.0,"Kb_mi":2.0,"Kv_mi":10.0,"theta_mi_max":1.309,"Kp_po":2000.0,"Kf_po":2000.0,"Kb_po":5.0,"Kv_po":10.0,"theta_po_max":1.309,"Kp_tr":3000.0,"Kf_tr":500.0,"Kb_tr":2.0,"Kv_tr":7.0,"theta_tr_max":1.309,"alpha_O2":3.17e-05,"R_po":350.0,"R_mi":350.0,"R_tr":350.0,"R_ao":350.0,"C_O2_param1":0.00134,"C_O2_param2":2.6,"C_O2_param3":3.03e-05,"PAMO2_nominal":104.0,"Vu_sa":1.0,"Vu_bv":279.49,"Vu_hv":93.16,"Vu_jp":579.76,"Vu_vc":123.0,"Vvc_max":350.0,"Vu_pa":1.0,"Vu_pp":116.6775,"Vu_pv":214.0,"Vu_la":10.0,"Vu_lv":10.0,"Vu_ra":10.0,"Vu_rv":10.0,"V_tot":5027.6,"tau_Emax_lv":8.0,"tau_Emax_rv":8.0,"tau_Ramp":2.0,"tau_Rep":2.0,"tau_Rrmp":2.0,"tau_Rsp":2.0,"tau_Vamv":20.0,"tau_Vev":20.0,"tau_Vrmv":20.0,"tau_Vsv":20.0,"Vu_amv0":286.4,"Vu_ev0":607.8,"Vu_rmv0":190.95,"Vu_sv0":1361.6,"tau_cc":20.0,"tau_isc":30.0,"tau_p":2.076,"tau_z":0.8,"tau_ac":2.0,"tau_ap":2.0,"tau_Ts":2.0,"tau_Tv":1.5,"tau_CO2":20.0,"tau_O2":10.0,"tau_w":5.0,"tau_M":40.0,"tau_met":10.0,"DEmax_lv":2.0,"DEmax_rv":2.0,"DR_amp":2.0,"DR_ep":2.0,"DR_rmp":2.0,"DR_sp":2.0,"DV_amv":5.0,"DV_ev":5.0,"DV_rmv":5.0,"DV_sv":5.0,"DT_s":2.0,"DT_v":0.2,"Dmet":4.0,"Ta":3.0,"KE_lv":0.014,"KE_rv":0.011,"T1":1.0,"T2":2.0,"VL_CO2":3.0,"VL_O2":2.5,"KCSFCO2":20.0,"VB":0.09,"tauMR":50.0,"VTCO2":0.25,"VTO2":0.25,"tau_MRV":50.0,"scale_param1":4.9,"scale_param2":1.5,"scale_param3":0.3,"scale_param4":26.6,"scale_param5":0.5,"scale_param6":1.2,"scale_param7":30.0,"Pa_O2_lower":80.0,"rise_time_atr":0.05,"rise_time_ven":0.15,"fall_time_ven":0.3,"ahead1":0.9,"theta_min":0.0872665,"r":1.2,"l":2.0,"V_nominal":280.0,"V_scale":40.0}

    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=-1)
    # Result = parallel_simulations(param_samples, Next_Conditions)

    # print(Result)
    # change
    np.save('LHCS_just_check_result.npy', Result)









