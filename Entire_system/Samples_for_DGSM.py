import os
import copy

import numpy as np
from SALib import ProblemSpec
from SALib.sample import finite_diff
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective

from tqdm import tqdm
import tqdm_joblib

from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from All_Parameter_ranges import parameters as parameters_change
from All_derivatives_njit import model_derivatives
from All_Cardiovascular_controller import cardiovascular_controller
from All_Cardiovascular_system import cardiovascular_system
from All_Gas_exchange import gas_exchange
from Parameters import Parameters as Old_Parameters
from All_Respiratory_controller import resp_control_vent


from Selected_Conditions import Selected_Conditions as previous_Selected_Conditions
from Initial_Conditions_after_running_again import Initial_Conditions
# from All_Next_Conditions import Next_Conditions
from All_Next_Conditions import Next_Conditions

from Parameters_test import Parameters as Para


target_values = np.arange(0, 10000, 10)
time_saved = 0.005
BUFFER_LIMIT = 20000

max_time = 400 # Maximum time limit to avoid infinite loops
time_step = 10  # Chunk size per solve

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

    # Debugging check for progress
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
                           "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc",
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
    dt = 0.001
    bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
    tolerance = 0.0001

    VAflow_vals = np.linspace(0.06, 1.2, 200)
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

    return c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6


def simulate_cpu(Current_Parameters, storage,  old_parameters, IC_initial=None):
    local_updates = {key: copy.deepcopy(value) for key, value in storage.items()}

    if IC_initial is None:
        IC_current = IC_overall.copy()
        t_span = [0, max_time]
    else:
        IC_current = IC_initial.copy()
        t_span = [max_time, max_time + 200]

    # Cardio parameters
    (A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, Vu_pa, Vu_pp, Vu_pv, KE_lv, KE_rv, P0_lv, P0_rv, Vu_la, Vu_lv, Vu_ra, Vu_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, Vu_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n, Vu_vc, Vvc_max, Vvc_min,
    C_jp, V_tot, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    Vu_ep, Vu_sp, Vu_bp, Vu_hp, Vu_rmp, Vu_amp, kr_am, Vu_bv, Vu_hv) = (
    Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
    ["A_im", "Tc", "T_im", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa",
     "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Vu_pa", "Vu_pp", "Vu_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv",
     "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "Emax_la", "P0_la", "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa",
     "R_sa", "Vu_sa", "D1", "D2", "K1_vc", "K2_vc", "Kr_vc", "Rvc_n", "Vu_vc", "Vvc_max", "Vvc_min", "C_jp", "V_tot",
     "R_ev_n", "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv",
     "Vu_ep", "Vu_sp", "Vu_bp", "Vu_hp", "Vu_rmp", "Vu_amp", "kr_am", "Vu_bv", "Vu_hv"])

    # Cardio controller parameters
    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
    kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
    Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv,
    GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, tau_Emax_lv,
    tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0,
    Vu_sv0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, tau_cc,
    tau_isc, theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max, tau_p, tau_z,
    f_acCO2_n, f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, tau_ac, G_ap, tau_ap, DT_v, GT_s, GT_v, T0, tau_Ts, tau_Tv, A, B, C, D,
    Cvb_O2_n, gb_O2, R_bpn, tau_CO2, tau_O2, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
    MO2_rmp, R_hpn, tau_w, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min, tau_M, tau_met) = \
    [Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in
     ["fab_o", "fes_o", "fes_inf", "fes_max", "fev_o",
      "fev_inf", "kes", "kev", "Io_sh", "Io_sp", "Io_sv", "Io_v", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max",
      "Ysh_min", "Ysp_max", "Ysp_min", "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp",
      "Wb_sv", "Wc_sh", "Wc_sp", "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
      "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
      "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp",
      "tau_Rep", "tau_Rrmp", "tau_Rsp", "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0",
      "Vu_rmv0", "Vu_sv0", "AT", "g_ccsh", "g_ccsp", "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh",
      "PO2_sp", "PO2_sv", "tau_cc", "tau_isc", "theta_shn", "theta_spn", "theta_svn", "x_sh", "x_sp", "x_sv",
      "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max", "tau_p", "tau_z", "f_acCO2_n", "f_ac_max", "f_ac_min",
      "k_ac", "K_H", "PaO2_ac_n", "tau_ac", "G_ap", "tau_ap", "DT_v", "GT_s", "GT_v", "T0", "tau_Ts", "tau_Tv", "A", "B", "C", "D",
      "Cvb_O2_n", "gb_O2", "R_bpn", "tau_CO2", "tau_O2", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
      "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "tau_w", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "Io_met",
      "kmet", "MO2_ampn", "phi_max", "phi_min", "tau_M", "tau_met"]]

    # Gas exchange and mixing
    (a2_gas, alpha2, beta2, C2, Fi_CO2, Fi_O2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm,
     P_ws, T1, T2, VL_CO2, VL_O2, Z, dc, KCCO2, KCSFCO2, MRBCO2, MO2_bp, VB, MRTCO2_basal, MRTO2_basal, tauMR,
     VTCO2, VTO2, MRCO2, MRO2, tau_MRV, s, Ta) = (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in [
    "a2", "alpha2", "beta2", "C2", "Fi_CO2", "Fi_O2", "K2", "PACO2_Delay_IC",
    "PAO2_Delay_IC", "P_atm", "P_ws", "T1", "T2", "VL_CO2", "VL_O2", "Z", "dc", "KCCO2", "KCSFCO2", "MRBCO2",
    "MO2_bp", "VB", "MRTCO2_basal", "MRTO2_basal", "tauMR", "VTCO2", "VTO2", "MRCO2", "MRO2", "tau_MRV", "s", "Ta"])

    # Resp control
    (GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
    (Current_Parameters[k] if k in Current_Parameters else old_parameters[k] for k in ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
   "V0_dead", "VA_rest", "lambda1", "lambda2", "n", "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"])

    # determine the correct breathing profile
    c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6 = minimise_breathing(Next_Conditions["t1_store"][0],
    Next_Conditions["t2_store"][0], GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao)

    Input_Parameters = [A_im, Tc, T_im, g_abd, g_thor, P_abdmax_n, P_abdmin_n, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, Vu_pa, Vu_pp, Vu_pv, KE_lv, KE_rv, P0_lv, P0_rv, Vu_la, Vu_lv, Vu_ra, Vu_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, Vu_sa, D1, D2, K1_vc, K2_vc, Kr_vc, Rvc_n, Vu_vc, Vvc_max, Vvc_min,
    C_jp, V_tot, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    Vu_ep, Vu_sp, Vu_bp, Vu_hp, Vu_rmp, Vu_amp, kr_am, Vu_bv, Vu_hv,
    fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
    kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
    Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv,
    GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, tau_Emax_lv,
    tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0,
    Vu_sv0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, tau_cc,
    tau_isc, theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n,  P_n_max, tau_p, tau_z, f_acCO2_n,
    f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, tau_ac, G_ap, tau_ap, DT_v, GT_s, GT_v, T0, tau_Ts, tau_Tv, A, B, C, D,
    Cvb_O2_n, gb_O2, R_bpn, tau_CO2, tau_O2, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
    MO2_rmp, R_hpn, tau_w, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min, tau_M, tau_met,
    a2_gas, alpha2, beta2, C2, Fi_CO2, Fi_O2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm,
    P_ws, T1, T2, VL_CO2, VL_O2, Z, dc, KCCO2, KCSFCO2, MRBCO2, MO2_bp, VB, MRTCO2_basal, MRTO2_basal, tauMR,
    VTCO2, VTO2, MRCO2, MRO2, tau_MRV, s, Ta,
    GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao,
    c0, c1, c2, c3, c4, c5, c6, d0, d1, d2, d3, d4, d5, d6]



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
        return [0.0, 0.0, 0.0], None, None

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT


    P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))
    peaks, _ = find_peaks(P_sa, distance=int(500))  # Adjust distance based on heart rate
    troughs, _ = find_peaks(-P_sa, distance=int(500))  # Find minima (inverted peaks)

    last_10_troughs = troughs[-10:-1]  # Get indices of last 5 minima
    last_10_min = P_sa[last_10_troughs]  # Get actual minimum values

    last_10_peaks = peaks[-10:-1]  # Get indices of last 5 max
    last_10_max = P_sa[last_10_peaks]  # Get actual max values


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
    print(np.mean(past_10_flat_segments))

    IC_current = ODE_solution.y[:, -1]

    return [np.mean(past_10_flat_segments), np.mean(last_10_max), np.mean(last_10_min)], IC_current, local_updates


#
# def chunked(iterable, n):
#     """Yield successive n-sized chunks from iterable."""
#     for i in range(0, len(iterable), n):
#         yield iterable[i:i + n]


# def parallel_simulations(param_samples, storage, n_jobs, chunk_size=5000, save_path='Result_DGSM_chunked.npy'):
#     results_all = []
#
#     # If file exists from previous run, remove it to start fresh
#     if os.path.exists(save_path):
#         os.remove(save_path)
#
#     for i, chunk in enumerate(chunked(param_samples, chunk_size)):
#         with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i+1)*chunk_size}", total=len(chunk))):
#             results_chunk = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, storage) for params in chunk)
#
#         results_all.extend(results_chunk)
#
#         # # Save chunk incrementally (appending)
#         # np.save(f'result_chunk_{i:03d}.npy', results_chunk)  # individual chunks
#
#         # Optional: also accumulate in a single array
#         np.save(save_path, np.array(results_all))  # full file overwritten
#
#     return results_all


def parallel_simulations(param_samples, storage, n_jobs, save_path='Result_DGSM_delay3.npy'):
    results_all = []

    if os.path.exists(save_path):
        os.remove(save_path)

    # Break into blocks of block_size (1 base + (block_size - 1) perturbations)
    block_size = 174
    param_blocks = [param_samples[i:i + block_size] for i in range(0, len(param_samples), block_size)]

    for i, block in enumerate(param_blocks):
        base_sample = block[0]
        copy_of_storage = copy.deepcopy(storage)

        # Run only the base sample first
        base_result, IC_final, storage_final = simulate_cpu(base_sample, copy_of_storage, Old_Parameters)

        # If base sample fails (e.g. returns 0 or some error code), skip the whole block
        if base_result[0] == 0:  # Adjust this condition to your failure criteria
            print(f"Skipping block {i + 1} due to base failure.")
            results_all.extend(np.zeros((block_size, 3)))
            np.save(save_path, np.array(results_all))
            continue

        # perturbations = block[1:]  # exclude the base sample

        # Otherwise, run full block in parallel
        with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim Block {i}", total=len(block), disable=True)):
            results_perturbations = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, copy.deepcopy(storage_final), Old_Parameters, IC_initial=IC_final) for params in block)

        results_block = [res[0] for res in results_perturbations]
        results_all.extend(results_block)

        # Save chunk incrementally (appending)
        # np.save(f'IC_final_{i:03d}.npy', IC_final)  # individual chunks
        # np.save(f'Next_final_{i:03d}.npy', storage_final)  # individual chunks

        # Save after each block
        np.save(save_path, np.array(results_all))

    return results_all


# def parallel_simulations(param_samples, storage, save_path='Result_DGSM_new.npy'):
#     results_all = []
#
#     if os.path.exists(save_path):
#         os.remove(save_path)
#
#     block_size = 170
#     param_blocks = [param_samples[i:i + block_size] for i in range(0, len(param_samples), block_size)]
#
#     for i, block in enumerate(param_blocks):
#         base_sample = block[0]
#         copy_of_storage = copy.deepcopy(storage)
#         print(f"Running base sample for block {i+1}...")
#
#         base_result, IC_final, storage_final = simulate_cpu(base_sample, copy_of_storage, Old_Parameters)
#
#         print(f"Base sample result: {base_result}")
#
#         if base_result[0] == 0:
#             print(f"Skipping block {i + 1} due to base failure.")
#             results_all.extend(np.zeros((174, 3)))
#             np.save(save_path, np.array(results_all))
#             continue
#
#         results_perturbations = []
#         for j, params in enumerate(block):
#             print(f"Running perturbation {j+1}/{len(block)} of block {i+1}...")
#             res = simulate_cpu(params, copy.deepcopy(storage_final), Old_Parameters, IC_initial=IC_final)
#             print(f"Perturbation result: {res[0]}")
#             results_perturbations.append(res)
#
#         results_block = [base_result] + results_perturbations
#         results_all.extend(results_block)
#
#         # Save checkpoint files for debugging
#         np.save(f'IC_final_{i:03d}.npy', IC_final)
#         np.save(f'Next_final_{i:03d}.npy', storage_final)
#
#         np.save(save_path, np.array(results_all))
#         print(f"Block {i+1} finished and results saved.")
#
#     return results_all


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
            "C_sa", "L_sa", "R_sa", "C_amv", "C_bv",
            "C_ev", "C_hv", "C_rmv", "C_sv", "R_amv_n", "R_bv_n",
            "R_ev_n", "R_hv_n", "R_rmv_n", "R_sv_n", "D1", "K1_vc", "Kr_vc", "Rvc_n",
            "C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "Emax_la", "P0_la", "Emax_ra",
            "P0_ra", "P0_lv", "P0_rv", "g_abd", "g_thor", "P_abdmax_n", "P_abdmin_n",
            # "P_thormax_n", "P_thormin_n",
            "VT_n", "A_im", "Tc", "T_im", "s",
            # cardio control
            "fab_o", "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf",
            "kes", "kev", "kcc_sh", "kcc_sp", "kcc_sv", "kcc_v", "Ysh_max", "Ysh_min", "Ysp_max", "Ysp_min",
            "Ysv_max", "Ysv_min", "Yv_max", "Yv_min", "theta_v", "Wb_sh", "Wb_sp", "Wb_sv", "Wc_sh", "Wc_sp",
            "Wc_sv", "Wc_v", "Wp_sh", "Wp_sp", "Wp_sv", "Wp_v", "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
            "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv",
            "GV_ev", "GV_rmv", "GV_sv", "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "AT", "g_ccsh", "g_ccsp",
            "g_ccsv", "kisc_sh", "kisc_sp", "kisc_sv", "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn", "theta_spn",
            "theta_svn", "x_sh", "x_sp", "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min", "k_ab", "P_n", "P_n_max", "f_acCO2_n", "f_ac_max",
            "f_ac_min", "k_ac", "K_H", "PaO2_ac_n", "G_ap", "GT_s", "GT_v", "T0", "A", "B",
            "C", "D", "Cvb_O2_n", "gb_O2", "MO2_bp", "R_bpn", "Cvh_O2_n", "Cvrm_O2_n", "gh_O2", "grm_O2",
            "Kh_CO2", "Krm_CO2", "MO2_hpn", "MO2_rmp", "R_hpn", "W_hn", "Cvam_O2_n", "gam_O2", "gM", "kmet",
            "MO2_ampn", "phi_max", "phi_min",
            # exercise added parameters
            "Io_sh", "Io_sp", "Io_sv", "Io_v"
        ],

        'bounds': [
            # gas
            [0.03255 * lower, 0.03255 * upper], [40 * lower, 40 * upper],
            [25 * lower, 25 * upper], [1.219 * lower, 1.219 * upper],
            [0.05591 * lower, 0.05591 * upper], [0.015 * lower, 0.015 * upper],
            [346000 * lower, 346000 * upper],
            # [0.0009 * lower, 0.0009 * upper],
            # resp control
            [0.1698 * lower, 0.1698 * upper],
            # [17.4 * lower, 17.4 * upper],
            [0.2332 * lower, 0.2332 * upper],
            [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
            [0.1587 * lower, 0.1587 * upper], [0.067 * lower, 0.067 * upper], [50 * lower, 50 * upper],
            [1000 * lower, 1000 * upper], [21.9 * lower, 21.9 * upper], [3.02 * lower, 3.02 * upper],
            # cardio
            [0.28 * lower, 0.28 * upper], [0.00066 * lower, 0.00066 * upper], [0.2 * lower, 0.2 * upper],
            [9.4 * lower, 9.4 * upper],
            [10.71 * lower, 10.71 * upper], [20 * lower, 20 * upper],
            [3.57 * lower, 3.57 * upper],
            [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper],
            [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper],
            [0.224 * lower, 0.224 * upper], [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper],
            [0.3855 * lower, 0.3855 * upper], [0.15 * lower, 0.15 * upper],
            [0.001 * lower, 0.001 * upper], [0.075 * lower, 0.075 * upper],
            [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper], [20.5 * lower, 20.5 * upper],
            [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.3 * lower, 0.3 * upper],
            [0.06 * lower, 0.06 * upper], [0.25 * lower, 0.25 * upper], [0.55 * lower, 0.55 * upper],
            [0.25 * lower, 0.25 * upper], [0.55 * lower, 0.55 * upper], [1.5 * lower, 1.5 * upper],
            [1.5 * lower, 1.5 * upper], [3.39 * lower, 3.39 * upper], [6.8 * lower, 6.8 * upper],
            [-1 * upper, -1 * lower], [-2.5 * upper, -2.5 * lower],
            # [-1 * upper, -1 * lower],
            # [-2 * upper, -2 * lower],
            [0.45 * lower, 0.45 * upper], [30 * lower, 30 * upper],
            [0.7 * lower, 0.7 * upper], [1.1 * lower, 1.1 * upper], [0.04 * lower, 0.04 * upper],
            # cardio control
            [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper],
            [80 * lower, 80 * upper], [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper],
            [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper], [0.114 * lower, 0.114 * upper],
            [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
            [20 * lower, 20 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper],
            [-0.037 * upper, -0.037 * lower], [64.9 * lower, 64.9 * upper], [-0.028 * upper, -0.028 * lower],
            [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower], [-0.68 * upper, -0.68 * lower],
            [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
            [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper],
            [0.2 * lower, 0.2 * upper], [-0.2 * upper, -0.2 * lower], [-0.3997 * upper, -0.3997 * lower],
            [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower], [0.4 * lower, 0.4 * upper],
            [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
            [1.412 * lower, 1.412 * upper], [0.7 * lower, 0.7 * upper], [2.66 * lower, 2.66 * upper],
            [0.475 * lower, 0.475 * upper], [0.282 * lower, 0.282 * upper], [2.47 * lower, 2.47 * upper],
            [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper], [0.695 * lower, 0.695 * upper],
            [-58.29 * upper, -58.29 * lower], [-74.21 * upper, -74.21 * lower], [-58.29 * upper, -58.29 * lower],
            [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [5.655 * lower, 5.655 * upper],
            [10.27 * lower, 10.27 * upper], [5.49 * lower, 5.49 * upper], [(1 / 60) * lower, (1 / 60) * upper],
            [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper], [0.2 * lower, 0.2 * upper],
            [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
            [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper],
            [3.6 * lower, 3.6 * upper], [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper],
            [53 * lower, 53 * upper], [6 * lower, 6 * upper], [6 * lower, 6 * upper],
            [40 * lower, 40 * upper], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
            [11.76 * lower, 11.76 * upper], [92 * lower, 92 * upper], [122 * lower, 122 * upper], [1.4 * lower, 1.4 * upper],
            [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper],
            [3 * lower, 3 * upper], [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper],
            [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper], [0.58 * lower, 0.58 * upper],
            [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
            [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper],
            [0.925 * lower, 0.925 * upper], [10.57 * lower, 10.57 * upper], [0.11 * lower, 0.11 * upper],
            [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper], [30 * lower, 30 * upper],
            [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
            [0.86 * lower, 0.86 * upper], [25.71 * lower, 25.71 * upper], [12660 * lower, 12660 * upper],
            [0.1555 * lower, 0.1555 * upper], [30 * lower, 30 * upper], [40 * lower, 40 * upper],
            [0.18 * lower, 0.18 * upper], [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper],
            [-1.87 * upper, -1.87 * lower],
            # exercise added parameters
            [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper], [0.22 * lower, 0.22 * upper]
        ],
    })

    param_keys = list(sp["names"])

    # DGSM uses finite differences sampling since it is a derivative based method
    # shape: (B * (P + 1), P) where B is the number of base points chosen in each parameter range P
    X = finite_diff.sample(sp, 500)
    # X = X[0::184, :]
    #
    # X_3 = X[41375:,:]
    # X_1 = X[:41374, :]
    # X_2 = np.array([X[41375,:]])
    # X = np.concatenate((X_1, X_2, X_3))

    # np.save('New_DGSM_250_X_samples_HR_P_sys_P_dia_no_bifur_delay.npy', X)
    #
    # X_fail = X_load[41374,:]
    # np.save('Fail_250_X_sample_41374_HR_P_sys_P_dia_exercise.npy', X_fail)

    # X = np.load('DGSM_500_X_samples_HR_P_sys_P_dia_filtered.npy')

    param_samples = [dict(zip(param_keys, row)) for row in X]
    # param_samples = [Old_Parameters]
    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=-1)
    # Result = parallel_simulations(param_samples, Next_Conditions)

    # print(Result)

    np.save('New_DGSM_500_Result_HR_P_sys_P_dia_no_bifur_delay_exericise.npy', Result)

