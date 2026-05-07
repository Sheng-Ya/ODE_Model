import pickle
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
from Resp_Control_Breath_Optimiser import objective
from scipy.signal import find_peaks, savgol_filter
# from line_profiler import LineProfiler
from All_derivatives import model_derivatives
from Entire_system.fixed_params import Parameters
from check import Parameters as new_params

from Initial_Conditions_after_running_again import Initial_Conditions
from Next_Conditions_all_derivatives import Next_Conditions


target_values = np.arange(0, 10000, 10)

time_saved = 0.005
BUFFER_LIMIT = 40000
CACHE_PATH = Path("Run_model_Paper_simulation_cache.pkl")
CACHE_VERSION = 1

RESULT_OUTPUT_NAMES_FULL = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
    "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure",
    "Min_RA_Volume", "Max_RA_Volume", "Min_RA_Pressure_A_descent",
    "Max_RA_Pressure_Atrial_contraction", "Max_RA_Pressure_Tricuspid_Opening",
    "Min_RA_Pressure_V_descent", "Min_LA_Volume", "Max_LA_Volume",
    "Min_LA_Pressure_A_descent", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "Min_LA_Pressure_V_descent",
    "LA_Volume_Before_Atrial_Contraction", "RA_Volume_Before_Atrial_Contraction",
    "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume",
    "Minute_Ventilation", "Cardiac_Output", "PaO2", "PaCO2",
    "Pericardial_Volume_Percentage_Change",
]
RESULT_COLS_TO_DROP = [11, 14, 17, 20, 27, 30]
STAGE_COLORS = [
    "#BBA3D6",
    "#9DB8D8",
    "#7DB6C0",
    "#D68484",
]
PLOT_COLORS = {
    "solid_red": "#D68484",
    "solid_blue": "#7DB6C0",
    "lavender": STAGE_COLORS[0],
    "blue": STAGE_COLORS[1],
    "teal": STAGE_COLORS[2],
    "rose": STAGE_COLORS[3],
    "lavender_dark": "#8F78AB",
    "blue_dark": "#789CC4",
    "teal_dark": "#5D9FA8",
    "rose_dark": "#B86A6A",
    "lavender_light": "#C9B8DE",
    "blue_light": "#B4CAE2",
    "teal_light": "#9ACBD1",
    "rose_light": "#E2A4A4",
    "ink": "#4A4E57",
}
SOLID_LINEWIDTH = 1.8
TARGET_LINEWIDTH = 1.4
ATRIAL_TARGET_TIME_START = 56.74
DPDT_TIME_START = 56.75
HEART_RATE_PLOT_SCALE = 60.0
SUBPLOT_LEGEND_FONT_SIZE = 11
JOURNAL_RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#555555",
    "axes.labelcolor": "#303030",
    "axes.linewidth": 1.1,
    "axes.labelsize": 13,
    "xtick.color": "#303030",
    "ytick.color": "#303030",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": SUBPLOT_LEGEND_FONT_SIZE,
    "font.size": 12,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
TARGET_LABELS = {
    "Heart_Rate": "HR",
    "Systolic_Pressure": r"$P_{\mathrm{sys,LV}}$",
    "Diastolic_Pressure": r"$P_{\mathrm{dia,LV}}$",
    "EDV": r"$V_{\mathrm{ED,LV}}$",
    "ESV": r"$V_{\mathrm{ES,LV}}$",
    "Max_RV_Volume": r"$V_{\mathrm{ED,RV}}$",
    "Min_RV_Volume": r"$V_{\mathrm{ES,RV}}$",
    "Max_RV_Pressure": r"$P_{\mathrm{sys,RV}}$",
    "Min_RV_Pressure": r"$P_{\mathrm{dia,RV}}$",
    "Min_RA_Volume": r"$V_{\min,\mathrm{RA}}$",
    "Max_RA_Volume": r"$V_{\max,\mathrm{RA}}$",
    "Max_RA_Pressure_Atrial_contraction": r"$P_{\max,A,\mathrm{RA}}$",
    "Max_RA_Pressure_Tricuspid_Opening": r"$P_{\max,V,\mathrm{RA}}$",
    "Min_LA_Volume": r"$V_{\min,\mathrm{LA}}$",
    "Max_LA_Volume": r"$V_{\max,\mathrm{LA}}$",
    "Max_LA_Pressure_Atrial_contraction": r"$P_{\max,A,\mathrm{LA}}$",
    "Max_LA_Pressure_Mitral_Opening": r"$P_{\max,V,\mathrm{LA}}$",
    "LA_Volume_Before_Atrial_Contraction": r"$V_{\mathrm{pre-A,LA}}$",
    "RA_Volume_Before_Atrial_Contraction": r"$V_{\mathrm{pre-A,RA}}$",
    "LV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{LV}}/\mathrm{d}t$",
    "RV_Pressure_Deriv": r"$\max\,\mathrm{d}P_{\mathrm{RV}}/\mathrm{d}t$",
    "Tidal_Volume": "Inspired/Expired Volume",
    "Minute_Ventilation": r"$\dot{V}_E$",
    "PaO2": r"$P_{\mathrm{a}O_2}$",
    "PaCO2": r"$P_{\mathrm{a}CO_2}$",
}
TARGET_MEAN_LABELS = {
    "Heart_Rate": r"$\overline{\mathrm{HR}}$",
    "PaO2": r"$\overline{P_{\mathrm{a}O_2}}$",
    "PaCO2": r"$\overline{P_{\mathrm{a}CO_2}}$",
    "Tidal_Volume": r"$V_T$",
}

min_time = 10 # Minimum time in seconds before checking
max_time = 60 # Maximum time limit to avoid infinite loops
time_step = 200  # Chunk size per solve

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1: # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            # # num_removed = 6
            # index = -1  # Set a default value for safety
            #
            # # Iterating through the buffer in circular order
            # for j in range(BUFFER_LIMIT):
            #     logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
            #     if all_time[logical_index] < t:
            #         index = (logical_index + 1) % BUFFER_LIMIT
            #         break
            #
            # num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (actual_index - index)
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

    # Debugging check for progress
    if t != 0:
        diff = np.abs(t - target_values)
        if np.any(diff < 0.0001):
            print(t)

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
required_resp_control_keys = ["VE_integral"] #, "v_r", "x_r"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))
# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))
# IC_overall = IC_cardio


# def minimise_breathing(t1, t2, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao):
#     dt = 0.001 # must edit in Resp_Control_Breath_Optimiser too
#     bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
#     tolerance = 0.0001
#
#     VAflow_vals = np.linspace(0.01, 1.2, 200)
#     # VAflow_vals = np.repeat(VAflow_vals, 3)
#
#     VD = GV_dead * VAflow_vals + V0_dead
#
#     optimal_t1 = np.empty_like(VAflow_vals)
#     optimal_t2 = np.empty_like(VAflow_vals)
#     initial_guess = np.array([t1, t2], dtype=float)
#     required_params = [lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao]
#
#     for i, (VAflow, VD_volume) in enumerate(zip(VAflow_vals, VD)):
#         res = minimize(objective, x0= initial_guess,
#                        args=(required_params, VAflow, VD_volume, dt, tolerance), method='nelder-mead', bounds=bounds)
#
#         initial_guess = res.x
#         optimal_t1[i] = initial_guess[0]
#         optimal_t2[i] = initial_guess[1]
#
#     cs_t1 = CubicSpline(VAflow_vals, optimal_t1, bc_type="natural")
#     cs_t2 = CubicSpline(VAflow_vals, optimal_t2, bc_type="natural")
#
#     # t1_spline = cs_t1(VAflow_vals)
#     # t2_spline = cs_t2(VAflow_vals)
#     # plt.figure(figsize=(10, 5))
#     # # Spline fits
#     # plt.plot(VAflow_vals, t1_spline, label='t1 CubicSpline', color='blue', linewidth=2)
#     # plt.plot(VAflow_vals, t2_spline, label='t2 CubicSpline', color='red', linewidth=2)
#     # plt.scatter(VAflow_vals, t1_mean, color='blue', s=10, marker='o', label='t1 mean (knots)')
#     # plt.scatter(VAflow_vals, t2_mean, color='red', s=10, marker='o', label='t2 mean (knots)')
#     # plt.scatter(VAflow_clean, t1_clean, label='Optimal t1 (Inspiration Time)', color='blue', alpha=0.6, s=5)
#     # plt.scatter(VAflow_clean, t2_clean, label='Optimal t2 (Expiration Time)', color='red', alpha=0.6, s=5)
#     # plt.xlabel('VAflow (L/s)')
#     # plt.ylabel('Time (s)')
#     # plt.title('Optimal t1 and t2 vs VAflow Using nelder-mead')
#     # plt.legend()
#     # plt.grid(True)
#     # plt.show()
#
#     return cs_t1.c, cs_t2.c, cs_t1.x, cs_t2.x


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

    return cs_t1.c, cs_t2.c, cs_t1.x, cs_t2.x


def simulate():
    # Initial setup
    IC_current = IC_overall.copy()

    (A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa, C_pp, C_pv, L_pa,
    R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
    Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n,
    C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv,
    kr_am, P_0) = (
    new_params[k] if k in new_params else Parameters[k] for k in
    ["A_im", "T_im", "Tc", "g_thor", "P_thormax_n", "P_thormin_n", "VT_n", "C_pa",
     "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv",
     "Emax_la", "P0_la", "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa",
     "R_sa", "K1_vc", "D1", "Vvc_min", "Kr_vc", "Rvc_n", "C_jp",
     "R_ev_n", "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv",
     "kr_am", "P_0"])

    # Cardio controller parameters
    (fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv, Io_v, kcc_sh, kcc_sp, kcc_sv,
    kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v, Wb_sh, Wb_sp, Wb_sv, Wc_sh,
    Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0, Emax_rv0, fes_min, GEmax_lv,
    GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv,
    theta_shn, theta_spn, theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n, P_n_max,
    f_acCO2_n, f_ac_max, f_ac_min, k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D,
    Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2, grm_O2, Kh_CO2, Krm_CO2, MO2_hpn,
    MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max, phi_min) = \
    [new_params[k] if k in new_params else Parameters[k] for k in
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
     MRCO2, MRO2, s) = (new_params[k] if k in new_params else Parameters[k] for k in [
    "a2", "alpha2", "beta2", "C2", "K2", "PACO2_Delay_IC",
    "PAO2_Delay_IC", "P_atm", "P_ws", "Z", "dc", "KCCO2", "MRBCO2",
    "MO2_bp", "MRTCO2_basal", "MRTO2_basal", "MRCO2", "MRO2", "s"])

    # Resp control
    (GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao) = \
    (new_params[k] if k in new_params else Parameters[k] for k in ["GV_dead", "KcCO2", "KcMRV", "KpCO2", "KpO2",
   "V0_dead", "VA_rest", "lambda1", "lambda2", "n", "Pmax", "Pmax_dot", "E_rs", "R_rs", "P_ao"])

    # added params
    (Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
    Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
    R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
    scale_param1, scale_param3, scale_param4, scale_param6,
    Pa_O2_lower, rise_time_atr, rise_time_ven,
    fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale
     ) = \
    (new_params[k] if k in new_params else Parameters[k] for k in ["Kp_ao", "Kf_ao", "Kb_ao",
    "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po", "Kf_po", "Kb_po", "Kv_po",
    "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "alpha_O2", "R_po", "R_mi", "R_tr", "R_ao",
    "C_O2_param1", "C_O2_param2", "C_O2_param3", "PAMO2_nominal", "Vu_sa", "V_tot", "Vu_jp",
    "Vu_bv", "Vu_hv", "Vu_vc", "Vu_pa", "Vu_pp", "Vu_pv",
    "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp",
    "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_cc", "tau_isc",
    "tau_p", "tau_z", "tau_ac", "tau_ap", "tau_Ts", "tau_Tv", "tau_CO2", "tau_O2", "tau_w", "tau_M", "tau_met",
    "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv", "DT_s", "DT_v",
    "Dmet", "Fi_CO2", "Fi_O2", "Ta", "T1", "T2", "VL_CO2", "VL_O2", "KCSFCO2", "VB", "tauMR", "VTCO2", "VTO2", "tau_MRV",
    "scale_param1", "scale_param3", "scale_param4", "scale_param6",
    "Pa_O2_lower", "rise_time_atr", "rise_time_ven",
     "fall_time_ven", "ahead1", "theta_min", "delta_P", "r", "l", "V_nominal", "V_scale"])

    # # determine the correct breathing profile
    cs_t1, cs_t2, knots_1, knots_2 = (minimise_breathing(1.5,
    1.85, GV_dead, V0_dead, lambda1, lambda2, n, Pmax, Pmax_dot, E_rs, R_rs, P_ao))

    # data = {
    #     "cs_t1": cs_t1,
    #     "cs_t2": cs_t2,
    #     "knots_1": knots_1,
    #     "knots_2": knots_2,
    # }
    #
    # with open("breathing_splines.pkl", "wb") as f:
    #     pickle.dump(data, f)

    # with open("breathing_splines.pkl", "rb") as f:
    #     data = pickle.load(f)
    #
    # cs_t1 = data["cs_t1"]
    # cs_t2 = data["cs_t2"]
    # knots_1 = data["knots_1"]
    # knots_2 = data["knots_2"]

    Input_Parameters = np.array([A_im, T_im, Tc, g_thor, P_thormax_n, P_thormin_n, VT_n, C_pa,
    C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra, C_sa,
    L_sa, R_sa, K1_vc, D1, Vvc_min, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev, C_sv,
    C_bv, C_hv, C_rmv, C_amv, kr_am, P_0, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Io_sh, Io_sp, Io_sv,
    Io_v, kcc_sh, kcc_sp, kcc_sv, kcc_v, Ysh_max, Ysh_min, Ysp_max, Ysp_min, Ysv_max, Ysv_min, Yv_max, Yv_min, theta_v,
    Wb_sh, Wb_sp, Wb_sv, Wc_sh, Wc_sp, Wc_sv, Wc_v, Wp_sh, Wp_sp, Wp_sv, Wp_v, Wt_sh, Wt_sp, Wt_sv, Wt_v, Emax_lv0,
    Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev, GV_rmv, GV_sv, R_amp0, R_ep0,
    R_rmp0, R_sp0, AT, g_ccsh, g_ccsp, g_ccsv, kisc_sh, kisc_sp, kisc_sv, PO2_sh, PO2_sp, PO2_sv, theta_shn, theta_spn,
    theta_svn, x_sh, x_sp, x_sv, PaCO2_n, f_ab_max, f_ab_min, k_ab, P_n,  P_n_max, f_acCO2_n, f_ac_max, f_ac_min,
    k_ac, K_H, PaO2_ac_n, G_ap, DT_v, GT_s, GT_v, T0, A, B, C, D, Cvb_O2_n, gb_O2, R_bpn, Cvh_O2_n, Cvrm_O2_n, gh_O2,
    grm_O2, Kh_CO2, Krm_CO2, MO2_hpn, MO2_rmp, R_hpn, W_hn, Cvam_O2_n, gam_O2, gM, Io_met, kmet, MO2_ampn, phi_max,
    phi_min, a2_gas, alpha2, beta2, C2, K2, PACO2_Delay_IC, PAO2_Delay_IC, P_atm, P_ws, Z, dc, KCCO2, MRBCO2, MO2_bp,
    MRTCO2_basal, MRTO2_basal, MRCO2, MRO2, s, GV_dead, KcCO2, KcMRV, KpCO2, KpO2, V0_dead, VA_rest, lambda1, lambda2,
    n, Pmax, Pmax_dot, E_rs, R_rs, P_ao,
    # added params
    Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po,
    Kf_po, Kb_po, Kv_po, theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, alpha_O2, R_po, R_mi, R_tr,
    R_ao, C_O2_param1, C_O2_param2, C_O2_param3, PAMO2_nominal,
    Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv, Vu_vc, Vu_pa, Vu_pp,
    Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep, tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev,
    tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_cc, tau_isc, tau_p, tau_z, tau_ac, tau_ap, tau_Ts, tau_Tv,
    tau_CO2, tau_O2, tau_w, tau_M, tau_met, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv,
    DV_sv, DT_s, DT_v, Dmet, Fi_CO2, Fi_O2, Ta, T1, T2, VL_CO2, VL_O2, KCSFCO2, VB, tauMR, VTCO2, VTO2, tau_MRV,
    scale_param1, scale_param3, scale_param4, scale_param6,
     Pa_O2_lower, rise_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P, r, l, V_nominal, V_scale])

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        (0, max_time),
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(Next_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control, Input_Parameters, cs_t1, cs_t2, knots_1, knots_2)
    )

    if ODE_solution.status == -1:
        print("ODE solver failed:", ODE_solution.message)
        return ODE_solution

    # Post-processing: use buffer to get recent data
    i_buffer = Next_Conditions["i"].item() % BUFFER_LIMIT

    all_time = np.concatenate((Next_Conditions["all_time"][i_buffer:], Next_Conditions["all_time"][:i_buffer]))
    time_since_beat_store = np.concatenate((Next_Conditions["time_since_beat_store"][i_buffer:], Next_Conditions["time_since_beat_store"][:i_buffer]))
    finish_breath_time = np.concatenate((Next_Conditions["finish_breath_time"][i_buffer:], Next_Conditions["finish_breath_time"][:i_buffer]))

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
    print(combined)
    np.save("combined.npy", combined)

    theta_ao = np.concatenate((Next_Conditions["theta_ao_store"][i_buffer:], Next_Conditions["theta_ao_store"][:i_buffer]))
    theta_po = np.concatenate((Next_Conditions["theta_po_store"][i_buffer:], Next_Conditions["theta_po_store"][:i_buffer]))
    theta_mi = np.concatenate((Next_Conditions["theta_mi_store"][i_buffer:], Next_Conditions["theta_mi_store"][:i_buffer]))
    theta_tr = np.concatenate((Next_Conditions["theta_tr_store"][i_buffer:], Next_Conditions["theta_tr_store"][:i_buffer]))

    V_rv = np.concatenate((Next_Conditions["V_rv_store"][i_buffer:], Next_Conditions["V_rv_store"][:i_buffer]))
    V_ra = np.concatenate((Next_Conditions["V_ra_store"][i_buffer:], Next_Conditions["V_ra_store"][:i_buffer]))
    V_la = np.concatenate((Next_Conditions["V_la_store"][i_buffer:], Next_Conditions["V_la_store"][:i_buffer]))

    N = 50  # number of consecutive closed samples required

    is_open_ao = theta_ao > theta_min
    open_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_open_ao[k] and not np.any(is_open_ao[k - N:k]):
            open_idx1.append(k)
    open_idx1 = np.array(open_idx1)

    is_closed_ao = theta_ao <= theta_min
    close_idx1 = []
    for k in range(N, len(theta_ao)):
        if is_closed_ao[k] and not np.any(is_closed_ao[k - N:k]):
            close_idx1.append(k)
    close_idx1 = np.array(close_idx1)

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

    pairs_ao = np.array([
        (o, close_idx1[(close_idx1 > o) & (close_idx1 < o_next)][-1])
        for o, o_next in zip(open_idx1[:-1], open_idx1[1:])
        if np.any((close_idx1 > o) & (close_idx1 < o_next))])

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

    pairs_ao = pairs_ao[-11:-1]
    pairs_po = pairs_po[-11:-1]
    pairs_mi = pairs_mi[-11:-1]
    pairs_tr = pairs_tr[-11:-1]

    # Max pressure during atrial contraction takes the max p between phi_atr = 0 & 1
    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i_buffer:], Next_Conditions["phi_atr_store"][:i_buffer]))

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

    # systolic pressure
    P_sa = np.concatenate((Next_Conditions["P_sa_store"][i_buffer:], Next_Conditions["P_sa_store"][:i_buffer]))
    P_sa_max_idx = np.array([o + np.argmax(P_sa[o:c]) for o, c in pairs_ao])

    P_la = np.concatenate((Next_Conditions["P_la_store"][i_buffer:], Next_Conditions["P_la_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_la_max_idx = np.array([s + np.argmax(P_la[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    # period of V descent when mitral valve is open -> get second min la P
    P_la_descent2_idx = np.array([o + np.argmin(P_la[o:c]) for o, c in pairs_mi])
    P_la_descent1_idx = np.array([c + np.argmin(P_la[c:o_next]) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])])

    P_ra = np.concatenate((Next_Conditions["P_ra_store"][i_buffer:], Next_Conditions["P_ra_store"][:i_buffer]))
    # max pressure at atrial contraction
    P_ra_max_idx = np.array([s + np.argmax(P_ra[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    # period of V descent when tricuspid valve is open -> get second min la P
    P_ra_descent2_idx = np.array([o + np.argmin(P_ra[o:c]) for o, c in pairs_tr])
    P_ra_descent1_idx = np.array([c + np.argmin(P_ra[c:o_next]) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])])

    V_lv = np.concatenate((Next_Conditions["V_lv_store"][i_buffer:], Next_Conditions["V_lv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)

    last_10_troughs_V_lv = troughs[-11:-1]
    last_10_min_V_lv = V_lv[last_10_troughs_V_lv]

    last_10_peaks_V_lv = peaks[-11:-1]
    last_10_max_V_lv = V_lv[last_10_peaks_V_lv]

    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i_buffer:], Next_Conditions["P_rv_store"][:i_buffer]))
    P_rv_max_idx = np.array([o + np.argmax(P_rv[o:c]) for o, c in pairs_po])
    P_rv_min_idx = np.array([c + np.argmin(P_rv[c:o_next]) for (_, c), (o_next, _) in zip(pairs_po[:-1], pairs_po[1:])])

    HR = np.concatenate((Next_Conditions["HR_store"][i_buffer:], Next_Conditions["HR_store"][:i_buffer]))

    past_10_flat_segments = []
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break

    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_mins = starts[-11:-1]
    last_10_b4_LA_atrial_contract = V_la[local_mins]
    last_10_b4_RA_atrial_contract = V_ra[local_mins]

    # maximum ventricular pressure derivative
    is_active = phi_atr > 0.0  # atrial contraction window
    edges = np.diff(is_active.astype(int))

    start_idx = np.where(edges == 1)[0] + 1  # 0 â†’ active
    end_idx = np.where(edges == -1)[0] + 1  # active â†’ 0

    if len(start_idx) and len(end_idx) and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]

    n_pairs = min(len(start_idx), len(end_idx))
    start_idx = start_idx[:n_pairs]
    end_idx = end_idx[:n_pairs]

    dP_lv_dt_store = np.concatenate((Next_Conditions["dP_lv_dt_store"][i_buffer:], Next_Conditions["dP_lv_dt_store"][:i_buffer]))
    dP_lv_dt_idx = np.array([s + np.argmax(dP_lv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    dP_rv_dt_store = np.concatenate((Next_Conditions["dP_rv_dt_store"][i_buffer:], Next_Conditions["dP_rv_dt_store"][:i_buffer]))
    dP_rv_dt_idx = np.array([s + np.argmax(dP_rv_dt_store[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    tidal = np.concatenate((Next_Conditions["tidal_store"][i_buffer:], Next_Conditions["tidal_store"][:i_buffer]))

    breath_starts = np.where(dtr > 0)[0] + 1
    if breath_starts.size >= 2:
        max_tidal = np.max(tidal[breath_starts[-2]:breath_starts[-1]])
    else:
        max_tidal = np.max(tidal[tidal > 0]) if np.any(tidal > 0) else 0.0

    VAflow = np.concatenate((Next_Conditions["VAflow_store"][i_buffer:], Next_Conditions["VAflow_store"][:i_buffer]))
    t1 = np.concatenate((Next_Conditions["t1_store"][i_buffer:], Next_Conditions["t1_store"][:i_buffer]))
    t2 = np.concatenate((Next_Conditions["t2_store"][i_buffer:], Next_Conditions["t2_store"][:i_buffer]))
    VD = GV_dead * VAflow[-1] + V0_dead
    VDflow = (1 / (t1[-1] + t2[-1])) * VD
    Minute_Ventilation = (VAflow[-1] + VDflow) * 60

    cardiac_output = np.mean(Next_Conditions["Q_pp_store"])
    Pa_O2 = np.mean(Next_Conditions["Pa_O2_every_store"])
    Pa_CO2 = np.mean(Next_Conditions["Pa_CO2_every_store"])

    Total_Volume = V_ra + V_rv + V_lv + V_la

    Total_Vol_min_idx = np.array([s + np.argmin(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]
    Total_Vol_max_idx = np.array([s + np.argmax(Total_Volume[s:e]) for s, e in zip(start_idx, end_idx)])[-11:-1]

    mean_min_Total_Volume = np.mean(Total_Volume[Total_Vol_min_idx])
    mean_max_Total_Volume = np.mean(Total_Volume[Total_Vol_max_idx])
    Pericardial_Volume_difference = mean_max_Total_Volume - mean_min_Total_Volume
    Vol_percentage_change = Pericardial_Volume_difference / mean_max_Total_Volume

    # LA_Contraction_Volume_diff = np.mean(last_10_b4_LA_atrial_contract) - np.mean(V_la[pairs_mi[:, 1]])
    # RA_Contraction_Volume_diff = np.mean(last_10_b4_RA_atrial_contract) - np.mean(V_ra[pairs_tr[:, 1]])

    # np.savez(f'HR_vs_time.npz', HR=Next_Conditions["HR_check"], time=Next_Conditions["time_history"], HR_average = Next_Conditions["HR"])
    print(np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
          np.mean(V_lv[pairs_ao[:, 0]]), np.mean(V_lv[pairs_ao[:, 1]]), np.mean(V_rv[pairs_po[:, 0]]), np.mean(V_rv[pairs_po[:, 1]]),
          np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_min_idx]),
          np.mean(V_ra[pairs_tr[:, 1]]), np.mean(V_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent1_idx]),
          np.mean(P_ra[P_ra_max_idx]), np.mean(P_ra[pairs_tr[:, 0]]), np.mean(P_ra[P_ra_descent2_idx]),
          np.mean(V_la[pairs_mi[:, 1]]), np.mean(V_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent1_idx]),
          np.mean(P_la[P_la_max_idx]), np.mean(P_la[pairs_mi[:, 0]]), np.mean(P_la[P_la_descent2_idx]),
          np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
          np.mean(dP_lv_dt_store[dP_lv_dt_idx]), np.mean(dP_rv_dt_store[dP_rv_dt_idx]), max_tidal,
          Minute_Ventilation, cardiac_output, Pa_O2, Pa_CO2, Vol_percentage_change, sep=", ")


    return (ODE_solution, np.mean(past_10_flat_segments), np.mean(P_sa[P_sa_max_idx]), np.mean(P_sa[open_idx1]),
            np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(V_rv[pairs_po[:, 0]]),
            np.mean(V_rv[pairs_po[:, 1]]),
            np.mean(P_rv[P_rv_max_idx]), np.mean(P_rv[P_rv_min_idx]),
            IC_current, Next_Conditions, ODE_solution.t, ODE_solution.y)


def _parameter_value(name):
    return new_params[name] if name in new_params else Parameters[name]


def _sorted_buffer(values, start_idx):
    return np.concatenate((values[start_idx:], values[:start_idx]))


def _tail_complete(values):
    values = np.asarray(values)
    return values[-11:-1] if values.shape[0] > 10 else values


def _last_complete_pairs(pairs):
    pairs = np.asarray(pairs, dtype=int)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    pairs = pairs.reshape((-1, 2))
    return pairs[-11:-1] if pairs.shape[0] > 10 else pairs


def _mean_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanmean(values)) if values.size else np.nan


def _mean_at(values, indices):
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return np.nan
    return _mean_values(np.asarray(values)[indices])


def _window_indices(values, windows, reducer):
    values = np.asarray(values)
    indices = []
    for start, end in np.asarray(windows, dtype=int).reshape((-1, 2)):
        if end > start and start >= 0 and end <= values.size:
            indices.append(start + int(reducer(values[start:end])))
    return np.asarray(indices, dtype=int)


def _valve_events(theta, theta_min, n_closed=50):
    is_open = theta > theta_min
    open_idx = []
    for k in range(n_closed, len(theta)):
        if is_open[k] and not np.any(is_open[k - n_closed:k]):
            open_idx.append(k)
    open_idx = np.asarray(open_idx, dtype=int)

    is_closed = theta <= theta_min
    close_idx = []
    for k in range(n_closed, len(theta)):
        if is_closed[k] and not np.any(is_closed[k - n_closed:k]):
            close_idx.append(k)
    close_idx = np.asarray(close_idx, dtype=int)

    pairs = [
        (open_now, close_idx[(close_idx > open_now) & (close_idx < open_next)][-1])
        for open_now, open_next in zip(open_idx[:-1], open_idx[1:])
        if np.any((close_idx > open_now) & (close_idx < open_next))
    ]
    return open_idx, close_idx, np.asarray(pairs, dtype=int).reshape((-1, 2)) if pairs else np.empty((0, 2), dtype=int)


def _rising_windows(values):
    d_values = np.diff(values, prepend=values[0])
    is_rising = d_values > 0
    edges = np.diff(is_rising.astype(int))
    start_idx = np.where(edges == 1)[0] + 1
    end_idx = np.where(edges == -1)[0] + 1
    if end_idx.size and start_idx.size and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]
    n_pairs = min(start_idx.size, end_idx.size)
    return np.column_stack((start_idx[:n_pairs], end_idx[:n_pairs])) if n_pairs else np.empty((0, 2), dtype=int)


def _active_windows(values):
    is_active = values > 0.0
    edges = np.diff(is_active.astype(int))
    start_idx = np.where(edges == 1)[0] + 1
    end_idx = np.where(edges == -1)[0] + 1
    if end_idx.size and start_idx.size and end_idx[0] < start_idx[0]:
        end_idx = end_idx[1:]
    n_pairs = min(start_idx.size, end_idx.size)
    return np.column_stack((start_idx[:n_pairs], end_idx[:n_pairs])) if n_pairs else np.empty((0, 2), dtype=int)


def _horizontal_target(ax, value, label, color, linestyle=(0, (2, 2))):
    if np.isfinite(value):
        ax.axhline(value, linestyle=linestyle, linewidth=TARGET_LINEWIDTH, color=color, alpha=0.9, label=label)


def _vertical_target(ax, value, label, color):
    if np.isfinite(value):
        ax.axvline(value, linestyle=(0, (2, 2)), linewidth=TARGET_LINEWIDTH, color=color, alpha=0.9, label=label)


def _legend_above(ax, handles, labels):
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _combined_legend(ax, *other_axes):
    handles, labels = ax.get_legend_handles_labels()
    for other_ax in other_axes:
        more_handles, more_labels = other_ax.get_legend_handles_labels()
        handles.extend(more_handles)
        labels.extend(more_labels)
    _legend_above(ax, handles, labels)


def _atrial_targets_legend(ax, pressure_ax):
    volume_handles, volume_labels = ax.get_legend_handles_labels()
    pressure_handles, pressure_labels = pressure_ax.get_legend_handles_labels()
    if volume_handles:
        volume_legend = ax.legend(
            volume_handles,
            volume_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(volume_handles),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        ax.add_artist(volume_legend)
    if pressure_handles:
        ax.legend(
            pressure_handles,
            pressure_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(pressure_handles),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _ventricular_pv_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    top_handles, top_labels = handles[:4], labels[:4]
    pressure_handles, pressure_labels = handles[4:8], labels[4:8]
    if top_handles:
        top_legend = ax.legend(
            top_handles,
            top_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=len(top_handles),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )
        ax.add_artist(top_legend)
    if pressure_handles:
        ax.legend(
            pressure_handles,
            pressure_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(pressure_handles),
            frameon=False,
            fontsize=SUBPLOT_LEGEND_FONT_SIZE,
            handlelength=1.8,
            columnspacing=0.8,
            handletextpad=0.4,
            borderaxespad=0.0,
        )


def _style_journal_axis(ax, secondary_y=False):
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", which="major", width=1.0, length=4, colors="#303030")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(1.1)
    if secondary_y:
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(True)
        ax.spines["right"].set_color("#555555")
        ax.spines["right"].set_linewidth(1.1)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    else:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#555555")
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.margins(y=0.08)


def _add_panel_letters(axes):
    for letter, ax in zip("ABCDEFGH", axes):
        ax.text(
            -0.16,
            1.12,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="#303030",
            clip_on=False,
            in_layout=False,
            zorder=20,
        )


def _collect_results_plot_data(conditions, buffer_limit):
    i_buffer = conditions["i"].item() % buffer_limit
    sorted_times = _sorted_buffer(conditions["all_time"], i_buffer)
    valid_time = np.isfinite(sorted_times) & (sorted_times < 1e5)

    buffer_keys = [
        "time_since_beat_store", "finish_breath_time", "HR_store", "P_sa_store",
        "V_lv_store", "V_rv_store", "P_lv_store", "P_rv_store", "V_la_store",
        "V_ra_store", "P_la_store", "P_ra_store", "theta_ao_store",
        "theta_po_store", "theta_mi_store", "theta_tr_store", "phi_atr_store",
        "tidal_store", "VAflow_store", "t1_store", "t2_store", "Q_pp_store",
        "Pa_O2_every_store", "Pa_CO2_every_store", "dP_lv_dt_store",
        "dP_rv_dt_store",
    ]
    traces = {"time": sorted_times[valid_time]}
    for key in buffer_keys:
        traces[key] = _sorted_buffer(conditions[key], i_buffer)[valid_time]

    theta_min = _parameter_value("theta_min")
    open_ao, _, pairs_ao = _valve_events(traces["theta_ao_store"], theta_min)
    _, _, pairs_po = _valve_events(traces["theta_po_store"], theta_min)
    _, _, pairs_mi = _valve_events(traces["theta_mi_store"], theta_min)
    _, _, pairs_tr = _valve_events(traces["theta_tr_store"], theta_min)

    open_ao = _tail_complete(open_ao)
    pairs_ao = _last_complete_pairs(pairs_ao)
    pairs_po = _last_complete_pairs(pairs_po)
    pairs_mi = _last_complete_pairs(pairs_mi)
    pairs_tr = _last_complete_pairs(pairs_tr)

    P_sa = traces["P_sa_store"]
    V_lv = traces["V_lv_store"]
    V_rv = traces["V_rv_store"]
    P_rv = traces["P_rv_store"]
    V_ra = traces["V_ra_store"]
    P_ra = traces["P_ra_store"]
    V_la = traces["V_la_store"]
    P_la = traces["P_la_store"]
    phi_atr = traces["phi_atr_store"]

    P_sa_max_idx = _window_indices(P_sa, pairs_ao, np.argmax)
    P_rv_max_idx = _window_indices(P_rv, pairs_po, np.argmax)
    P_rv_min_windows = np.asarray([(c, o_next) for (_, c), (o_next, _) in zip(pairs_po[:-1], pairs_po[1:])], dtype=int)
    P_rv_min_idx = _window_indices(P_rv, P_rv_min_windows, np.argmin)

    atrial_rise_windows = _last_complete_pairs(_rising_windows(phi_atr))
    P_la_max_idx = _window_indices(P_la, atrial_rise_windows, np.argmax)
    P_ra_max_idx = _window_indices(P_ra, atrial_rise_windows, np.argmax)
    P_la_descent2_idx = _window_indices(P_la, pairs_mi, np.argmin)
    P_la_descent1_windows = np.asarray([(c, o_next) for (_, c), (o_next, _) in zip(pairs_mi[:-1], pairs_mi[1:])], dtype=int)
    P_la_descent1_idx = _window_indices(P_la, P_la_descent1_windows, np.argmin)
    P_ra_descent2_idx = _window_indices(P_ra, pairs_tr, np.argmin)
    P_ra_descent1_windows = np.asarray([(c, o_next) for (_, c), (o_next, _) in zip(pairs_tr[:-1], pairs_tr[1:])], dtype=int)
    P_ra_descent1_idx = _window_indices(P_ra, P_ra_descent1_windows, np.argmin)

    atrial_active_windows = _last_complete_pairs(_active_windows(phi_atr))
    dP_lv_dt_idx = _window_indices(traces["dP_lv_dt_store"], atrial_active_windows, np.argmax)
    dP_rv_dt_idx = _window_indices(traces["dP_rv_dt_store"], atrial_active_windows, np.argmax)

    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_atrial_start_idx = _tail_complete(starts)

    HR_values = []
    previous_hr = None
    for current_hr in traces["HR_store"][::-1]:
        if not np.isfinite(current_hr):
            continue
        if previous_hr is None or current_hr != previous_hr:
            HR_values.append(current_hr)
            previous_hr = current_hr
            if len(HR_values) == 10:
                break

    dtr = np.diff(traces["finish_breath_time"])
    breath_starts = np.where(dtr > 0)[0] + 1
    tidal = traces["tidal_store"]
    if breath_starts.size >= 2 and breath_starts[-1] > breath_starts[-2]:
        max_tidal = float(np.nanmax(tidal[breath_starts[-2]:breath_starts[-1]]))
    else:
        positive_tidal = tidal[np.isfinite(tidal) & (tidal > 0)]
        max_tidal = float(np.nanmax(positive_tidal)) if positive_tidal.size else np.nan

    breath_period = traces["t1_store"] + traces["t2_store"]
    minute_ventilation_series = np.full_like(traces["VAflow_store"], np.nan, dtype=float)
    valid_breath_period = np.isfinite(breath_period) & (breath_period > 0)
    if np.any(valid_breath_period):
        VD = _parameter_value("GV_dead") * traces["VAflow_store"] + _parameter_value("V0_dead")
        minute_ventilation_series[valid_breath_period] = (
            traces["VAflow_store"][valid_breath_period] + VD[valid_breath_period] / breath_period[valid_breath_period]
        ) * 60
    valid_minute_ventilation = minute_ventilation_series[np.isfinite(minute_ventilation_series)]
    minute_ventilation = float(valid_minute_ventilation[-1]) if valid_minute_ventilation.size else np.nan

    total_volume = V_ra + V_rv + V_lv + V_la
    total_min_idx = _window_indices(total_volume, atrial_active_windows, np.argmin)
    total_max_idx = _window_indices(total_volume, atrial_active_windows, np.argmax)
    mean_min_total_volume = _mean_at(total_volume, total_min_idx)
    mean_max_total_volume = _mean_at(total_volume, total_max_idx)
    volume_percentage_change = (
        (mean_max_total_volume - mean_min_total_volume) / mean_max_total_volume
        if np.isfinite(mean_max_total_volume) and mean_max_total_volume != 0
        else np.nan
    )

    target_values_full = np.array([
        _mean_values(HR_values),
        _mean_at(P_sa, P_sa_max_idx),
        _mean_at(P_sa, open_ao),
        _mean_at(V_lv, pairs_ao[:, 0]),
        _mean_at(V_lv, pairs_ao[:, 1]),
        _mean_at(V_rv, pairs_po[:, 0]),
        _mean_at(V_rv, pairs_po[:, 1]),
        _mean_at(P_rv, P_rv_max_idx),
        _mean_at(P_rv, P_rv_min_idx),
        _mean_at(V_ra, pairs_tr[:, 1]),
        _mean_at(V_ra, pairs_tr[:, 0]),
        _mean_at(P_ra, P_ra_descent1_idx),
        _mean_at(P_ra, P_ra_max_idx),
        _mean_at(P_ra, pairs_tr[:, 0]),
        _mean_at(P_ra, P_ra_descent2_idx),
        _mean_at(V_la, pairs_mi[:, 1]),
        _mean_at(V_la, pairs_mi[:, 0]),
        _mean_at(P_la, P_la_descent1_idx),
        _mean_at(P_la, P_la_max_idx),
        _mean_at(P_la, pairs_mi[:, 0]),
        _mean_at(P_la, P_la_descent2_idx),
        _mean_at(V_la, local_atrial_start_idx),
        _mean_at(V_ra, local_atrial_start_idx),
        _mean_at(traces["dP_lv_dt_store"], dP_lv_dt_idx),
        _mean_at(traces["dP_rv_dt_store"], dP_rv_dt_idx),
        max_tidal,
        minute_ventilation,
        _mean_values(traces["Q_pp_store"]),
        _mean_values(traces["Pa_O2_every_store"]),
        _mean_values(traces["Pa_CO2_every_store"]),
        volume_percentage_change,
    ], dtype=float)

    keep_mask = np.ones(len(RESULT_OUTPUT_NAMES_FULL), dtype=bool)
    keep_mask[RESULT_COLS_TO_DROP] = False
    targets = {
        name: value for name, value, keep in zip(RESULT_OUTPUT_NAMES_FULL, target_values_full, keep_mask) if keep
    }
    return traces, minute_ventilation_series, targets


def plot_results_section(conditions, buffer_limit=BUFFER_LIMIT):
    traces, minute_ventilation_series, targets = _collect_results_plot_data(conditions, buffer_limit)
    time = traces["time"]
    if time.size == 0:
        raise ValueError("No valid circular-buffer time points were available for plotting.")

    print("Kept result targets:")
    for name, value in targets.items():
        print(f"{name}: {value:.6g}")

    step = max(1, int(np.ceil(time.size / 8000)))
    plot_slice = slice(None, None, step)
    t_plot = time[plot_slice]
    finite_time = time[np.isfinite(time)]
    time_axis_start = float(finite_time[0])
    time_axis_end = float(finite_time[-1])

    colors = PLOT_COLORS

    plt.rcParams.update(JOURNAL_RC_PARAMS)
    fig, axes = plt.subplots(4, 2, figsize=(11.0, 14.0), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.09, hspace=0.14, wspace=0.04)
    axes = axes.ravel()
    secondary_axes = []
    full_time_axes = []

    ax = axes[0]
    full_time_axes.append(ax)
    ax.plot(t_plot, traces["P_sa_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{sa}}$")
    _horizontal_target(ax, targets["Systolic_Pressure"], TARGET_LABELS["Systolic_Pressure"], colors["rose_dark"])
    _horizontal_target(ax, targets["Diastolic_Pressure"], TARGET_LABELS["Diastolic_Pressure"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$P_{\mathrm{sa}}$ (mmHg)")
    ax_hr = ax.twinx()
    secondary_axes.append(ax_hr)
    ax_hr.plot(
        t_plot,
        traces["HR_store"][plot_slice] * HEART_RATE_PLOT_SCALE,
        color=colors["solid_blue"],
        linewidth=SOLID_LINEWIDTH,
        label=TARGET_LABELS["Heart_Rate"],
    )
    _horizontal_target(ax_hr, targets["Heart_Rate"] * HEART_RATE_PLOT_SCALE, TARGET_MEAN_LABELS["Heart_Rate"], colors["teal_dark"])
    ax_hr.set_ylabel("HR (BPM)")
    _combined_legend(ax, ax_hr)

    ax = axes[1]
    pv_slice = slice(None, None, max(1, int(np.ceil(time.size / 12000))))
    ax.plot(traces["V_lv_store"][pv_slice], traces["P_lv_store"][pv_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label="LV")
    ax.plot(traces["V_rv_store"][pv_slice], traces["P_rv_store"][pv_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label="RV")
    _vertical_target(ax, targets["EDV"], TARGET_LABELS["EDV"], colors["rose_dark"])
    _vertical_target(ax, targets["ESV"], TARGET_LABELS["ESV"], colors["rose_light"])
    _vertical_target(ax, targets["Max_RV_Volume"], TARGET_LABELS["Max_RV_Volume"], colors["blue_dark"])
    _vertical_target(ax, targets["Min_RV_Volume"], TARGET_LABELS["Min_RV_Volume"], colors["blue_light"])
    _horizontal_target(ax, targets["Max_RV_Pressure"], TARGET_LABELS["Max_RV_Pressure"], colors["blue_dark"])
    _horizontal_target(ax, targets["Min_RV_Pressure"], TARGET_LABELS["Min_RV_Pressure"], colors["blue_light"])
    ax.set_xlabel(r"$V$ (mL)")
    ax.set_ylabel(r"$P$ (mmHg)")
    _ventricular_pv_legend(ax)

    ax = axes[2]
    ax.plot(traces["V_la_store"][pv_slice], traces["P_la_store"][pv_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label="LA")
    ax.plot(traces["V_ra_store"][pv_slice], traces["P_ra_store"][pv_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label="RA")
    ax.set_xlabel(r"$V$ (mL)")
    ax.set_ylabel(r"$P$ (mmHg)")
    _legend_above(ax, *ax.get_legend_handles_labels())

    ax = axes[3]
    ax.plot(t_plot, traces["V_ra_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$V_{\mathrm{RA}}$")
    _horizontal_target(ax, targets["Min_RA_Volume"], TARGET_LABELS["Min_RA_Volume"], colors["rose_dark"])
    _horizontal_target(ax, targets["Max_RA_Volume"], TARGET_LABELS["Max_RA_Volume"], colors["lavender_dark"], linestyle=(0, (5, 2)))
    _horizontal_target(ax, targets["RA_Volume_Before_Atrial_Contraction"], TARGET_LABELS["RA_Volume_Before_Atrial_Contraction"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_{\mathrm{RA}}$ (mL)")
    ax_p = ax.twinx()
    secondary_axes.append(ax_p)
    ax_p.plot(t_plot, traces["P_ra_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{RA}}$")
    _horizontal_target(ax_p, targets["Max_RA_Pressure_Atrial_contraction"], TARGET_LABELS["Max_RA_Pressure_Atrial_contraction"], colors["blue_dark"])
    _horizontal_target(ax_p, targets["Max_RA_Pressure_Tricuspid_Opening"], TARGET_LABELS["Max_RA_Pressure_Tricuspid_Opening"], colors["teal_light"])
    ax_p.set_ylabel(r"$P_{\mathrm{RA}}$ (mmHg)")
    _atrial_targets_legend(ax, ax_p)
    ax.set_xlim(ATRIAL_TARGET_TIME_START, time_axis_end)

    ax = axes[4]
    ax.plot(t_plot, traces["V_la_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$V_{\mathrm{LA}}$")
    _horizontal_target(ax, targets["Min_LA_Volume"], TARGET_LABELS["Min_LA_Volume"], colors["rose_dark"])
    _horizontal_target(ax, targets["Max_LA_Volume"], TARGET_LABELS["Max_LA_Volume"], colors["lavender_dark"], linestyle=(0, (5, 2)))
    _horizontal_target(ax, targets["LA_Volume_Before_Atrial_Contraction"], TARGET_LABELS["LA_Volume_Before_Atrial_Contraction"], colors["rose_light"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$V_{\mathrm{LA}}$ (mL)")
    ax_p = ax.twinx()
    secondary_axes.append(ax_p)
    ax_p.plot(t_plot, traces["P_la_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$P_{\mathrm{LA}}$")
    _horizontal_target(ax_p, targets["Max_LA_Pressure_Atrial_contraction"], TARGET_LABELS["Max_LA_Pressure_Atrial_contraction"], colors["blue_dark"])
    _horizontal_target(ax_p, targets["Max_LA_Pressure_Mitral_Opening"], TARGET_LABELS["Max_LA_Pressure_Mitral_Opening"], colors["teal_light"])
    ax_p.set_ylabel(r"$P_{\mathrm{LA}}$ (mmHg)")
    _atrial_targets_legend(ax, ax_p)
    ax.set_xlim(ATRIAL_TARGET_TIME_START, time_axis_end)

    ax = axes[5]
    ax.plot(t_plot, traces["dP_lv_dt_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=r"$\mathrm{d}P_{\mathrm{LV}}/\mathrm{d}t$")
    ax.plot(t_plot, traces["dP_rv_dt_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=r"$\mathrm{d}P_{\mathrm{RV}}/\mathrm{d}t$")
    _horizontal_target(ax, targets["LV_Pressure_Deriv"], TARGET_LABELS["LV_Pressure_Deriv"], colors["rose_dark"])
    _horizontal_target(ax, targets["RV_Pressure_Deriv"], TARGET_LABELS["RV_Pressure_Deriv"], colors["blue_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\mathrm{d}P/\mathrm{d}t$ (mmHg/s)")
    _legend_above(ax, *ax.get_legend_handles_labels())
    ax.set_xlim(DPDT_TIME_START, time_axis_end)

    ax = axes[6]
    full_time_axes.append(ax)
    ax.plot(t_plot, traces["tidal_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["Tidal_Volume"])
    _horizontal_target(ax, targets["Tidal_Volume"], TARGET_MEAN_LABELS["Tidal_Volume"], colors["rose_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inspired/Expired Volume (L)")
    ax_mv = ax.twinx()
    secondary_axes.append(ax_mv)
    ax_mv.plot(t_plot, minute_ventilation_series[plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["Minute_Ventilation"])
    _horizontal_target(ax_mv, targets["Minute_Ventilation"], TARGET_LABELS["Minute_Ventilation"], colors["teal_dark"])
    ax_mv.set_ylabel(r"$\dot{V}_E$ (L/min)")
    _combined_legend(ax, ax_mv)

    ax = axes[7]
    full_time_axes.append(ax)
    ax.plot(t_plot, traces["Pa_O2_every_store"][plot_slice], color=colors["solid_red"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["PaO2"])
    ax.plot(t_plot, traces["Pa_CO2_every_store"][plot_slice], color=colors["solid_blue"], linewidth=SOLID_LINEWIDTH, label=TARGET_LABELS["PaCO2"])
    _horizontal_target(ax, targets["PaO2"], TARGET_MEAN_LABELS["PaO2"], colors["rose_dark"])
    _horizontal_target(ax, targets["PaCO2"], TARGET_MEAN_LABELS["PaCO2"], colors["blue_dark"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$P_{\mathrm{a}O_2}$ / $P_{\mathrm{a}CO_2}$ (mmHg)")
    _legend_above(ax, *ax.get_legend_handles_labels())

    for axis in full_time_axes:
        axis.set_xlim(time_axis_start, time_axis_end)

    for axis in axes:
        _style_journal_axis(axis)
    for axis in secondary_axes:
        _style_journal_axis(axis, secondary_y=True)
    _add_panel_letters(axes)

    fig.savefig("Run_model_Paper_results_targets.png", dpi=600, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _load_simulation_cache(cache_path=CACHE_PATH):
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    if cache.get("version") != CACHE_VERSION:
        print(f"Ignoring stale simulation cache: {cache_path}")
        return None
    print(f"Loaded cached simulation results from {cache_path}")
    return cache


def _save_simulation_cache(conditions, solution, cache_path=CACHE_PATH):
    cache = {
        "version": CACHE_VERSION,
        "conditions": conditions,
        "solution_status": solution.status,
        "solution_message": solution.message,
    }
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved simulation results cache to {cache_path}")


def _use_cached_results():
    import argparse

    parser = argparse.ArgumentParser(description="Run or replot cached Run_model_Paper results.")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="rerun the ODE simulation and overwrite the cached plotting data",
    )
    args = parser.parse_args()
    return not args.rerun


if __name__ == "__main__":

    cache = _load_simulation_cache() if _use_cached_results() else None
    if cache is None:
        simulation_result = simulate()
        solution = simulation_result if hasattr(simulation_result, "status") else simulation_result[0]
        print("ODE Status:", solution.status)
        print("ODE Message:", solution.message)
        if hasattr(simulation_result, "status"):
            raise SystemExit("Skipping plots because the ODE solve did not return post-processed results.")
        conditions = simulation_result[11]
        _save_simulation_cache(conditions, solution)
    else:
        conditions = cache["conditions"]
        print("ODE Status:", cache.get("solution_status", "cached"))
        print("ODE Message:", cache.get("solution_message", "loaded from cache"))

    history_end = np.where(conditions["time_history"] == 1e6)[0]
    if history_end.size:
        print(history_end[0] - 1)

    plot_results_section(conditions)
