import os
import warnings
import joblib
import numpy as np
import pyro
from multiprocessing import resource_tracker
# Prevent the resource tracker from complaining about shared memory cleanup
resource_tracker._resource_tracker._STOP = True
from SALib import ProblemSpec
from autoemulate.data.utils import set_random_seed
from History_matching_function_new import HistoryMatchingWorkflow
from AutoEmulate_Simulator import Cardiopulmonary

# ----------------------------
# SETTINGS
# ----------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 42
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

# ----------------------------
# PROBLEM SPECIFICATION
# ----------------------------
# change
percent = 0.2
lower = 1 - percent
upper = 1 + percent
#
sp = ProblemSpec({
    'names': [
        # gas
        "beta2", "C2", "K2", "a2",
        "alpha2", "KCCO2", "GV_dead",
        # resp control
        "KcCO2", "KcMRV", "KpCO2", "KpO2",
        "V0_dead", "VA_rest",
        "E_rs", "R_rs",
        # cardio
        "C_jp", "C_sa", "L_sa", "R_sa",
        "C_amv", "C_bv", "C_ev", "C_hv",
        "C_rmv", "C_sv", "kr_am", "P_0",
        "R_amv_n", "R_bv_n", "R_ev_n", "R_hv_n",
        "R_rmv_n", "R_sv_n", "K1_vc",
        "Rvc_n", "C_pa", "C_pp",
        "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv", # "g_thor", "P_thormax_n", "P_thormin_n",
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
        "Wp_sp", "Wp_sv", "Wp_v",
        "Wt_sh", "Wt_sp", "Wt_sv", "Wt_v",
        "Emax_lv0", "Emax_rv0", "fes_min", "GEmax_lv",
        "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp",
        "GR_sp", "GV_amv", "GV_ev", "GV_rmv",
        "GV_sv", "R_amp0", "R_ep0", "R_rmp0",
        #
        "R_sp0", "g_ccsh", "g_ccsp",
        "kisc_sh", "kisc_sp", "kisc_sv",
        "PO2_sh", "PO2_sp", "PO2_sv", "theta_shn",
        "theta_spn", "theta_svn", "x_sh", "x_sp",
        "x_sv", "PaCO2_n", "f_ab_max", "f_ab_min",
        "k_ab", "P_n", "P_n_max", "f_acCO2_n",
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
        "PAMO2_nominal", "Vu_bv", "Vu_hv",
        "Vu_jp", "Vu_vc",
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
        "scale_param1", "scale_param3", "scale_param4",
        "scale_param6", "Pa_O2_lower",
        "rise_time_atr", "rise_time_ven", "fall_time_ven", "ahead1",
        "theta_min", "r", "l", "V_nominal", "V_scale"
    ],

    'bounds': [
        # gas
        [0.03255 * lower, 0.03255 * upper], [87 * lower, 87 * upper], [194.4 * lower, 194.4 * upper], [1.819 * lower, 1.819 * upper],
        [0.05591 * lower, 0.05591 * upper], [346000 * lower, 346000 * upper], [0.1698 * lower, 0.1698 * upper],
        # resp control
        [0.2332 * lower, 0.2332 * upper], [1 * lower, 1 * upper], [0.2025 * lower, 0.2025 * upper], [4.72e-09 * lower, 4.72e-09 * upper],
        [0.1587 * lower, 0.1587 * upper], [0.0673 * lower, 0.0673 * upper],
        [21.9 * 0.8, 21.9 * 1.2], [3.02 * 0.8, 3.02 * 1.2],
        # cardio
        [3.72 * lower, 3.72 * upper], [0.28 * lower, 0.28 * upper], [0.00022 * lower, 0.00022 * upper], [0.06 * lower, 0.06 * upper],
        [9.4 * lower, 9.4 * upper], [10.71 * lower, 10.71 * upper], [20 * lower, 20 * upper], [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper], [24.17 * lower, 24.17 * upper], [3.93 * lower, 3.93 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.15 * lower, 0.15 * upper],
        [0.0025 * lower, 0.0025 * upper], [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper], [0.05 * lower, 0.05 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper], # [6.8 * lower, 6.8 * upper], [-2 * upper, -2 * lower], [-6 * upper, -6 * lower],
        [0.73 * lower, 0.73 * upper], [0.04 * lower, 0.04 * upper],
        # cardio control
        [25 * lower, 25 * upper], [16.11 * lower, 16.11 * upper], [2.1 * lower, 2.1 * upper], [80 * lower, 80 * upper],
        [3.2 * lower, 3.2 * upper], [6.3 * lower, 6.3 * upper], [0.0675 * lower, 0.0675 * upper], [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper], [0.65 * lower, 0.65 * upper], [0.45 * lower, 0.45 * upper], [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper], [0.13 * lower, 0.13 * upper], [0.09 * lower, 0.09 * upper], [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper], [-0.0283 * upper, -0.0283 * lower], [5.5 * lower, 5.5 * upper], [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper], [-0.437 * upper, -0.437 * lower], [1.9 * lower, 1.9 * upper], [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower], [-1.75 * upper, -1.75 * lower], [-1.1375 * upper, -1.1375 * lower], [-1.1375 * upper, -1.1375 * lower],
        [1 * lower, 1 * upper], [1.716 * lower, 1.716 * upper], [1.716 * lower, 1.716 * upper], [0.2 * lower, 0.2 * upper],
        [-0.3997 * upper, -0.3997 * lower], [-0.3997 * upper, -0.3997 * lower], [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper], [0.4 * lower, 0.4 * upper],
        [2.392 * lower, 2.392 * upper], [1.412 * lower, 1.412 * upper], [2.66 * lower, 2.66 * upper], [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper], [2.47 * lower, 2.47 * upper], [1.94 * lower, 1.94 * upper], [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper], [-58.29 * upper, -58.29 * lower], [-74.21 * upper, -74.21 * lower], [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower], [3.51 * lower, 3.51 * upper], [1.655 * lower, 1.655 * upper], [5.27 * lower, 5.27 * upper],
        #
        [2.49 * lower, 2.49 * upper], [1 * lower, 1 * upper], [1.5 * lower, 1.5 * upper],
        [6 * lower, 6 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [45 * lower, 45 * upper], [30 * lower, 30 * upper], [30 * lower, 30 * upper], [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper], [13.32 * lower, 13.32 * upper], [53 * lower, 53 * upper], [6 * lower, 6 * upper],
        [6 * lower, 6 * upper], [40 * lower, 40 * upper], [47.78 * lower, 47.78 * upper], [2.52 * lower, 2.52 * upper],
        [11.76 * lower, 11.76 * upper], [92 * lower, 92 * 1.05], [112 * 0.9, 112 * upper], [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper], [0.835 * lower, 0.835 * upper], [29.27 * lower, 29.27 * upper], [3 * lower, 3 * upper],
        [45 * lower, 45 * upper], [11.76 * lower, 11.76 * upper], [-0.13 * upper, -0.13 * lower], [0.09 * lower, 0.09 * upper],
        [0.58 * lower, 0.58 * upper],  [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower], [0.14 * lower, 0.14 * upper], [10 * lower, 10 * upper], [0.925 * lower, 0.925 * upper],
        [6.57 * lower, 6.57 * upper], [0.11 * lower, 0.11 * upper], [0.155 * lower, 0.155 * upper], [35 * lower, 35 * upper],
        [30 * lower, 30 * upper], [11.11 * lower, 11.11 * upper], [142.8 * lower, 142.8 * upper], [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper], [19.71 * lower, 19.71 * upper], [12660 * lower, 12660 * upper], [0.1555 * lower, 0.1555 * upper],
        [30 * lower, 30 * upper], [40 * lower, 40 * upper], [0.4266 * lower, 0.4266 * upper], [0.18 * lower, 0.18 * upper],
        [0.516 * lower, 0.516 * upper], [20 * lower, 20 * upper], [-1.87 * upper, -1.87 * lower],
        # added params
        [1000 * lower, 1000 * upper], [5000 * lower, 5000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper],
        [1200 * lower, 1200 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [2000 * lower, 2000 * upper], [2 * lower, 2 * upper], [7 * lower, 7 * upper], [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper], [200 * lower, 200 * upper], [2 * lower, 2 * upper], [3.5 * lower, 3.5 * upper], [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper], [350 * lower, 350 * upper], [400 * lower, 400 * upper], [400 * lower, 400 * upper],
        [350 * lower, 350 * upper], [0.00134 * lower, 0.00134 * upper], [2.6 * lower, 2.6 * upper], [3.03e-5 * lower, 3.03e-5 * upper],
        [104 * lower, 104 * upper], [279.49 * lower, 279.49 * upper], [93.16 * lower, 93.16 * upper],
        [579.76 * lower, 579.76 * upper], [123 * lower, 123 * upper],
        [116.6775 * lower, 116.6775 * upper], [114 * lower, 114 * upper], [50 * lower, 50 * upper], [15.908 * lower, 15.908 * upper],
        [90 * lower, 90 * upper], [38.703 * lower, 38.703 * upper],

        # [5027.6 * 0.8, 5027.6 * 1.2],
        [8 * lower, 8 * upper], [8 * lower, 8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [20 * lower, 20 * upper],
        [20 * lower, 20 * upper], [20 * lower, 20 * upper], [20 * lower, 20 * upper], [286.4 * lower, 286.4 * upper],
        [607.8 * lower, 607.8 * upper], [190.95 * lower, 190.95 * upper], [1361.6 * lower, 1361.6 * upper], [20 * lower, 20 * upper],
        [30 * lower, 30 * upper], [2.076 * lower, 2.076 * upper], [0.8 * lower, 0.8 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [1.5 * lower, 1.5 * upper], [20 * lower, 20 * upper],
        [10 * lower, 10 * upper], [5 * lower, 5 * upper], [40 * lower, 40 * upper], [10 * lower, 10 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper], [2 * lower, 2 * upper],
        [2 * lower, 2 * upper], [2 * lower, 2 * upper], [5 * lower, 5 * upper], [5 * lower, 5 * upper],
        [5 * lower, 5 * upper], [5 * lower, 5 * upper], [2 * lower, 2 * upper], [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper], [0.3 * lower, 0.3 * upper], [0.014 * lower, 0.014 * upper], [0.011 * lower, 0.011 * upper],
        [0.1 * lower, 0.1 * upper], [0.2 * lower, 0.2 * upper], [3 * lower, 3 * upper], [2.5 * lower, 2.5 * upper],
        [20 * lower, 20 * upper], [0.01 * lower, 0.01 * upper], [50 * lower, 50 * upper], [0.25 * lower, 0.25 * upper],
        [0.25 * lower, 0.25 * upper], [50 * lower, 50 * upper],

        # further added params
        [4.9 * lower, 4.9 * upper], [0.3 * lower, 0.3 * upper], [26.6 * lower, 26.6 * upper],
        [0.04 * lower, 0.04 * upper], [80 * lower, 80 * upper],
        [0.05 * lower, 0.05 * upper], [0.15 * lower, 0.15 * upper], [0.3 * 0.8, 0.3 * 1.2], [0.9 * 0.95, 0.9 * 1.05],
        [0.0872665 * lower, 0.0872665 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15], [150 * lower, 150 * upper], [50 * lower, 50 * upper]]
})

# 64 parameters contribute at least 1% and up to 90% sensitivity for 21 targets
subset_vars = {'a2', 'C2', 'C_jp', 'C_O2_param1', 'C_pa', 'C_sa', 'C_sv', 'Cvam_O2_n', 'E_rs', 'Emax_la',
                   'Emax_lv0', 'Emax_rv0', 'f_ab_max', 'fab_o', 'fes_inf', 'fes_min', 'fes_o', 'fev_inf', 'fev_o',
                   'GT_s', 'GT_v', 'Io_met', 'Io_sv', 'K2', 'k_ab', 'kcc_sv', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'kes',
                   'kmet', 'Kv_mi', 'MO2_bp', 'P0_la', 'P0_lv', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_pa', 'R_po', 'R_pp',
                   'R_rs', 'R_sa', 'rise_time_atr', 'rise_time_ven', 'T0', 'theta_svn', 'V0_dead', 'V_nominal',
                   'V_scale', 'Vu_amv0', 'Vu_bv', 'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0',
                   'Wb_sh', 'Wb_sp', 'Wb_sv'}

# MUST SORT SO ITS THE SAME ORDER
subset_vars = [name for name in sp["names"] if name in subset_vars]

# Convert to dictionary
param_ranges: dict[str, tuple[float, float]] = {
    str(name): (lo := round(float(b[0]), 12), hi := round(float(b[1]), 12),) if str(name) in subset_vars else (
        m := round(0.5 * (float(b[0]) + float(b[1])), 12), m,)
    for name, b in zip(sp["names"], sp["bounds"])
}

output_names = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV",
    "ESV", "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure",
    "Min_RV_Pressure", "Min_RA_Volume", "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume", "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff", "LV_Pressure_Deriv",
    "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation", "PaO2",
    "PaCO2"]

# ----------------------------
# LOAD SIMULATOR
# ----------------------------
Simulator = Cardiopulmonary(param_ranges=param_ranges, output_names=output_names)


# ----------------------------
# LOAD EMULATOR
# ----------------------------
# change (emulator for rest/exercise)
Heart_Rate_emulator = joblib.load("Heart_Rate/GaussianProcessMatern32_Heart_Rate_best.joblib")


# # # rest
# observation = {"Heart Rate": (1.1, 0.1), "Systolic Pressure": (105, 5), "Diastolic Pressure": (70, 3), "EDV": (163, 23),
# "ESV": (50, 10), "Max RV Volume": (186, 21), "Min RV Volume": (52, 9), "Max RV Pressure": (24, 2),
# "Min RV Pressure": (2, 1), "Min RA Volume": (45, 15), "Max RA Volume": (93, 16),
# "Max RA Pressure Atrial contraction": (7, 2), "Max RA Pressure Tricuspid Opening": (7, 2), "Min LA Volume": (45, 15),
# "Max LA Volume": (72, 12),
# "Max LA Pressure Atrial contraction": (7, 2),
# "Max LA Pressure Mitral Opening": (7, 2), "LA Contraction Volume diff": (10, 2), "RA Contraction Volume diff": (10, 2),
# "LV Pressure Deriv": (1600, 305), "RV Pressure Deriv": (500, 150), "Tidal Volume": (0.5, 0.1),
# "Minute Ventilation": (6.5, 0.5), "PaO2": (95, 4.5), "PaCO2": (40, 2)}

# # rest (second is variance, not standard deviation)
observation = {"Heart Rate": (1.1, 0.01), "Systolic Pressure": (105, 25), "Diastolic Pressure": (70, 9), "EDV": (163, 529),
"ESV": (50, 100), "Max RV Volume": (186, 441), "Min RV Volume": (52, 81), "Max RV Pressure": (24, 4),
"Min RV Pressure": (2, 1), "Min RA Volume": (45, 225), "Max RA Volume": (93, 256),
"Max RA Pressure Atrial contraction": (7, 4), "Max RA Pressure Tricuspid Opening": (7, 4), "Min LA Volume": (45, 225),
"Max LA Volume": (72, 144),
"Max LA Pressure Atrial contraction": (7, 4),
"Max LA Pressure Mitral Opening": (7, 4), "LA Contraction Volume diff": (10, 4), "RA Contraction Volume diff": (10, 4),
"LV Pressure Deriv": (1600, 93025), "RV Pressure Deriv": (500, 22500), "Tidal Volume": (0.5, 0.01),
"Minute Ventilation": (6.5, 0.25), "PaO2": (95, 20.25), "PaCO2": (40, 4)}


# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":

    # AAA = np.load("NROY_Points_rest_20.npy")
    # AAAA = np.load("NROY_Implaus_rest_20.npy")
    # AAAAA = np.load("test_param_rest_20.npy")
    # # # # Filter A and AA
    # mask = np.all(AAAA < 3, axis=1)
    # AAAA_filtered = AAAA[mask]
    # AAAAA_filtered = AAAAA[mask]
    # index_for_sort = np.argsort(-AAAA_filtered, axis=1)
    # I_sorted = np.take_along_axis(AAAA_filtered, index_for_sort, axis=1)
    # row_idx = np.argsort(-I_sorted[:, 0])
    # implausibility_sorted_by_col0 = I_sorted[row_idx]
    # index_of_implausibility_sorted_by_col0 = index_for_sort[row_idx]
    # samples = AAAAA_filtered[row_idx]
    # #
    # # mask2 = AA_filtered[:, 7] < 2
    # # AA_filtered1  = AA_filtered[mask2]
    # # AAA_filtered1 = AAA_filtered[mask2]
    # #
    # param_keys = list(sp["names"])
    # param_samples = [dict(zip(param_keys, row)) for row in samples]
    # print(param_samples[-1])


    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=Heart_Rate_emulator,
        observations=observation,
        # optional parameters
        threshold=6,
        random_seed=random_seed,
        # train_x=X,
        # train_y=Result,
        calibration_params=subset_vars,
    )

    size = 200000
    _ = hmw.run_waves(n_waves=9, n_simulations=2048, n_test_samples=size, refit_on_all_data=False, refit_emulator_on_last_wave=True, max_retries=15, resume_wave=True)

    # Get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)

    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points,
        param_names=sp["names"],
        buffer_ratio=0.0
    )

    np.save(f"NROY_Points_rest_{percent}.npy", nroy_points)
    np.save(f"NROY_Params_rest_{percent}.npy", params_post_hm)
    np.save(f"NROY_Implaus_rest_{percent}.npy", impl_scores)
    np.save(f"test_param_rest_{percent}.npy", test_parameters)


    hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{size}_wave_{(len(hmw.wave_results)-1)}_{percent}.png")
    print(len(hmw.wave_results)-1)
    hmw.plot_wave((len(hmw.wave_results)-2), fname=f"{size}_wave_{(len(hmw.wave_results)-2)}_{percent}.png")
    hmw.plot_wave((len(hmw.wave_results)-3), fname=f"{size}_wave_{(len(hmw.wave_results)-3)}_{percent}.png")
    hmw.plot_wave((len(hmw.wave_results)-4), fname=f"{size}_wave_{(len(hmw.wave_results)-4)}_{percent}.png")
    hmw.plot_wave((len(hmw.wave_results)-4), fname=f"{size}_wave_{(len(hmw.wave_results)-5)}_{percent}.png")






    #
    # model_post_hm = hmw.emulator  # Use the emulator from history matching
    # parameter_range = {k: v for k, v in params_post_hm.items() if k in subset_vars}
    #
    # bc = BayesianCalibration(
    #     emulator=model_post_hm,
    #     parameter_range=parameter_range,
    #     observations={k: torch.tensor(v[0]) for k, v in observation.items()},
    #     # take account of the emulator uncertainty
    #     model_uncertainty=True,
    #     # specify observation noise as variance
    #     observation_noise={k: v[1] ** 2 for k, v in observation.items()}
    # )
