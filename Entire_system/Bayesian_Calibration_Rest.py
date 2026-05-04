import os
import warnings
import joblib
import numpy as np
import pyro
from multiprocessing import resource_tracker

import torch

# Prevent the resource tracker from complaining about shared memory cleanup
resource_tracker._resource_tracker._STOP = True
from SALib import ProblemSpec
from autoemulate.data.utils import set_random_seed
from History_matching_function_rest import HistoryMatchingWorkflow
# from draft import HistoryMatchingWorkflow
from AutoEmulate_Simulator_Rest import Cardiopulmonary

# ----------------------------
# SETTINGS
# ----------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 42
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

# Treat atrial contraction as a physiologic interval constraint rather than a
# point target. The workflow estimates the ratio-prediction error empirically
# from the emulator outputs and uses that to score how likely each sample is to
# fall inside this band.
atrial_ratio_bounds = (0.20, 0.30)
atrial_ratio_min_probability = 0.05

# ----------------------------
# PROBLEM SPECIFICATION
# ----------------------------
# change
percent = 20
lower = 1 - percent/100
upper = 1 + percent/100
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
        "R_rmv_n", "R_sv_n", "K1_vc", "D1",
        "Vvc_min", "Kr_vc",
        "Rvc_n", "C_pa", "C_pp",
        "C_pv", "L_pa", "R_pa", "R_pp",
        "R_pv", "Emax_la", "P0_la", "Emax_ra",
        "P0_ra", "KE_la", "KE_ra", "P0_lv",
        "P0_rv",
        "s",
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
        [6.28 * lower, 6.28 * upper], [61.11 * lower, 61.11 * upper], [24.17 * lower, 24.17 * upper], [10 * lower, 10 * upper],
        [0.0833 * lower, 0.0833 * upper], [0.075 * lower, 0.075 * upper], [0.04 * lower, 0.04 * upper], [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper], [0.038 * lower, 0.038 * upper], [0.15 * lower, 0.15 * upper], [0.3855 * lower, 0.3855 * upper],
        [50 * lower, 50 * upper], [10000 * lower, 10000 * upper],
        [0.025 * lower, 0.025 * upper], [0.76 * lower, 0.76 * upper], [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper], [0.00018 * lower, 0.00018 * upper], [0.023 * lower, 0.023 * upper], [0.0894 * lower, 0.0894 * upper],
        [0.0056 * lower, 0.0056 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper], [0.45 * lower, 0.45 * upper],
        [0.45 * lower, 0.45 * upper], [0.05 * lower, 0.05 * upper], [0.05 * lower, 0.05 * upper], [1.5 * lower, 1.5 * upper],
        [1.5 * lower, 1.5 * upper],
        [0.04 * lower, 0.04 * upper],
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
        [0.58 * lower, 0.58 * upper], [20.9 * lower, 20.9 * upper], [92.8 * lower, 92.8 * upper], [10570 * lower, 10570 * upper],
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
        [116.68 * lower, 116.68 * upper], [114 * lower, 114 * upper], [24 * lower, 24 * upper], [15.908 * lower, 15.908 * upper],
        [24 * lower, 24 * upper], [38.703 * lower, 38.703 * upper],

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
        [0.045 * lower, 0.045 * upper], [0.3 * 0.8, 0.3 * 1.2], [0.45 * 0.85, 0.45 * 1.15], [0.92 * 0.92, 0.92 * 1.08],
        [0.0873 * lower, 0.0873 * upper], [1.2 * 0.85, 1.2 * 1.15], [1.2 * 0.85, 1.2 * 1.15], [150 * lower, 150 * upper], [50 * lower, 50 * upper]]
})

# # Rest: parameters contribute at least 1% and up to 90% sensitivity for 25 targets
subset_vars = {'a2', 'ahead1', 'beta2', 'C2', 'C_jp', 'C_O2_param1', 'C_sv', 'Cvam_O2_n', 'E_rs', 'Emax_la',
               'Emax_lv0', 'Emax_ra', 'Emax_rv0', 'f_ab_max', 'fab_o', 'fall_time_ven', 'fes_inf', 'fes_min',
               'fes_o', 'fev_inf', 'fev_o', 'GT_s', 'GT_v', 'Io_met', 'Io_sv', 'K2', 'k_ab', 'kcc_sv', 'KE_la',
               'KE_lv', 'KE_ra', 'KE_rv', 'kes', 'kmet', 'Kv_mi', 'Kv_po', 'Kv_tr', 'l', 'MO2_bp', 'P0_la', 'P0_lv',
               'P0_ra', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_pa', 'R_pp', 'R_rs', 'R_sa', 'rise_time_atr',
               'rise_time_ven', 'Rvc_n', 'T0', 'theta_svn', 'V0_dead', 'V_nominal', 'V_scale', 'Vu_amv0', 'Vu_bv',
               'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wb_sh', 'Wb_sv'}


# MUST SORT SO ITS THE SAME ORDER
subset_vars = [name for name in sp["names"] if name in subset_vars]

# Convert to dictionary
param_ranges: dict[str, tuple[float, float]] = {
    str(name): (lo := round(float(b[0]), 12), hi := round(float(b[1]), 12),) if str(name) in subset_vars else (
        m := round(0.5 * (float(b[0]) + float(b[1])), 12), m,)
    for name, b in zip(sp["names"], sp["bounds"])
}

output_names = [
    # The emulator folders for LA/RA contraction outputs are legacy names.
    # The raw simulator columns at indices 17/18 are pre-atrial chamber volumes,
    # which are later transformed inside `HistoryMatchingWorkflow` before
    # implausibility scoring.
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
# The LA/RA pre-atrial entries are retained for output alignment; when the
# ratio bounds below are active, the workflow enforces the atrial target via a
# probability-of-being-in-range filter instead of a point-target score.
observation = {"Heart Rate": (1.23, 0.05), "Systolic Pressure": (123, 324), "Diastolic Pressure": (76.7, 65.61), "EDV": (152.1, 767.29),
"ESV": (62.3, 243.36), "Max RV Volume": (151.9, 1004.89), "Min RV Volume": (64.4, 299.29), "Max RV Pressure": (22.5, 56.25),
"Min RV Pressure": (4.0, 9.0), "Min RA Volume": (45.7, 125.44), "Max RA Volume": (92.4, 380.25),
"Max RA Pressure Atrial contraction": (8.0, 9.0), "Max RA Pressure Tricuspid Opening": (5.0, 9.0), "Min LA Volume": (30.6, 84.64),
"Max LA Volume": (68.3, 306.25),
"Max LA Pressure Atrial contraction": (13.0, 9.0),
"Max LA Pressure Mitral Opening": (12.0, 9.0), "LA Contraction Volume diff": (41.8, 62.41), "RA Contraction Volume diff": (46.1, 73.96),
"LV Pressure Deriv": (1461.0, 146689.0), "RV Pressure Deriv": (271.0, 3025.0), "Tidal Volume": (0.850, 0.16),
"Minute Ventilation": (11.4, 15.21), "PaO2": (102.3, 125.44), "PaCO2": (35.5, 24.01)}


# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":

    # # nroy_samples_rest.pt rows are the same as in NROY_Points_rest_20.npy
    # AAA = np.load("Calibration_Rest_New/NROY_Points_rest_20_15_03.npy")
    # AAAA = np.load("Calibration_Rest_New/NROY_Implaus_rest_20_15_03.npy")
    # AAAAA = np.load("Calibration_Rest_New/test_param_rest_20_15_03.npy")
    # #
    # # # # Filter A and AA
    # mask = np.all(AAAA < 2.75, axis=1)
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
    # X = np.load("LHCS_X_20.npy")
    # Result = np.load("LHCS_Result_20.npy")
    # cols_to_drop = np.array([11, 14, 17, 20, 27, 30])
    # keep = np.setdiff1d(np.arange(Result.shape[1]), cols_to_drop)
    # Result = Result[:, keep]

    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=Heart_Rate_emulator,
        observations=observation,
        # optional parameters
        threshold=3.0,
        random_seed=random_seed,
        calibration_params=subset_vars,
        atrial_ratio_bounds=atrial_ratio_bounds,
        atrial_ratio_min_probability=atrial_ratio_min_probability,
    )

    # # X = torch.load("nroy_samples_rest_new.pt", map_location="cpu").to("cpu")
    # X_np = np.load('Calibration_31_03_26/NROY_Points_rest_20_26_03.npy')
    # X = torch.from_numpy(X_np).float()  # convert to torch tensor
    # subset_X = X[:, hmw.parameter_idx]
    # AA = np.load('Calibration_31_03_26/NROY_Params_rest_20_26_03.npy', allow_pickle=True)
    #
    #
    # models = {}
    # for name in output_names:
    #     folder = name
    #     path1 = os.path.join("Calibration_31_03_26/Emulator_wave", folder, f"GaussianProcessMatern32_{name}_best.joblib")
    #     models[name] = joblib.load(path1)
    #
    # means = {}
    # variances = {}
    #
    # for name in output_names:
    #     target_emulator = models[name]
    #
    #     means[name], variances[name] = target_emulator.predict_mean_and_variance(
    #         X[:, hmw.parameter_idx])
    #
    # mean_tensor = torch.cat([means[name].reshape(-1, 1) for name in output_names], dim=1)
    # var_tensor = torch.cat([variances[name].reshape(-1, 1) for name in output_names], dim=1)
    # impl_scores = hmw.calculate_implausibility(mean_tensor, var_tensor)
    # I_sorted, index_for_sort = torch.sort(impl_scores, dim=1, descending=True)
    # values, row_idx = torch.sort(I_sorted[:, 0], descending=True)
    #
    # worst_output_idx = index_for_sort[row_idx, 0]
    # mean_sorted = mean_tensor[row_idx]
    # bad_mean_values = mean_sorted[torch.arange(mean_sorted.shape[0]), worst_output_idx]
    #
    # implausibility_sorted_by_col0 = I_sorted[row_idx]
    # index_of_implausibility_sorted_by_col0 = index_for_sort[row_idx]
    # col0 = index_of_implausibility_sorted_by_col0[:, 0]
    # counts = np.bincount(col0, minlength=26)
    #
    # for value, count in enumerate(counts):
    #     print(f"{value}: {count}")
    #
    # import math
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.stats import gaussian_kde
    #
    # # indices of the 2500 lowest implausibility rows
    # lowest_2500_idx = row_idx[-2500:]
    #
    # # subset x and y to only those rows
    # x = subset_X[lowest_2500_idx]
    # y = mean_tensor[lowest_2500_idx, 7]
    #
    # X_np_low = X_np[lowest_2500_idx]
    # param_samples = [dict(zip(list(sp["names"]), row)) for row in X_np_low]
    # print(param_samples[0])

    # x = subset_X
    # y = mean_tensor[:, 0]
    #
    # D = x.shape[1]  # number of parameters
    # ncols = 4
    # nrows = math.ceil(D / ncols)
    #
    # fig, axes = plt.subplots(
    #     nrows, ncols,
    #     figsize=(3.5 * ncols, 3 * nrows),
    #     sharey=True
    # )
    #
    # axes = np.array(axes).reshape(nrows, ncols)
    # param_names = list(subset_vars)
    #
    # for j in range(D):
    #     r = j // ncols
    #     c = j % ncols
    #     ax = axes[r, c]
    #
    #     xj = np.asarray(x[:, j]).ravel()
    #     yj = np.asarray(y).ravel()
    #
    #     # Compute 2D point density
    #     xy = np.vstack([xj, yj])
    #     z = gaussian_kde(xy)(xy)
    #
    #     # Sort so dense points are plotted on top
    #     idx = z.argsort()
    #     xj, yj, z = xj[idx], yj[idx], z[idx]
    #
    #     sc = ax.scatter(
    #         xj, yj,
    #         c=z,
    #         cmap="viridis",
    #         s=6,
    #         alpha=0.8
    #     )
    #
    #     ax.set_title(param_names[j])
    #     ax.set_xlabel(param_names[j])
    #
    #     if c == 0:
    #         ax.set_ylabel("Mean tensor[:, 0]")
    #
    #     # Set x-axis bounds from AA
    #     if param_names[j] in AA:
    #         lower, upper = AA[param_names[j]]
    #         ax.set_xlim(lower, upper)
    #
    #     ax.grid(True, linestyle="--", alpha=0.3)
    #
    # # Turn off unused axes
    # for j in range(D, nrows * ncols):
    #     r = j // ncols
    #     c = j % ncols
    #     axes[r, c].axis("off")
    #
    # # # Optional shared colorbar
    # # cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
    # # cbar.set_label("Point density")
    #
    # plt.tight_layout()
    # plt.savefig("my_plot3.png", dpi=300, bbox_inches="tight")
    # # plt.show()
    # plt.close()

    size = 200000
    _ = hmw.run_waves(n_waves=5, n_simulations=2048, n_test_samples=size, refit_on_all_data=False, refit_emulator_on_last_wave=False, max_retries=15, resume_wave=False)

    # Get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)

    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points,
        param_names=sp["names"],
        buffer_ratio=0.0
    )

    np.save(f"NROY_Points_rest_{percent}_12_4.npy", nroy_points)
    np.save(f"NROY_Params_rest_{percent}_12_4.npy", params_post_hm)
    np.save(f"NROY_Implaus_rest_{percent}_12_4.npy", impl_scores)
    np.save(f"test_param_rest_{percent}_12_4.npy", test_parameters)


    # hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{size}_wave_{(len(hmw.wave_results))}_{percent}_rest.png")
    print(len(hmw.wave_results)-1)





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
