import os
import warnings
import joblib
import numpy as np
import pyro
from datetime import datetime
from multiprocessing import resource_tracker

import torch

# Prevent the resource tracker from complaining about shared memory cleanup
resource_tracker._resource_tracker._STOP = True
from SALib import ProblemSpec
from autoemulate.data.utils import set_random_seed
from History_matching_function_exercise_only import HistoryMatchingWorkflow
from AutoEmulate_Simulator_Exercise import Cardiopulmonary

# ----------------------------
# SETTINGS
# ----------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 42
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HM_RUNS_ROOT = os.path.join(SCRIPT_DIR, "HM_try")


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def create_hm_run_dir() -> str:
    requested_run_dir = os.environ.get("HM_RUN_DIR")
    if requested_run_dir:
        run_dir = requested_run_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(SCRIPT_DIR, run_dir)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    run_name = os.environ.get("HM_RUN_NAME") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(HM_RUNS_ROOT, run_name)
    base_run_dir = run_dir
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{suffix:02d}"
        suffix += 1

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Treat atrial contraction as a physiologic interval constraint rather than a
# point target for columns 17/18.
ATRIAL_RATIO_BOUNDS = (0.20, 0.30)
ATRIAL_RATIO_MIN_PROBABILITY = 0.05
ATRIAL_RATIO_MC_SAMPLES = 128

# ----------------------------
# PROBLEM SPECIFICATION
# ----------------------------
# change
percent = 50
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
        [0.026056388103 * lower, 0.026056388103 * upper],  # beta2 [MAP]
        [104.314359039737 * lower, 104.314359039737 * upper],  # C2 [MAP]
        [221.095091804686 * lower, 221.095091804686 * upper],  # K2 [MAP]
        [1.45760013289 * lower, 1.45760013289 * upper],  # a2 [MAP]
        [0.05591 * lower, 0.05591 * upper],
        [346000 * lower, 346000 * upper],
        [0.1698 * lower, 0.1698 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper],
        [0.2025 * lower, 0.2025 * upper],
        [0.00000000472 * lower, 0.00000000472 * upper],
        [0.190316563756 * lower, 0.190316563756 * upper],  # V0_dead [MAP]
        [0.0673 * lower, 0.0673 * upper],
        [24.914957306103 * lower, 24.914957306103 * upper],  # E_rs [MAP]
        [3.621416064005 * lower, 3.621416064005 * upper],  # R_rs [MAP]
        [3.150315009113 * lower, 3.150315009113 * upper],  # C_jp [MAP]
        [0.28 * lower, 0.28 * upper],
        [0.00022 * lower, 0.00022 * upper],
        [0.069297821233 * lower, 0.069297821233 * upper],  # R_sa [MAP]
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper],
        [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper],
        [53.700745884199 * lower, 53.700745884199 * upper],  # C_sv [MAP]
        [24.17 * lower, 24.17 * upper],
        [10 * lower, 10 * upper],
        [0.0833 * lower, 0.0833 * upper],
        [0.075 * lower, 0.075 * upper],
        [0.04 * lower, 0.04 * upper],
        [0.224 * lower, 0.224 * upper],
        [0.125 * lower, 0.125 * upper],
        [0.038 * lower, 0.038 * upper],
        [0.15 * lower, 0.15 * upper],
        [0.3855 * lower, 0.3855 * upper],
        [50 * lower, 50 * upper],
        [10000 * lower, 10000 * upper],
        [0.020008843735 * lower, 0.020008843735 * upper],  # Rvc_n [MAP]
        [5.85 * lower, 5.85 * upper],
        [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper],
        [0.018414673946 * lower, 0.018414673946 * upper],  # R_pa [MAP]
        [0.071550840611 * lower, 0.071550840611 * upper],  # R_pp [MAP]
        [0.0056 * lower, 0.0056 * upper],
        [0.539484450273 * lower, 0.539484450273 * upper],  # Emax_la [MAP]
        [0.539711575691 * lower, 0.539711575691 * upper],  # P0_la [MAP]
        [0.383211426561 * lower, 0.383211426561 * upper],  # Emax_ra [MAP]
        [0.539669442011 * lower, 0.539669442011 * upper],  # P0_ra [MAP]
        [0.057486653259 * lower, 0.057486653259 * upper],  # KE_la [MAP]
        [0.040029311567 * lower, 0.040029311567 * upper],  # KE_ra [MAP]
        [1.442044484136 * lower, 1.442044484136 * upper],  # P0_lv [MAP]
        [1.2459103951 * lower, 1.2459103951 * upper],  # P0_rv [MAP]
        [0.04 * lower, 0.04 * upper],
        [20.015768519964 * lower, 20.015768519964 * upper],  # fab_o [MAP]
        [18.538774839743 * lower, 18.538774839743 * upper],  # fes_o [MAP]
        [1.819256530955 * lower, 1.819256530955 * upper],  # fes_inf [MAP]
        [80 * lower, 80 * upper],
        [2.737077669881 * lower, 2.737077669881 * upper],  # fev_o [MAP]
        [5.491207566281 * lower, 5.491207566281 * upper],  # fev_inf [MAP]
        [0.059948847087 * lower, 0.059948847087 * upper],  # kes [MAP]
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper],
        [0.382454133527 * lower, 0.382454133527 * upper],  # Io_sv [MAP]
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper],
        [0.073914097993 * lower, 0.073914097993 * upper],  # kcc_sv [MAP]
        [0.0162 * lower, 0.0162 * upper],
        [9 * lower, 9 * upper],
        [-0.0283 * upper, -0.0283 * lower],
        [5.5 * lower, 5.5 * upper],
        [-0.037 * upper, -0.037 * lower],
        [64.9 * lower, 64.9 * upper],
        [-0.437 * upper, -0.437 * lower],
        [1.9 * lower, 1.9 * upper],
        [-0.0008 * upper, -0.0008 * lower],
        [-0.68 * upper, -0.68 * lower],
        [-1.98712918802 * upper, -1.98712918802 * lower],  # Wb_sh [MAP]
        [-1.1375 * upper, -1.1375 * lower],
        [-0.960143990803 * upper, -0.960143990803 * lower],  # Wb_sv [MAP]
        [1 * lower, 1 * upper],
        [1.716 * lower, 1.716 * upper],
        [1.716 * lower, 1.716 * upper],
        [0.2 * lower, 0.2 * upper],
        [-0.3997 * upper, -0.3997 * lower],
        [-0.3997 * upper, -0.3997 * lower],
        [-0.103 * upper, -0.103 * lower],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.4 * lower, 0.4 * upper],
        [2.666380393754 * lower, 2.666380393754 * upper],  # Emax_lv0 [MAP]
        [1.390021697651 * lower, 1.390021697651 * upper],  # Emax_rv0 [MAP]
        [3.189025830605 * lower, 3.189025830605 * upper],  # fes_min [MAP]
        [0.475 * lower, 0.475 * upper],
        [0.282 * lower, 0.282 * upper],
        [2.47 * lower, 2.47 * upper],
        [1.94 * lower, 1.94 * upper],
        [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper],
        [-58.29 * upper, -58.29 * lower],
        [-74.21 * upper, -74.21 * lower],
        [-58.29 * upper, -58.29 * lower],
        [-265.4 * upper, -265.4 * lower],
        [3.51 * lower, 3.51 * upper],
        [1.655 * lower, 1.655 * upper],
        [5.27 * lower, 5.27 * upper],
        [2.49 * lower, 2.49 * upper],
        [1 * lower, 1 * upper],
        [1.5 * lower, 1.5 * upper],
        [6 * lower, 6 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [45 * lower, 45 * upper],
        [30 * lower, 30 * upper],
        [30 * lower, 30 * upper],
        [3.6 * lower, 3.6 * upper],
        [13.32 * lower, 13.32 * upper],
        [11.458386800796 * lower, 11.458386800796 * upper],  # theta_svn [MAP]
        [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],
        [32.617140230052 * lower, 32.617140230052 * upper],  # PaCO2_n [MAP]
        [38.255264072196 * lower, 38.255264072196 * upper],  # f_ab_max [MAP]
        [2.52 * lower, 2.52 * upper],
        [10.652774100123 * lower, 10.652774100123 * upper],  # k_ab [MAP]
        [83.075364716124 * lower, 83.075364716124 * 1.05],  # P_n [MAP]
        [112 * 0.9, 112 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],
        [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],
        [-0.104052194839 * upper, -0.104052194839 * lower],  # GT_s [MAP]
        [0.075051827744 * lower, 0.075051827744 * upper],  # GT_v [MAP]
        [0.493091533616 * lower, 0.493091533616 * upper],  # T0 [MAP]
        [20.9 * lower, 20.9 * upper],
        [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower],
        [0.14 * lower, 0.14 * upper],
        [10 * lower, 10 * upper],
        [0.740470614424 * lower, 0.740470614424 * upper],  # MO2_bp [MAP]
        [6.57 * lower, 6.57 * upper],
        [0.11 * lower, 0.11 * upper],
        [0.155 * lower, 0.155 * upper],
        [35 * lower, 35 * upper],
        [30 * lower, 30 * upper],
        [11.11 * lower, 11.11 * upper],
        [142.8 * lower, 142.8 * upper],
        [0.4 * lower, 0.4 * upper],
        [0.86 * lower, 0.86 * upper],
        [19.71 * lower, 19.71 * upper],
        [12660 * lower, 12660 * upper],
        [0.135218326531 * lower, 0.135218326531 * upper],  # Cvam_O2_n [MAP]
        [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],
        [0.341510756868 * lower, 0.341510756868 * upper],  # Io_met [MAP]
        [0.149970059492 * lower, 0.149970059492 * upper],  # kmet [MAP]
        [0.516 * lower, 0.516 * upper],
        [20 * lower, 20 * upper],
        [-1.87 * upper, -1.87 * lower],
        [1000 * lower, 1000 * upper],
        [5000 * lower, 5000 * upper],
        [2 * lower, 2 * upper],
        [7 * lower, 7 * upper],
        [1.309 * lower, 1.309 * upper],
        [1200 * lower, 1200 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [4.197484149057 * lower, 4.197484149057 * upper],  # Kv_mi [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [2000 * lower, 2000 * upper],
        [2 * lower, 2 * upper],
        [8.082182497661 * lower, 8.082182497661 * upper],  # Kv_po [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [2.802882618089 * lower, 2.802882618089 * upper],  # Kv_tr [MAP]
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper],
        [350 * lower, 350 * upper],
        [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper],
        [0.001446768212 * lower, 0.001446768212 * upper],  # C_O2_param1 [MAP]
        [2.6 * lower, 2.6 * upper],
        [0.0000303 * lower, 0.0000303 * upper],
        [104 * lower, 104 * upper],
        [247.007713832753 * lower, 247.007713832753 * upper],  # Vu_bv [MAP]
        [93.16 * lower, 93.16 * upper],
        [464.099601030366 * lower, 464.099601030366 * upper],  # Vu_jp [MAP]
        [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper],
        [114 * lower, 114 * upper],
        [28.782795216021 * lower, 28.782795216021 * upper],  # Vu_la [MAP]
        [18.382176871651 * lower, 18.382176871651 * upper],  # Vu_lv [MAP]
        [31.978194395465 * lower, 31.978194395465 * upper],  # Vu_ra [MAP]
        [46.425736923554 * lower, 46.425736923554 * upper],  # Vu_rv [MAP]
        [8 * lower, 8 * upper],
        [8 * lower, 8 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [20 * lower, 20 * upper],
        [229.299959741592 * lower, 229.299959741592 * upper],  # Vu_amv0 [MAP]
        [727.971826624144 * lower, 727.971826624144 * upper],  # Vu_ev0 [MAP]
        [190.95 * lower, 190.95 * upper],
        [1163.779935864205 * lower, 1163.779935864205 * upper],  # Vu_sv0 [MAP]
        [20 * lower, 20 * upper],
        [30 * lower, 30 * upper],
        [2.076 * lower, 2.076 * upper],
        [0.8 * lower, 0.8 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [1.5 * lower, 1.5 * upper],
        [20 * lower, 20 * upper],
        [10 * lower, 10 * upper],
        [5 * lower, 5 * upper],
        [40 * lower, 40 * upper],
        [10 * lower, 10 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [2 * lower, 2 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [5 * lower, 5 * upper],
        [2 * lower, 2 * upper],
        [0.2 * lower, 0.2 * upper],
        [4 * lower, 4 * upper],
        [0.3 * lower, 0.3 * upper],
        [0.013195593572 * lower, 0.013195593572 * upper],  # KE_lv [MAP]
        [0.008823078488 * lower, 0.008823078488 * upper],  # KE_rv [MAP]
        [0.1 * lower, 0.1 * upper],
        [0.2 * lower, 0.2 * upper],
        [3 * lower, 3 * upper],
        [2.5 * lower, 2.5 * upper],
        [20 * lower, 20 * upper],
        [0.01 * lower, 0.01 * upper],
        [50 * lower, 50 * upper],
        [0.25 * lower, 0.25 * upper],
        [0.25 * lower, 0.25 * upper],
        [50 * lower, 50 * upper],
        [4.9 * lower, 4.9 * upper],
        [0.3 * lower, 0.3 * upper],
        [26.6 * lower, 26.6 * upper],
        [0.04 * lower, 0.04 * upper],
        [80 * lower, 80 * upper],
        [0.05165157622 * lower, 0.05165157622 * upper],  # rise_time_atr [MAP]
        [0.35970683371 * lower, 0.35970683371 * upper],  # rise_time_ven [MAP]
        [0.501698599678 * 0.85, 0.501698599678 * 1.15],  # fall_time_ven [MAP]
        [0.84695219825 * 0.92, 0.84695219825 * 1.08],  # ahead1 [MAP]
        [0.0873 * lower, 0.0873 * upper],
        [1.063125062813 * 0.85, 1.063125062813 * 1.15],  # r [MAP]
        [1.37966131513 * 0.85, 1.37966131513 * 1.15],  # l [MAP]
        [132.336714485812 * lower, 132.336714485812 * upper],  # V_nominal [MAP]
        [44.106896806161 * lower, 44.106896806161 * upper],  # V_scale [MAP]
    ]
})

# Exercise
# subset_vars = {'alpha2', 'C_amv', 'C_bv', 'C_ev', 'C_pa', 'C_pp', 'C_pv', 'C_sa', 'Cvb_O2_n', 'f_acCO2_n', 'G_ap',
#                'g_ccsh', 'GEmax_lv', 'GEmax_rv', 'gM', 'GR_amp', 'GV_amv', 'GV_dead', 'GV_ev', 'GV_sv', 'Io_sh',
#                'Io_sp', 'K1_vc', 'k_ac', 'KcCO2', 'KcMRV', 'kev', 'Kp_mi', 'Kp_po', 'MO2_ampn', 'P_n_max', 'PaO2_ac_n',
#                'phi_max', 'R_amp0', 'R_bpn', 'R_mi', 'R_po', 'R_tr', 'tauMR', 'theta_spn', 'theta_v', 'VA_rest', 'Wc_v',
#                'Wp_v', 'Ysh_max', 'Ysv_max', 'Yv_max'}
# subset_vars = {'a2', 'ahead1', 'C2', 'C_O2_param1', 'C_pv', 'C_sv', 'E_rs', 'Emax_lv0', 'Emax_ra', 'Emax_rv0',
#                'fall_time_ven', 'fes_o', 'fev_o', 'G_ap', 'GEmax_lv', 'GEmax_rv', 'GR_amp', 'GT_s', 'GT_v', 'GV_dead',
#                'GV_sv', 'Io_sh', 'KcCO2', 'KcMRV', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'l', 'P0_la', 'P0_lv', 'P0_rv',
#                'P_n_max', 'PaCO2_n', 'phi_max', 'r', 'R_amp0', 'R_pa', 'R_po', 'R_pp', 'R_rs', 'R_sa', 'rise_time_ven',
#                'Rvc_n', 'T0', 'tauMR', 'V0_dead', 'V_nominal', 'V_scale', 'VA_rest', 'Vu_ev0', 'Vu_jp', 'Vu_la',
#                'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wp_v', 'Yv_max',
#                # added
#                "C_pa", 'alpha2', 'C_ev', 'GV_ev', 'GV_amv', 'C_sa', 's'}
subset_vars = {'C_pv', 'G_ap', 'GEmax_lv', 'GEmax_rv', 'GR_amp', 'GV_dead', 'GV_sv', 'Io_sh', 'KcCO2', 'KcMRV',
                        'P_n_max', 'phi_max', 'R_amp0', 'R_po', 'tauMR', 'VA_rest', 'Wp_v', 'Yv_max',
                        'alpha2', 'C_amv', 'C_bv', 'C_ev', 'C_pa', 'C_pp', 'C_sa',
                        'Cvb_O2_n', 'f_acCO2_n', 'g_ccsh', 'gM', 'GV_amv', 'GV_ev',
                        'Io_sp', 'K1_vc', 'k_ac', 'kev', 'Kp_mi', 'Kp_po', 'MO2_ampn',
                        'PaO2_ac_n', 'R_bpn', 'R_mi', 'R_tr', 'theta_spn', 'theta_v',
                        'Wc_v', 'Ysh_max', 'Ysv_max'}


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
Heart_Rate_emulator = joblib.load(
    os.path.join(SCRIPT_DIR, "Heart_Rate", "GaussianProcessMatern32_Heart_Rate_best.joblib")
)

# # Exercise (second is variance, not standard deviation)
observation = {"Heart Rate": (2.58, 0.12), "Systolic Pressure": (165, 529), "Diastolic Pressure": (76.4, 82.81),
"EDV": (145.5, 681.21), "ESV": (45.5, 75.69), "Max RV Volume": (139.4, 681.21), "Min RV Volume": (40.3, 112.36),
"Max RV Pressure": (29.5, 56.25), "Min RV Pressure": (9.9, 31.36), "Min RA Volume": (27.9, 25.0),
"Max RA Volume": (77.3, 342.25), "Max RA Pressure Atrial contraction": (12, 16),
"Max RA Pressure Tricuspid Opening": (11, 16), "Min LA Volume": (23.0, 94.09), "Max LA Volume": (66.3, 388.09),
"Max LA Pressure Atrial contraction": (19, 49), "Max LA Pressure Mitral Opening": (19, 64),
"LA Contraction Volume diff": (33.8, 77.4), "RA Contraction Volume diff": (40.3, 36.0),
"LV Pressure Deriv": (1750, 272484), "RV Pressure Deriv": (713, 12100), "Tidal Volume": (2.22, 0.4096),
"Minute Ventilation": (62.6, 320.41), "PaO2": (97.2, 36.0), "PaCO2": (38.4, 6.76)}

# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":
    hm_run_dir = create_hm_run_dir()
    print(f"HM artifacts will be written to: {hm_run_dir}")

    # # nroy_samples_rest.pt rows are the same as in NROY_Points_rest_20.npy
    # AAA = np.load("Calibration_Exercise_New/NROY_Points_exercise_50.npy")
    # AAAA = np.load("Calibration_Exercise_New/NROY_Implaus_exercise_50.npy")
    # AAAAA = np.load("Calibration_Exercise_New/test_param_exercise_50.npy")
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
    #
    # mask2 = AA_filtered[:, 7] < 2
    # AA_filtered1  = AA_filtered[mask2]
    # AAA_filtered1 = AAA_filtered[mask2]
    #
    # param_keys = list(sp["names"])
    # param_samples = [dict(zip(param_keys, row)) for row in samples]
    # print(param_samples[-50])
    # A = np.load(r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper\HM_fifth_90_Exercise_Only\Wave_5\Min_LA_Volume\x_test_all.npy")
    # AA = np.load(r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper\HM_fifth_90_Exercise_Only\Wave_5\Min_LA_Volume\y_test.npy")
    # AAA = np.load(r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper\HM_fifth_90_Exercise_Only\Wave_5\LA_Contraction_Volume_diff\y_test.npy")
    # AA = AA.ravel()
    # AAA = AAA.ravel()
    # AAAA = AAA - AA
    # mask = AA < 27.28
    #
    # A_filt = A[mask]
    # AA_filt = AA[mask]
    # AAAA_filt = AAAA[mask]
    #
    # param_keys = list(sp["names"])
    # param_samples = [dict(zip(param_keys, row)) for row in A_filt]
    # print(param_samples[0])


    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=Heart_Rate_emulator,
        observations=observation,
        # optional parameters
        threshold=3.95,
        random_seed=random_seed,
        # train_x=X,
        # train_y=Result,
        calibration_params=subset_vars,
        atrial_ratio_bounds=ATRIAL_RATIO_BOUNDS,
        atrial_ratio_min_probability=ATRIAL_RATIO_MIN_PROBABILITY,
        atrial_ratio_mc_samples=ATRIAL_RATIO_MC_SAMPLES,
        run_dir=hm_run_dir,
    )

    # # --- PRE-WAVE: Train initial emulators from hybrid samples ---
    pre_wave_simulations = env_int("HM_PRE_WAVE_SIMULATIONS", 4096)
    wave_simulations = env_int("HM_WAVE_SIMULATIONS", 2048)
    size = env_int("HM_TEST_SAMPLES", 200000)

    # hmw.pre_wave_train_emulators(n_simulations=pre_wave_simulations, refit_on_all_data=False)
    # A = torch.load(r"C:\Users\vanes\Downloads\exercise_model\ODE_Exercise\Entire_system\DGSM_Exercise_Paper\HM_third_improved_LHCS_no_neg\last_wave.pt")
    _ = hmw.run_waves(
        n_waves=4,
        n_simulations=wave_simulations,
        n_test_samples=size,
        refit_on_all_data=False,
        refit_emulator_on_last_wave=True,
        max_retries=15,
        resume_wave=False,
        keep_all_wave_results=False,
    )

    # Get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)

    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points,
        param_names=sp["names"],
        buffer_ratio=0.0
    )

    np.save(os.path.join(hm_run_dir, f"NROY_Points_exercise_{percent}.npy"), nroy_points)
    np.save(os.path.join(hm_run_dir, f"NROY_Params_exercise_{percent}.npy"), params_post_hm)
    np.save(os.path.join(hm_run_dir, f"NROY_Implaus_exercise_{percent}.npy"), impl_scores)
    np.save(os.path.join(hm_run_dir, f"test_param_exercise_{percent}.npy"), test_parameters)

    print(hmw._last_completed_wave_idx)
    # hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{size}_wave_{(len(hmw.wave_results)-1)}_{percent}.png")
