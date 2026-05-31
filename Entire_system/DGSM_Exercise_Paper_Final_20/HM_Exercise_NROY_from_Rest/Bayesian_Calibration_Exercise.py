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
from History_matching_function_exercise import HistoryMatchingWorkflow
from AutoEmulate_Simulator_Exercise import Cardiopulmonary

# ----------------------------
# SETTINGS
# ----------------------------
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

random_seed = 42
set_random_seed(random_seed)
pyro.set_rng_seed(random_seed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REST_POSTERIOR_RUN_DIR = os.path.join(
    BASE_DIR, "MCMC_Rest_20_25_05_1500_logspline_copula_prior"
)
REST_POSTERIOR_MASS = 0.95
REST_POSTERIOR_REGION = "hpd"
REST_OVERLAP_SAMPLING = "cloud"

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
        [0.03255 * lower, 0.03255 * upper],
        [98.640209672481 * lower, 98.640209672481 * upper],  # C2 [MAP]
        [168.668058520599 * lower, 168.668058520599 * upper],  # K2 [MAP]
        [2.05720263191 * lower, 2.05720263191 * upper],  # a2 [MAP]
        [0.05591 * lower, 0.05591 * upper],
        [346000 * lower, 346000 * upper],
        [0.1698 * lower, 0.1698 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper],
        [0.2025 * lower, 0.2025 * upper],
        [0.00000000472 * lower, 0.00000000472 * upper],
        [0.180760531763 * lower, 0.180760531763 * upper],  # V0_dead [MAP]
        [0.0673 * lower, 0.0673 * upper],
        [24.771332348072 * lower, 24.771332348072 * upper],  # E_rs [MAP]
        [3.308951736968 * lower, 3.308951736968 * upper],  # R_rs [MAP]
        [3.178371624738 * lower, 3.178371624738 * upper],  # C_jp [MAP]
        [0.28 * lower, 0.28 * upper],
        [0.00022 * lower, 0.00022 * upper],
        [0.067627524644 * lower, 0.067627524644 * upper],  # R_sa [MAP]
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper],
        [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper],
        [57.296181668841 * lower, 57.296181668841 * upper],  # C_sv [MAP]
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
        [0.020011011618 * lower, 0.020011011618 * upper],  # Rvc_n [MAP]
        [5.85 * lower, 5.85 * upper],
        [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper],
        [0.023 * lower, 0.023 * upper],
        [0.082324220616 * lower, 0.082324220616 * upper],  # R_pp [MAP]
        [0.0056 * lower, 0.0056 * upper],
        [0.387283349395 * lower, 0.387283349395 * upper],  # Emax_la [MAP]
        [0.51807469575 * lower, 0.51807469575 * upper],  # P0_la [MAP]
        [0.360330241084 * lower, 0.360330241084 * upper],  # Emax_ra [MAP]
        [0.380765929195 * lower, 0.380765929195 * upper],  # P0_ra [MAP]
        [0.057937819808 * lower, 0.057937819808 * upper],  # KE_la [MAP]
        [0.04293465892 * lower, 0.04293465892 * upper],  # KE_ra [MAP]
        [1.442691243336 * lower, 1.442691243336 * upper],  # P0_lv [MAP]
        [1.29130584142 * lower, 1.29130584142 * upper],  # P0_rv [MAP]
        [0.034856261421 * lower, 0.034856261421 * upper],  # s [MAP]
        [21.617810506166 * lower, 21.617810506166 * upper],  # fab_o [MAP]
        [17.175133514906 * lower, 17.175133514906 * upper],  # fes_o [MAP]
        [2.07231183456 * lower, 2.07231183456 * upper],  # fes_inf [MAP]
        [80 * lower, 80 * upper],
        [2.770365085494 * lower, 2.770365085494 * upper],  # fev_o [MAP]
        [7.423024627183 * lower, 7.423024627183 * upper],  # fev_inf [MAP]
        [0.05880583949 * lower, 0.05880583949 * upper],  # kes [MAP]
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper],
        [0.511820670232 * lower, 0.511820670232 * upper],  # Io_sv [MAP]
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper],
        [0.080130200394 * lower, 0.080130200394 * upper],  # kcc_sv [MAP]
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
        [-2.018544472453 * upper, -2.018544472453 * lower],  # Wb_sh [MAP]
        [-1.1375 * upper, -1.1375 * lower],
        [-1.004859455505 * upper, -1.004859455505 * lower],  # Wb_sv [MAP]
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
        [2.060317555315 * lower, 2.060317555315 * upper],  # Emax_lv0 [MAP]
        [1.279086612055 * lower, 1.279086612055 * upper],  # Emax_rv0 [MAP]
        [2.996981540095 * lower, 2.996981540095 * upper],  # fes_min [MAP]
        [0.431945385857 * lower, 0.431945385857 * upper],  # GEmax_lv [MAP]
        [0.282 * lower, 0.282 * upper],
        [2.47 * lower, 2.47 * upper],
        [1.94 * lower, 1.94 * upper],
        [2.47 * lower, 2.47 * upper],
        [0.695 * lower, 0.695 * upper],
        [-58.29 * upper, -58.29 * lower],
        [-74.21 * upper, -74.21 * lower],
        [-58.29 * upper, -58.29 * lower],
        [-237.648313155424 * upper, -237.648313155424 * lower],  # GV_sv [MAP]
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
        [11.1940122515 * lower, 11.1940122515 * upper],  # theta_svn [MAP]
        [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],
        [33.674334951783 * lower, 33.674334951783 * upper],  # PaCO2_n [MAP]
        [47.041027400563 * lower, 47.041027400563 * upper],  # f_ab_max [MAP]
        [2.52 * lower, 2.52 * upper],
        [10.038635495577 * lower, 10.038635495577 * upper],  # k_ab [MAP]
        [101.115183178836 * lower, 101.115183178836 * 1.05],  # P_n [MAP]
        [112 * 0.9, 112 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],
        [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],
        [-0.112650592783 * upper, -0.112650592783 * lower],  # GT_s [MAP]
        [0.100153101641 * lower, 0.100153101641 * upper],  # GT_v [MAP]
        [0.654880149316 * lower, 0.654880149316 * upper],  # T0 [MAP]
        [20.9 * lower, 20.9 * upper],
        [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower],
        [0.158362780305 * lower, 0.158362780305 * upper],  # Cvb_O2_n [MAP]
        [10 * lower, 10 * upper],
        [0.925 * lower, 0.925 * upper],
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
        [0.135937785465 * lower, 0.135937785465 * upper],  # Cvam_O2_n [MAP]
        [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],
        [0.36721777927 * lower, 0.36721777927 * upper],  # Io_met [MAP]
        [0.204798869807 * lower, 0.204798869807 * upper],  # kmet [MAP]
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
        [4.198293711872 * lower, 4.198293711872 * upper],  # Kv_mi [MAP]
        [1.309 * lower, 1.309 * upper],
        [2293.073313146301 * lower, 2293.073313146301 * upper],  # Kp_po [MAP]
        [2000 * lower, 2000 * upper],
        [2 * lower, 2 * upper],
        [8.115098186092 * lower, 8.115098186092 * upper],  # Kv_po [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [2.958475913209 * lower, 2.958475913209 * upper],  # Kv_tr [MAP]
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper],
        [393.134478510141 * lower, 393.134478510141 * upper],  # R_po [MAP]
        [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper],
        [0.001513621371 * lower, 0.001513621371 * upper],  # C_O2_param1 [MAP]
        [2.891202516105 * lower, 2.891202516105 * upper],  # C_O2_param2 [MAP]
        [0.0000303 * lower, 0.0000303 * upper],
        [104 * lower, 104 * upper],
        [311.969906535 * lower, 311.969906535 * upper],  # Vu_bv [MAP]
        [93.16 * lower, 93.16 * upper],
        [513.955410368231 * lower, 513.955410368231 * upper],  # Vu_jp [MAP]
        [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper],
        [114 * lower, 114 * upper],
        [27.301439733132 * lower, 27.301439733132 * upper],  # Vu_la [MAP]
        [17.35873806827 * lower, 17.35873806827 * upper],  # Vu_lv [MAP]
        [32.331282965822 * lower, 32.331282965822 * upper],  # Vu_ra [MAP]
        [43.648498038143 * lower, 43.648498038143 * upper],  # Vu_rv [MAP]
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
        [246.325150657968 * lower, 246.325150657968 * upper],  # Vu_amv0 [MAP]
        [646.987007375066 * lower, 646.987007375066 * upper],  # Vu_ev0 [MAP]
        [190.95 * lower, 190.95 * upper],
        [1262.699536885199 * lower, 1262.699536885199 * upper],  # Vu_sv0 [MAP]
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
        [0.012573867389 * lower, 0.012573867389 * upper],  # KE_lv [MAP]
        [0.00928516807 * lower, 0.00928516807 * upper],  # KE_rv [MAP]
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
        [4.300117727822 * lower, 4.300117727822 * upper],  # scale_param1 [MAP]
        [0.3 * lower, 0.3 * upper],
        [30.621117364076 * lower, 30.621117364076 * upper],  # scale_param4 [MAP]
        [0.04 * lower, 0.04 * upper],
        [80 * lower, 80 * upper],
        [0.038470374015 * lower, 0.038470374015 * upper],  # rise_time_atr [MAP]
        [0.351802683206 * lower, 0.351802683206 * upper],  # rise_time_ven [MAP]
        [0.504893619857 * 0.85, 0.504893619857 * 1.15],  # fall_time_ven [MAP]
        [0.941623732815 * 0.92, 0.941623732815 * 1.08],  # ahead1 [MAP]
        [0.0873 * lower, 0.0873 * upper],
        [1.101085481093 * 0.85, 1.101085481093 * 1.15],  # r [MAP]
        [1.356014731 * 0.85, 1.356014731 * 1.15],  # l [MAP]
        [132.141948246527 * lower, 132.141948246527 * upper],  # V_nominal [MAP]
        [50.289524292913 * lower, 50.289524292913 * upper],  # V_scale [MAP]
    ]
})

# Parameters sensitive in both rest and exercise.
subset_overlap = {
    'a2', 'ahead1', 'C2', 'C_jp', 'C_O2_param1', 'C_O2_param2', 'C_sv', 'Cvb_O2_n', 'E_rs', 'Emax_lv0',
    'Emax_rv0', 'fab_o', 'fall_time_ven', 'fes_o', 'fev_o', 'GEmax_lv', 'GT_s', 'GT_v', 'GV_sv', 'KE_la',
    'KE_lv', 'KE_ra', 'KE_rv', 'Kv_po', 'l', 'P0_la', 'P0_lv', 'P0_rv', 'P_n', 'PaCO2_n', 'r', 'R_po',
    'R_pp', 'R_rs', 'R_sa', 'rise_time_ven', 'Rvc_n', 's', 'scale_param4', 'T0', 'V0_dead', 'V_nominal',
    'V_scale', 'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0'}

# Rest-only posterior nuisance parameters sampled together with the overlap set.
subset_rest_only = {
    'Cvam_O2_n', 'Emax_la', 'Emax_ra', 'f_ab_max', 'fes_inf', 'fes_min', 'fev_inf', 'Io_met', 'Io_sv', 'K2', 'k_ab',
    'kcc_sv', 'kes', 'kmet', 'Kp_po', 'Kv_mi', 'Kv_tr', 'P0_ra', 'rise_time_atr', 'scale_param1', 'theta_svn',
    'Vu_amv0', 'Vu_bv', 'Wb_sh', 'Wb_sv'}

subset_overlap_and_rest = subset_overlap | subset_rest_only

# exercise
subset_exercise_only = {'C_pv', 'G_ap', 'GEmax_rv', 'GR_amp', 'GV_dead', 'KcCO2', 'KcMRV', 'MO2_bp', 'P_n_max',
                        'phi_max', 'R_amp0', 'tauMR', 'VA_rest', 'Wp_v', 'Yv_max'}

# MUST SORT SO ITS THE SAME ORDER
subset_overlap = [name for name in sp["names"] if name in subset_overlap]
subset_rest_only = [name for name in sp["names"] if name in subset_rest_only]
subset_overlap_and_rest = [name for name in sp["names"] if name in subset_overlap_and_rest]
subset_exercise_only = [name for name in sp["names"] if name in subset_exercise_only]
subset_exercise_and_overlap_set = set(subset_exercise_only) | set(subset_overlap)
subset_exercise_and_overlap = [
    name for name in sp["names"] if name in subset_exercise_and_overlap_set
]
subset_vars = subset_exercise_and_overlap_set

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
    "Max_LA_Pressure_Mitral_Opening", "LA_Pre_Atrial_Contraction_Volume", "RA_Pre_Atrial_Contraction_Volume", "LV_Pressure_Deriv",
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

# # Exercise (second is variance, not standard deviation)
observation = {"Heart Rate": (2.58, 0.12), "Systolic Pressure": (165, 529), "Diastolic Pressure": (76.4, 82.81),
"EDV": (145.5, 681.21), "ESV": (45.5, 75.69), "Max RV Volume": (139.4, 681.21), "Min RV Volume": (40.3, 112.36),
"Max RV Pressure": (29.5, 56.25), "Min RV Pressure": (9.9, 31.36), "Min RA Volume": (27.9, 25.0),
"Max RA Volume": (77.3, 342.25), "Max RA Pressure Atrial contraction": (12, 16),
"Max RA Pressure Tricuspid Opening": (11, 16), "Min LA Volume": (23.0, 94.09), "Max LA Volume": (66.3, 388.09),
"Max LA Pressure Atrial contraction": (19, 49), "Max LA Pressure Mitral Opening": (19, 64),
"LA Pre Atrial Contraction Volume": (33.8, 77.4), "RA Pre Atrial Contraction Volume": (40.3, 36.0),
"LV Pressure Deriv": (1750, 272484), "RV Pressure Deriv": (713, 12100), "Tidal Volume": (2.22, 0.4096),
"Minute Ventilation": (62.6, 320.41), "PaO2": (97.2, 36.0), "PaCO2": (38.4, 6.76)}

# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":

    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=Heart_Rate_emulator,
        observations=observation,
        # optional parameters
        threshold=3.5,
        random_seed=random_seed,
        # train_x=X,
        # train_y=Result,
        calibration_params=subset_exercise_and_overlap,
        overlap_params=subset_overlap_and_rest,
        exercise_only_params=subset_exercise_only,
        atrial_ratio_bounds=ATRIAL_RATIO_BOUNDS,
        atrial_ratio_min_probability=ATRIAL_RATIO_MIN_PROBABILITY,
        atrial_ratio_mc_samples=ATRIAL_RATIO_MC_SAMPLES,
        rest_overlap_source="posterior",
        rest_overlap_path=REST_POSTERIOR_RUN_DIR,
        rest_posterior_mass=REST_POSTERIOR_MASS,
        rest_posterior_region=REST_POSTERIOR_REGION,
        rest_overlap_sampling=REST_OVERLAP_SAMPLING,
    )

    # # --- PRE-WAVE: Train initial emulators from hybrid samples ---
    hmw.pre_wave_train_emulators(n_simulations=8192, refit_on_all_data=False)

    # hmw.rest_overlap_sampling = "cloud"
    size = 400000
    _ = hmw.run_waves(n_waves=4, n_simulations=6000, n_test_samples=size, refit_on_all_data=False, refit_emulator_on_last_wave=True, max_retries=15, resume_wave=False, use_phys_seed_wave0=False)

    # Get the last wave results
    test_parameters, impl_scores = hmw.wave_results[-1]
    nroy_points = hmw.get_nroy(impl_scores, test_parameters)

    # Get exact min/max bounds for the parameters from the NROY points
    params_post_hm = hmw.generate_param_bounds(
        nroy_x=nroy_points,
        param_names=sp["names"],
        buffer_ratio=0.0
    )

    np.save(f"NROY_Points_exercise_{percent}.npy", nroy_points)
    np.save(f"NROY_Params_exercise_{percent}.npy", params_post_hm)
    np.save(f"NROY_Implaus_exercise_{percent}.npy", impl_scores)
    np.save(f"test_param_exercise_{percent}.npy", test_parameters)

    print(len(hmw.wave_results)-1)
    # hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{size}_wave_{(len(hmw.wave_results)-1)}_{percent}.png")
