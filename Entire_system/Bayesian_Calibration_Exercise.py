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
    BASE_DIR, "MCMC_HPC/MCMC_Rest_20_05_05_1500_logspline_copula_prior"
)
REST_POSTERIOR_MASS = 0.95
REST_POSTERIOR_REGION = "hpd"
REST_OVERLAP_SAMPLING = "empirical"

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
        [0.032569754878 * lower, 0.032569754878 * upper],  # beta2 [MAP]
        [75.07418230369 * lower, 75.07418230369 * upper],  # C2 [MAP]
        [167.144972777718 * lower, 167.144972777718 * upper],  # K2 [MAP]
        [2.064005449204 * lower, 2.064005449204 * upper],  # a2 [MAP]
        [0.05591 * lower, 0.05591 * upper],
        [346000 * lower, 346000 * upper],
        [0.1698 * lower, 0.1698 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper],
        [0.2025 * lower, 0.2025 * upper],
        [0.00000000472 * lower, 0.00000000472 * upper],
        [0.137907482418 * lower, 0.137907482418 * upper],  # V0_dead [MAP]
        [0.0673 * lower, 0.0673 * upper],
        [18.865464214162 * lower, 18.865464214162 * upper],  # E_rs [MAP]
        [3.425655660867 * lower, 3.425655660867 * upper],  # R_rs [MAP]
        [3.180597455363 * lower, 3.180597455363 * upper],  # C_jp [MAP]
        [0.28 * lower, 0.28 * upper],
        [0.00022 * lower, 0.00022 * upper],
        [0.051839943263 * lower, 0.051839943263 * upper],  # R_sa [MAP]
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper],
        [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper],
        [53.340709006166 * lower, 53.340709006166 * upper],  # C_sv [MAP]
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
        [0.021862201529 * lower, 0.021862201529 * upper],  # Rvc_n [MAP]
        [0.76 * lower, 0.76 * upper],
        [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper],
        [0.019828017421 * lower, 0.019828017421 * upper],  # R_pa [MAP]
        [0.076316866291 * lower, 0.076316866291 * upper],  # R_pp [MAP]
        [0.0056 * lower, 0.0056 * upper],
        [0.389267762687 * lower, 0.389267762687 * upper],  # Emax_la [MAP]
        [0.504855552204 * lower, 0.504855552204 * upper],  # P0_la [MAP]
        [0.382347744148 * lower, 0.382347744148 * upper],  # Emax_ra [MAP]
        [0.387423247065 * lower, 0.387423247065 * upper],  # P0_ra [MAP]
        [0.056375357675 * lower, 0.056375357675 * upper],  # KE_la [MAP]
        [0.043179183322 * lower, 0.043179183322 * upper],  # KE_ra [MAP]
        [1.50056676174 * lower, 1.50056676174 * upper],  # P0_lv [MAP]
        [1.338495841285 * lower, 1.338495841285 * upper],  # P0_rv [MAP]
        [0.04 * lower, 0.04 * upper],
        [21.551737299718 * lower, 21.551737299718 * upper],  # fab_o [MAP]
        [15.971069930971 * lower, 15.971069930971 * upper],  # fes_o [MAP]
        [2.06580197715 * lower, 2.06580197715 * upper],  # fes_inf [MAP]
        [80 * lower, 80 * upper],
        [3.627818402509 * lower, 3.627818402509 * upper],  # fev_o [MAP]
        [5.421471609934 * lower, 5.421471609934 * upper],  # fev_inf [MAP]
        [0.065341862051 * lower, 0.065341862051 * upper],  # kes [MAP]
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper],
        [0.395105159176 * lower, 0.395105159176 * upper],  # Io_sv [MAP]
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper],
        [0.099698834039 * lower, 0.099698834039 * upper],  # kcc_sv [MAP]
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
        [-1.995254021212 * upper, -1.995254021212 * lower],  # Wb_sh [MAP]
        [-1.1375 * upper, -1.1375 * lower],
        [-0.981421145881 * upper, -0.981421145881 * lower],  # Wb_sv [MAP]
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
        [2.079377662633 * lower, 2.079377662633 * upper],  # Emax_lv0 [MAP]
        [1.207199670036 * lower, 1.207199670036 * upper],  # Emax_rv0 [MAP]
        [2.602033266982 * lower, 2.602033266982 * upper],  # fes_min [MAP]
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
        [11.46674882662 * lower, 11.46674882662 * upper],  # theta_svn [MAP]
        [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],
        [38.771561996394 * lower, 38.771561996394 * upper],  # PaCO2_n [MAP]
        [41.210583308438 * lower, 41.210583308438 * upper],  # f_ab_max [MAP]
        [2.52 * lower, 2.52 * upper],
        [10.241696157506 * lower, 10.241696157506 * upper],  # k_ab [MAP]
        [92.268364155564 * lower, 92.268364155564 * 1.05],  # P_n [MAP]
        [112 * 0.9, 112 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],
        [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],
        [-0.112245382259 * upper, -0.112245382259 * lower],  # GT_s [MAP]
        [0.078282298884 * lower, 0.078282298884 * upper],  # GT_v [MAP]
        [0.50742169344 * lower, 0.50742169344 * upper],  # T0 [MAP]
        [20.9 * lower, 20.9 * upper],
        [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower],
        [0.14 * lower, 0.14 * upper],
        [10 * lower, 10 * upper],
        [1.043348622918 * lower, 1.043348622918 * upper],  # MO2_bp [MAP]
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
        [0.133448870752 * lower, 0.133448870752 * upper],  # Cvam_O2_n [MAP]
        [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],
        [0.399026272887 * lower, 0.399026272887 * upper],  # Io_met [MAP]
        [0.156823424904 * lower, 0.156823424904 * upper],  # kmet [MAP]
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
        [3.034085813966 * lower, 3.034085813966 * upper],  # Kv_mi [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [2000 * lower, 2000 * upper],
        [2 * lower, 2 * upper],
        [6.059819855009 * lower, 6.059819855009 * upper],  # Kv_po [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [3.003002129452 * lower, 3.003002129452 * upper],  # Kv_tr [MAP]
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper],
        [350 * lower, 350 * upper],
        [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper],
        [0.001510640109 * lower, 0.001510640109 * upper],  # C_O2_param1 [MAP]
        [2.6 * lower, 2.6 * upper],
        [0.0000303 * lower, 0.0000303 * upper],
        [104 * lower, 104 * upper],
        [242.392116675323 * lower, 242.392116675323 * upper],  # Vu_bv [MAP]
        [93.16 * lower, 93.16 * upper],
        [510.877098670281 * lower, 510.877098670281 * upper],  # Vu_jp [MAP]
        [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper],
        [114 * lower, 114 * upper],
        [26.609335699832 * lower, 26.609335699832 * upper],  # Vu_la [MAP]
        [18.125484078127 * lower, 18.125484078127 * upper],  # Vu_lv [MAP]
        [34.114134017235 * lower, 34.114134017235 * upper],  # Vu_ra [MAP]
        [43.687753626913 * lower, 43.687753626913 * upper],  # Vu_rv [MAP]
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
        [246.112339205338 * lower, 246.112339205338 * upper],  # Vu_amv0 [MAP]
        [546.87106973063 * lower, 546.87106973063 * upper],  # Vu_ev0 [MAP]
        [190.95 * lower, 190.95 * upper],
        [1202.05160085603 * lower, 1202.05160085603 * upper],  # Vu_sv0 [MAP]
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
        [0.012678051291 * lower, 0.012678051291 * upper],  # KE_lv [MAP]
        [0.012526831858 * lower, 0.012526831858 * upper],  # KE_rv [MAP]
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
        [0.044588980624 * lower, 0.044588980624 * upper],  # rise_time_atr [MAP]
        [0.338070928104 * lower, 0.338070928104 * upper],  # rise_time_ven [MAP]
        [0.498317998504 * 0.85, 0.498317998504 * 1.15],  # fall_time_ven [MAP]
        [0.903728113538 * 0.92, 0.903728113538 * 1.08],  # ahead1 [MAP]
        [0.0873 * lower, 0.0873 * upper],
        [1.175949527879 * 0.85, 1.175949527879 * 1.15],  # r [MAP]
        [1.312192049544 * 0.85, 1.312192049544 * 1.15],  # l [MAP]
        [136.066107675331 * lower, 136.066107675331 * upper],  # V_nominal [MAP]
        [43.442083472162 * lower, 43.442083472162 * upper],  # V_scale [MAP]
    ]
})

# Exercise
subset_vars = {'a2', 'ahead1', 'C2', 'C_O2_param1', 'C_pv', 'C_sv', 'E_rs', 'Emax_lv0', 'Emax_ra', 'Emax_rv0',
               'fall_time_ven', 'fes_o', 'fev_o', 'G_ap', 'GEmax_lv', 'GEmax_rv', 'GR_amp', 'GT_s', 'GT_v', 'GV_dead',
               'GV_sv', 'Io_sh', 'KcCO2', 'KcMRV', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'l', 'P0_la', 'P0_lv', 'P0_rv',
               'P_n_max', 'PaCO2_n', 'phi_max', 'r', 'R_amp0', 'R_pa', 'R_po', 'R_pp', 'R_rs', 'R_sa', 'rise_time_ven',
               'Rvc_n', 'T0', 'tauMR', 'V0_dead', 'V_nominal', 'V_scale', 'VA_rest', 'Vu_ev0', 'Vu_jp', 'Vu_la',
               'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0', 'Wp_v', 'Yv_max'}

subset_overlap = {'a2', 'ahead1', 'C2', 'C_O2_param1', 'C_sv', 'E_rs', 'Emax_lv0', 'Emax_ra', 'Emax_rv0',
                  'fall_time_ven', 'fes_o', 'fev_o', 'GT_s', 'GT_v', 'KE_la', 'KE_lv', 'KE_ra', 'KE_rv', 'l', 'P0_la',
                  'P0_lv', 'P0_rv', 'PaCO2_n', 'r', 'R_pa', 'R_pp', 'R_rs', 'R_sa', 'rise_time_ven', 'Rvc_n', 'T0',
                  'V0_dead', 'V_nominal', 'V_scale', 'Vu_ev0', 'Vu_jp', 'Vu_la', 'Vu_lv', 'Vu_ra', 'Vu_rv', 'Vu_sv0'}


subset_exercise_only = {'C_pv', 'G_ap', 'GEmax_lv', 'GEmax_rv', 'GR_amp', 'GV_dead', 'GV_sv', 'Io_sh', 'KcCO2', 'KcMRV',
                        'P_n_max', 'phi_max', 'R_amp0', 'R_po', 'tauMR', 'VA_rest', 'Wp_v', 'Yv_max'}


# MUST SORT SO ITS THE SAME ORDER
subset_vars = [name for name in sp["names"] if name in subset_vars]
subset_overlap = [name for name in sp["names"] if name in subset_overlap]
subset_exercise_only = [name for name in sp["names"] if name in subset_exercise_only]

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

# # Exercise (second is variance, not standard deviation)
observation = {"Heart Rate": (2.58, 0.12), "Systolic Pressure": (165, 529), "Diastolic Pressure": (76.4, 82.81),
"EDV": (145.5, 681.21), "ESV": (45.5, 75.69), "Max RV Volume": (139.4, 681.21), "Min RV Volume": (40.3, 112.36),
"Max RV Pressure": (29.5, 56.25), "Min RV Pressure": (9.9, 31.36), "Min RA Volume": (27.9, 25.0),
"Max RA Volume": (77.3, 342.25), "Max RA Pressure Atrial contraction": (12, 16),
"Max RA Pressure Tricuspid Opening": (11, 16), "Min LA Volume": (23.0, 94.09), "Max LA Volume": (66.3, 388.09),
"Max LA Pressure Atrial contraction": (19, 49), "Max LA Pressure Mitral Opening": (19, 64),
"LA Contraction Volume diff": (33.8, 81.0), "RA Contraction Volume diff": (40.3, 36.0),
"LV Pressure Deriv": (1750, 272484), "RV Pressure Deriv": (713, 12100), "Tidal Volume": (2220, 409600),
"Minute Ventilation": (62.6, 320.41), "PaO2": (97.2, 36.0), "PaCO2": (38.4, 6.76)}

# ----------------------------
# BAYESIAN CALIBRATION
# ----------------------------
if __name__ == "__main__":

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


    hmw = HistoryMatchingWorkflow(
        simulator=Simulator,
        result=Heart_Rate_emulator,
        observations=observation,
        # optional parameters
        threshold=3,
        random_seed=random_seed,
        # train_x=X,
        # train_y=Result,
        calibration_params=subset_vars,
        overlap_params=subset_overlap,
        exercise_only_params=subset_exercise_only,
        rest_overlap_source="posterior",
        rest_overlap_path=REST_POSTERIOR_RUN_DIR,
        rest_posterior_mass=REST_POSTERIOR_MASS,
        rest_posterior_region=REST_POSTERIOR_REGION,
        rest_overlap_sampling=REST_OVERLAP_SAMPLING,
    )

    # --- PRE-WAVE: Train initial emulators from hybrid samples ---
    hmw.pre_wave_train_emulators(n_simulations=4096, refit_on_all_data=False)

    hmw.rest_overlap_sampling = "cloud"

    size = 200000
    _ = hmw.run_waves(n_waves=3, n_simulations=2048, n_test_samples=size, refit_on_all_data=False, refit_emulator_on_last_wave=True, max_retries=15, resume_wave=False)

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
    hmw.plot_wave((len(hmw.wave_results)-1), fname=f"{size}_wave_{(len(hmw.wave_results)-1)}_{percent}.png")