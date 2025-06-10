lower = 0.5
upper = 1.5

# parameters = {
# # gas variables
#   "a2": (1.219 * upper, 1.219 * lower),
#   "alpha1": (0.03198 * upper, 0.03198 * lower),
#   "alpha2": (0.05591 * upper, 0.05591 * lower),
#   "beta1": (0.008275 * upper, 0.008275 * lower),
#   "beta2": (0.03255 * upper, 0.03255 * lower),
#   "C2": (40 * upper, 40 * lower),
#   # "Fi_CO2": (0.0421 * upper, 0.0421 * lower),
#   # "Fi_O2": (21.0379 * upper, 21.0379 * lower),
#   "K1": (13 * upper, 13 * lower),
#   "K2": (25 * upper, 25 * lower),
#   # "LCTV": (0.588 * upper, 0.588 * lower),
#   # "PACO2_Delay_IC": (40.4448 * upper, 40.4448 * lower),
#   # "PAO2_Delay_IC": (103.1223 * upper, 103.1223 * lower),
#   # "P_atm": (760 * upper, 760 * lower),
#   # "P_ws": (47 * upper, 47 * lower),
#   # "T1": (1 * upper, 1 * lower),
#   # "T2": (2 * upper, 2 * lower),
#   # "VL_CO2": (3 * upper, 3 * lower),
#   # "VL_O2": (2.5 * upper, 2.5 * lower),
#   # "Z": (0.0227 * upper, 0.0227 * lower),
#   "dc": (0.015 * upper, 0.015 * lower),
#   "KCCO2": (346000 * upper, 346000 * lower),
#   # "KCSFCO2": (320 * upper, 320 * lower),
#   "MRBCO2": (0.0009 * upper, 0.0009 * lower),
#   "Ta": (8.8 * upper, 8.8 * lower),
#   # "VB": (0.9 * upper, 0.9 * lower),
#   # "MRTCO2_basal": (0.00243333 * upper, 0.00243333 * lower),
#   # "MRTO2_basal": (0.00325708 * upper, 0.00325708 * lower),
#   # "tauMR": (50 * upper, 50 * lower),
#   # "VTCO2": (0.25 * upper, 0.25 * lower),
#   # "VTO2": (0.25 * upper, 0.25 * lower),
#   # "MRCO2": (0.00243333 * upper, 0.00243333 * lower),
#   # "MRO2": (0.00325708 * upper, 0.00325708 * lower),
#   # "tau_MRV": (50 * upper, 50 * lower),
#   "s": (0.04 * upper, 0.04 * lower),
#
#
#     # resp contr
#     "GV_dead": (0.1698 * upper, 0.1698 * lower),
#     "Kbg": (17.4, 17.4  * lower),
#     "KcCO2": (0.2332 * upper, 0.2332),
#     "KcMRV": (1 * upper, 1),
#     "KpCO2": (0.2025 * upper, 0.2025),
#     "KpO2": (4.72e-09 * upper, 4.72e-09),
#     "V0_dead": (0.1587 * upper, 0.1587 * lower),
#     "VA_rest": (0.067 * upper, 0.067 * lower),
#     # "lambda1": (0.4 * upper, 0.4 * lower),
#     # "lambda2": (0.05 * upper, 0.05 * lower),
#     # "n": (1.101 * upper, 1.101 * lower),
#     "Pmax": (50 * upper, 50 * lower),
#     "Pmax_dot": (1000 * upper, 1000 * lower),
#     "E_rs": (21.9 * upper, 21.9 * lower),
#     "R_rs": (3.02 * upper, 3.02 * lower),
#     # "P_ao": (0 * upper, 0 * lower)
#
#
#     # cardio
#     "C_sa": (0.28 * lower, 0.28 * upper),
#     "L_sa": (0.00022 * lower, 0.00022 * upper),
#     "R_sa": (0.06 * lower, 0.06 * upper),
#     # "Vu_sa": (0.0 * lower, 10.0 * upper),
#     "C_amp": (0.315 * lower, 0.315 * upper),
#     "C_amv": (9.4 * lower, 9.4 * upper),
#     "C_bp": (0.358 * lower, 0.358 * upper),
#     "C_bv": (10.71 * lower, 10.71 * upper),
#     "C_ep": (0.668 * lower, 0.668 * upper),
#     "C_ev": (20 * lower, 20 * upper),
#     "C_hp": (0.119 * lower, 0.119 * upper),
#     "C_hv": (3.57 * lower, 3.57 * upper),
#     "C_rmp": (0.21 * lower, 0.21 * upper),
#     "C_rmv": (6.28 * lower, 6.28 * upper),
#     "C_sp": (2.05 * lower, 2.05 * upper),
#     "C_sv": (61.11 * lower, 61.11 * upper),
#     # "kr_am": (24.17 * lower, 24.17 * upper),
#     # "P_0": (3.93 * lower, 3.93 * upper),
#     "R_amv_n": (0.0833 * lower, 0.0833 * upper),
#     "R_bv_n": (0.075 * lower, 0.075 * upper),
#     "R_ev_n": (0.04 * lower, 0.04 * upper),
#     "R_hv_n": (0.224 * lower, 0.224 * upper),
#     "R_rmv_n": (0.125 * lower, 0.125 * upper),
#     "R_sv_n": (0.038 * lower, 0.038 * upper),
#     # "V_tot": (5027.6 * lower, 5027.6 * upper),
#     # "Vu_amp": (60.22 * lower, 60.22 * upper),
#     # "Vu_bp": (68.42 * lower, 68.42 * upper),
#     # "Vu_bv": (279.49 * lower, 279.49 * upper),
#     # "Vu_ep": (127.72 * lower, 127.72 * upper),
#     # "Vu_hp": (23 * lower, 23 * upper),
#     # "Vu_hv": (93.16 * lower, 93.16 * upper),
#     # "Vu_rmp": (40.1 * lower, 40.1 * upper),
#     # "Vu_sp": (260.3 * lower, 260.3 * upper),
#     "D1": (0.3855 * lower, 0.3855 * upper),
#     "D2": (-5 * upper, -5 * lower),
#     "K1_vc": (0.15 * lower, 0.15 * upper),
#     "K2_vc": (0.4 * lower, 0.4 * upper),
#     "Kr_vc": (0.001 * lower, 0.001 * upper),
#     "Rvc_n": (0.025 * lower, 0.025 * upper),
#     # "Vu_vc": (123 * lower, 123 * upper),
#     # "Vvc_max": (350 * lower, 350 * upper),
#     # "Vvc_min": (50 * lower, 50 * upper),
#     "C_pa": (0.76 * lower, 0.76 * upper),
#     "C_pp": (5.8 * lower, 5.8 * upper),
#     "C_pv": (25.37 * lower, 25.37 * upper),
#     "L_pa": (0.00018 * lower, 0.00018 * upper),
#     "R_pa": (0.023 * lower, 0.023 * upper),
#     "R_pp": (0.0894 * lower, 0.0894 * upper),
#     "R_pv": (0.0056 * lower, 0.0056 * upper),
#     # "Vu_pa": (0.0 * lower, 10.0 * upper),
#     # "Vu_pp": (116.6775 * lower, 116.6775 * upper),
#     # "Vu_pv": (114 * lower, 114 * upper),
#     # "KE_lv": (0.014 * lower, 0.014 * upper),
#     # "KE_rv": (0.011 * lower, 0.011 * upper),
#     "Emax_la": (0.45 * lower, 0.45 * upper),
#     "P0_la": (0.45 * lower, 0.45 * upper),
#     # "KE_la": (0.05 * lower, 0.05 * upper),
#     "Emax_ra": (0.45 * lower, 0.45 * upper),
#     "P0_ra": (0.45 * lower, 0.45 * upper),
#     # "KE_ra": (0.05 * lower, 0.05 * upper),
#     "P0_lv": (1.5 * lower, 1.5 * upper),
#     "P0_rv": (1.5 * lower, 1.5 * upper),
#     # "Vu_la": (24 * lower, 24 * upper),
#     # "Vu_lv": (15.908 * lower, 15.908 * upper),
#     # "Vu_ra": (24 * lower, 24 * upper),
#     # "Vu_rv": (38.703 * lower, 38.703 * upper),
#     "g_abd": (3.39 * lower, 3.39 * upper),
#     "g_thor": (6.8 * lower, 6.8 * upper),
#     "P_abdmax_n": (-1 * upper, 0 * lower),
#     "P_abdmin_n": (-2.5 * upper, -2.5 * lower),
#     "P_thormax_n": (-1 * upper, 0.0 * lower),
#     "P_thormin_n": (-3 * upper, 0.0 * lower),
#     "VT_n": (0.45 * lower, 0.45 * upper),
#     "A_im": (50 * lower, 50 * upper),
#     "Tc": (0.75 * lower, 0.75),
#     "T_im": (1, 1 * upper),
#
#     # cardio control
#
#   "fab_o": (25 * upper, 25 * lower),
#   "fes_o": (16.11 * upper, 16.11 * lower),
#   "fes_inf": (2.1 * upper, 2.1 * lower),
#   "fes_max": (80 * upper, 80 * lower),
#   "fev_o": (3.2 * upper, 3.2 * lower),
#   "fev_inf": (6.3 * upper, 6.3 * lower),
#   "kes": (0.0675 * upper, 0.0675 * lower),
#   "kev": (7.06 * upper, 7.06 * lower),
#   # "Io_sh": (0.658 * upper, 0.658 * lower),
#   # "Io_sp": (0.65 * upper, 0.65 * lower),
#   # "Io_sv": (0.45 * upper, 0.45 * lower),
#   # "Io_v": (0.126 * upper, 0.126 * lower),
#   "kcc_sh": (0.114 * upper, 0.114 * lower),
#   "kcc_sp": (0.13 * upper, 0.13 * lower),
#   "kcc_sv": (0.09 * upper, 0.09 * lower),
#   "kcc_v": (0.0162 * upper, 0.0162 * lower),
#   "Ysh_max": (9 * upper, 9 * lower),
#   "Ysh_min": (-0.0283 * upper, -0.0283 * lower),
#   "Ysp_max": (5.5 * upper, 5.5 * lower),
#   "Ysp_min": (-0.037 * upper, -0.037 * lower),
#   "Ysv_max": (64.9 * upper, 64.9 * lower),
#   "Ysv_min": (-0.028 * upper, -0.028 * lower),
#   "Yv_max": (1.9 * upper, 1.9 * lower),
#   "Yv_min": (-0.0008 * upper, -0.0008 * lower),
#   "theta_v": (-0.68 * upper, -0.68 * lower),
#   "Wb_sh": (-1.75 * upper, -1.75 * lower),
#   "Wb_sp": (-1.1375 * upper, -1.1375 * lower),
#   "Wb_sv": (-1.1375 * upper, -1.1375 * lower),
#   "Wc_sh": (1 * upper, 1 * lower),
#   "Wc_sp": (1.716 * upper, 1.716 * lower),
#   "Wc_sv": (1.716 * upper, 1.716 * lower),
#   "Wc_v": (0.2 * upper, 0.2 * lower),
#   "Wp_sh": (0.1 * upper, 0 * lower),
#   "Wp_sp": (-0.3997 * upper, -0.3997 * lower),
#   "Wp_sv": (-0.3997 * upper, -0.3997 * lower),
#   "Wp_v": (-0.103 * upper, -0.103 * lower),
#   "Wt_sh": (0.4 * upper, 0.4 * lower),
#   "Wt_sp": (0.4 * upper, 0.4 * lower),
#   "Wt_sv": (0.4 * upper, 0.4 * lower),
#   "Wt_v": (0.4 * upper, 0.4 * lower),
#   "Emax_lv0": (2.392 * upper, 2.392 * lower),
#   "Emax_rv0": (1.412 * upper, 1.412 * lower),
#   "fes_min": (2.66 * upper, 2.66 * lower),
#   "GEmax_lv": (0.475 * upper, 0.475 * lower),
#   "GEmax_rv": (0.282 * upper, 0.282 * lower),
#   "GR_amp": (2.47 * upper, 2.47 * lower),
#   "GR_ep": (1.94 * upper, 1.94 * lower),
#   "GR_rmp": (2.47 * upper, 2.47 * lower),
#   "GR_sp": (0.695 * upper, 0.695 * lower),
#   "GV_amv": (-58.29 * upper, -58.29 * lower),
#   "GV_ev": (-74.21 * upper, -74.21 * lower),
#   "GV_rmv": (-58.29 * upper, -58.29 * lower),
#   "GV_sv": (-265.4 * upper, -265.4 * lower),
#   "R_amp0": (3.51 * upper, 3.51 * lower),
#   "R_ep0": (1.655 * upper, 1.655 * lower),
#   "R_rmp0": (5.27 * upper, 5.27 * lower),
#   "R_sp0": (2.49 * upper, 2.49 * lower),
#   # "tau_Emax_lv": (8 * upper, 8 * lower),
#   # "tau_Emax_rv": (8 * upper, 8 * lower),
#   # "tau_Ramp": (2 * upper, 2 * lower),
#   # "tau_Rep": (2 * upper, 2 * lower),
#   # "tau_Rrmp": (2 * upper, 2 * lower),
#   # "tau_Rsp": (2 * upper, 2 * lower),
#   # "tau_Vamv": (20 * upper, 20 * lower),
#   # "tau_Vev": (20 * upper, 20 * lower),
#   # "tau_Vrmv": (20 * upper, 20 * lower),
#   # "tau_Vsv": (20 * upper, 20 * lower),
#   "Vu_amv0": (286.4 * upper, 286.4 * lower),
#   "Vu_ev0": (607.8 * upper, 607.8 * lower),
#   "Vu_rmv0": (190.95 * upper, 190.95 * lower),
#   "Vu_sv0": (1361.6 * upper, 1361.6 * lower),
#   "AT": ((1/60) * upper, (1/60) * lower),
#   "g_ccsh": (1 * upper, 1 * lower),
#   "g_ccsp": (1.5 * upper, 1.5 * lower),
#   "g_ccsv": (0.1 * upper, 0 * lower),
#   "kisc_sh": (6 * upper, 6 * lower),
#   "kisc_sp": (2 * upper, 2 * lower),
#   "kisc_sv": (2 * upper, 2 * lower),
#   "PO2_sh": (45 * upper, 45 * lower),
#   "PO2_sp": (30 * upper, 30 * lower),
#   "PO2_sv": (30 * upper, 30 * lower),
#   # "tau_cc": (20 * upper, 20 * lower),
#   # "tau_isc": (30 * upper, 30 * lower),
#   "theta_shn": (3.6 * upper, 3.6 * lower),
#   "theta_spn": (13.32 * upper, 13.32 * lower),
#   "theta_svn": (13.32 * upper, 13.32 * lower),
#   "x_sh": (53 * upper, 53 * lower),
#   "x_sp": (6 * upper, 6 * lower),
#   "x_sv": (6 * upper, 6 * lower),
#   "PaCO2_n": (40 * upper, 40 * lower),
#   "f_ab_max": (47.78 * upper, 47.78 * lower),
#   "f_ab_min": (2.52 * upper, 2.52 * lower),
#   "k_ab": (11.76 * upper, 11.76 * lower),
#   "P_n": (92 * upper, 92 * lower),
#   # "tau_p": (2.076 * upper, 2.076 * lower),
#   # "tau_z": (6.37 * upper, 6.37 * lower),
#   "f_acCO2_n": (1.4 * upper, 1.4 * lower),
#   "f_ac_max": (12.3 * upper, 12.3 * lower),
#   "f_ac_min": (0.835 * upper, 0.835 * lower),
#   "k_ac": (29.27 * upper, 29.27 * lower),
#   "K_H": (3 * upper, 3 * lower),
#   "PaO2_ac_n": (45 * upper, 45 * lower),
#   # "tau_ac": (2 * upper, 2 * lower),
#   "G_ap": (11.76 * upper, 11.76 * lower),
#   # "tau_ap": (2 * upper, 2 * lower),
#   "GT_s": (-0.13 * upper, -0.13 * lower),
#   "GT_v": (0.09 * upper, 0.09 * lower),
#   "T0": (0.58 * upper, 0.58 * lower),
#   # "tau_Ts": (2 * upper, 2 * lower),
#   # "tau_Tv": (1.5 * upper, 1.5 * lower),
#   "A": (20.9 * upper, 20.9 * lower),
#   "B": (92.8 * upper, 92.8 * lower),
#   "C": (10570 * upper, 10570 * lower),
#   "D": (-5.251 * upper, -5.251 * lower),
#   "Cvb_O2_n": (0.14 * upper, 0.14 * lower),
#   "gb_O2": (10 * upper, 10 * lower),
#   "MO2_bp": (0.925 * upper, 0.925 * lower),
#   "R_bpn": (6.57 * upper, 6.57 * lower),
#   # "tau_CO2": (20 * upper, 20 * lower),
#   # "tau_O2": (10 * upper, 10 * lower),
#
#   # third
#   "Cvh_O2_n": (0.11 * upper, 0.11 * lower),
#   "Cvrm_O2_n": (0.155 * upper, 0.155 * lower),
#   "gh_O2": (35 * upper, 35 * lower),
#   "grm_O2": (30 * upper, 30 * lower),
#   "Kh_CO2": (11.11 * upper, 11.11 * lower),
#   "Krm_CO2": (142.8 * upper, 142.8 * lower),
#   "MO2_hpn": (0.4 * upper, 0.4 * lower),
#   "MO2_rmp": (0.86 * upper, 0.86 * lower),
#   "R_hpn": (19.71 * upper, 19.71 * lower),
#   # "tau_w": (5 * upper, 5 * lower),
#   "W_hn": (12660 * upper, 12660 * lower),
#   "Cvam_O2_n": (0.1555 * upper, 0.1555 * lower),
#   "gam_O2": (30 * upper, 30 * lower),
#   "gM": (40 * upper, 40 * lower),
#   # "Io_met": (0.4266 * upper, 0.4266 * lower),
#   "kmet": (0.18 * upper, 0.18 * lower),
#   "MO2_ampn": (0.516 * upper, 0.516 * lower),
#   "phi_max": (20 * upper, 20 * lower),
#   "phi_min": (-1.87 * upper, -1.87 * lower),
#   # "tau_M": (40 * upper, 40 * lower),
#   # "tau_met": (10 * upper, 10 * lower),
# }

    # # controller
    # g_ccsh = params["gccsh"]
    # g_ccsp = params["gccsp"]
    # g_ccsv = params["gccsv"]
    # kisc_sh = params["kisc_sh"]
    # kisc_sp = params["kisc_sp"]
    # kisc_sv = params["kisc_sv"]
    # PO2_sh = params["PO2_sh"]
    # PO2_sp = params["PO2_sp"]
    # PO2_sv = params["PO2_sv"]
    # tau_cc = params["tau_cc"]
    # tau_isc = params["tau_isc"]
    # theta_shn = params["theta_shn"]
    # theta_spn = params["theta_spn"]
    # theta_svn = params["theta_svn"]
    # x_sh = params["x_sh"]
    # x_sp = params["x_sp"]
    # x_sv = params["x_sv"]
    #
    # f_ab_max = params["f_ab_max"]
    # f_ab_min = params["f_ab_min"]
    # k_ab = params["k_ab"]
    # P_n = params["P_n"]
    # tau_p = params["tau_p"]
    # tau_z = params["tau_z"]
    #
    # f_acCO2_n = params["f_acCO2_n"]
    # f_ac_max = params["f_ac_max"]
    # f_ac_min = params["f_ac_min"]
    # k_ac = params["k_ac"]
    # K_H = params["K_H"]
    # PaO2_ac_n = params["PaO2_ac_n"]
    # PaCO2_n = params["PaCO2_n"]
    # tau_ac = params["tau_ac"]
    #
    # G_ap = params["G_ap"]
    # tau_ap = params["tau_ap"]



Parameters = {
  "a2": 1.219,
  "alpha1": 0.03198,
  "alpha2": 0.05591,
  "beta1": 0.008275,
  "beta2": 0.03255,
  "K1": 13,
  "K2": 25,
  "dc": 0.015,
  "KCCO2": 346000,
  "MRBCO2": 0.0009,
  "Ta": 8.8,
  "s": 0.04,
  "GV_dead": 0.1698,
  "Kbg": 17.4,
  "KcCO2": 0.2332,
  "KcMRV": 1,
  "KpCO2": 0.2025,
  "KpO2": 4.72e-09,
  "V0_dead": 0.1587,
  "VA_rest": 0.067,
  "Pmax": 50,
  "Pmax_dot": 1000,
  "E_rs": 21.9,
  "R_rs": 3.02,
  "C_sa": 0.28,
  "L_sa": 0.00022,
  "R_sa": 0.06,
  "C_amp": 0.315,
  "C_amv": 9.4,
  "C_bp": 0.358,
  "C_bv": 10.71,
  "C_ep": 0.668,
  "C_ev": 20,
  "C_hp": 0.119,
  "C_hv": 3.57,
  "C_rmp": 0.21,
  "C_rmv": 6.28,
  "C_sp": 2.05,
  "C_sv": 61.11,
  "R_amv_n": 0.0833,
  "R_bv_n": 0.075,
  "R_ev_n": 0.04,
  "R_hv_n": 0.224,
  "R_rmv_n": 0.125,
  "R_sv_n": 0.038,
  "D1": 0.3855,
  "D2": -5,
  "K1_vc": 0.15,
  "K2_vc": 0.4,
  "Kr_vc": 0.001,
  "Rvc_n": 0.025,
  "C_pa": 0.76,
  "C_pp": 5.8,
  "C_pv": 25.37,
  "L_pa": 0.00018,
  "R_pa": 0.023,
  "R_pp": 0.0894,
  "R_pv": 0.0056,
  "Emax_la": 0.45,
  "P0_la": 0.45,
  "Emax_ra": 0.45,
  "P0_ra": 0.45,
  "P0_lv": 1.5,
  "P0_rv": 1.5,
  "g_abd": 3.39,
  "g_thor": 6.8,
  "P_abdmax_n": -1,
  "P_abdmin_n": -2.5,
  "P_thormax_n": -1,
  "P_thormin_n": -3,
  "VT_n": 0.45,
  "A_im": 50,
  "Tc": 0.75,
  "T_im": 1,
  "fab_o": 25,
  "fes_o": 16.11,
  "fes_inf": 2.1,
  "fes_max": 80,
  "fev_o": 3.2,
  "fev_inf": 6.3,
  "kes": 0.0675,
  "kev": 7.06,
  "kcc_sh": 0.114,
  "kcc_sp": 0.13,
  "kcc_sv": 0.09,
  "kcc_v": 0.0162,
  "Ysh_max": 9,
  "Ysh_min": -0.0283,
  "Ysp_max": 5.5,
  "Ysp_min": -0.037,
  "Ysv_max": 64.9,
  "Ysv_min": -0.028,
  "Yv_max": 1.9,
  "Yv_min": -0.0008,
  "theta_v": -0.68,
  "Wb_sh": -1.75,
  "Wb_sp": -1.1375,
  "Wb_sv": -1.1375,
  "Wc_sh": 1,
  "Wc_sp": 1.716,
  "Wc_sv": 1.716,
  "Wc_v": 0.2,
  "Wp_sh": 0.1,
  "Wp_sp": -0.3997,
  "Wp_sv": -0.3997,
  "Wp_v": -0.103,
  "Wt_sh": 0.4,
  "Wt_sp": 0.4,
  "Wt_sv": 0.4,
  "Wt_v": 0.4,
  "Emax_lv0": 2.392,
  "Emax_rv0": 1.412,
  "fes_min": 2.66,
  "GEmax_lv": 0.475,
  "GEmax_rv": 0.282,
  "GR_amp": 2.47,
  "GR_ep": 1.94,
  "GR_rmp": 2.47,
  "GR_sp": 0.695,
  "GV_amv": -58.29,
  "GV_ev": -74.21,
  "GV_rmv": -58.29,
  "GV_sv": -265.4,
  "R_amp0": 3.51,
  "R_ep0": 1.655,
  "R_rmp0": 5.27,
  "R_sp0": 2.49,
  "Vu_amv0": 286.4,
  "Vu_ev0": 607.8,
  "Vu_rmv0": 190.95,
  "Vu_sv0": 1361.6,
  "AT": (1/60),
  "g_ccsh": 1,
  "g_ccsp": 1.5,
  "g_ccsv": 0.1,
  "kisc_sh": 6,
  "kisc_sp": 2,
  "kisc_sv": 2,
  "PO2_sh": 45,
  "PO2_sp": 30,
  "PO2_sv": 30,
  "theta_shn": 3.6,
  "theta_spn": 13.32,
  "theta_svn": 13.32,
  "x_sh": 53,
  "x_sp": 6,
  "x_sv": 6,
  "PaCO2_n": 40,
  "f_ab_max": 47.78,
  "f_ab_min": 2.52,
  "k_ab": 11.76,
  "P_n": 92,
  "f_acCO2_n": 1.4,
  "f_ac_max": 12.3,
  "f_ac_min": 0.835,
  "k_ac": 29.27,
  "K_H": 3,
  "PaO2_ac_n": 45,
  "G_ap": 11.76,
  "GT_s": -0.13,
  "GT_v": 0.09,
  "T0": 0.58,
  "A": 20.9,
  "B": 92.8,
  "C": 10570,
  "D": -5.251,
  "Cvb_O2_n": 0.14,
  "gb_O2": 10,
  "MO2_bp": 0.925,
  "R_bpn": 6.57,
  "Cvh_O2_n": 0.11,
  "Cvrm_O2_n": 0.155,
  "gh_O2": 35,
  "grm_O2": 30,
  "Kh_CO2": 11.11,
  "Krm_CO2": 142.8,
  "MO2_hpn": 0.4,
  "MO2_rmp": 0.86,
  "R_hpn": 19.71,
  "W_hn": 12660,
  "Cvam_O2_n": 0.1555,
  "gam_O2": 30,
  "gM": 40,
  "kmet": 0.18,
  "MO2_ampn": 0.516,
  "phi_max": 20,
  "phi_min": -1.87,
  "C2": 40,


  "Vu_pa": 0.0,
  "Vu_pp": 116.6775,
  "Vu_pv": 114,
  "KE_lv": 0.014, # 0.014      # End-diastolic P-V relationship in left ventricle # adjusted (changes a lot depending on whether it is 0.06 or 0.05)
  "KE_rv": 0.011, # 0.011
  "Vu_la": 4,           # Left atrial unstressed volume # adjusted to the shi paper Increasing these heart unstressed volumes decreases the maximum flow and pressures
  "Vu_lv": 5,       # Left ventricular unstressed volume # adjusted to the shi paper
  "Vu_ra": 4,           # Right atrial unstressed volume # adjusted to the shi paper
  "Vu_rv": 10,       # Right ventricular unstressed volume # adjusted to the shi paper
  "KE_la": 0.05,
  "KE_ra": 0.05,
  "Vu_sa": 0, # Systemic arterial unstressed volume
  "Vu_vc": 123,      # Vena cava unstressed volume
  "Vvc_max": 350,     # Maximum volume of vena cava
  "Vvc_min": 50,      # Minimum volume of vena cava
  "V_tot": 5027.6,    # Total blood volume
  "Vu_amp": 60.22,   # Active skeletal muscle peripheral unstressed volume
    "Vu_bp": 68.42,    # Brain peripheral unstressed volume
    "Vu_bv": 279.49,   # Brain venous unstressed volume
    "Vu_ep": 127.72,   # Extra-splanchnic peripheral unstressed volume
    "Vu_hp": 23,       # Coronary peripheral unstressed volume
    "Vu_hv": 93.16,    # Coronary venous unstressed volume
    "Vu_rmp": 40.1,    # Resting skeletal muscle peripheral unstressed volume
    "Vu_sp": 260.3,    # Splanchnic peripheral unstressed volume
  "kr_am": 24.17,  # Constant parameter
  "Io_sh": 0.658,            # Value of exercise intensity at the central point of the sigmoid (heart)
  "Io_sp": 0.65,             # Value of exercise intensity at the central point of the sigmoid (peripheral resistance)
    "Io_sv": 0.45,             # Value of exercise intensity at the central point of the sigmoid (unstressed volume of veins)
    "Io_v": 0.126,
  "tau_Emax_lv": 8,       # Time constant
    "tau_Emax_rv": 8,       # Time constant
    "tau_Ramp": 2,          # Time constant
    "tau_Rep": 2,           # Time constant
    "tau_Rrmp": 2,           # Time constant
    "tau_Rsp": 2,           # Time constant
    "tau_Vamv": 20,         # Time constant
    "tau_Vev": 20,          # Time constant
    "tau_Vrmv": 20,          # Time constant
    "tau_Vsv": 20,          # Time constant
  "tau_cc": 20,              # Time constant
    "tau_isc": 30,             # Time constant of the mechanism

  "MRTCO2_basal": 0.2/60,            # Basal metabolic production rate for CO2 (l/min STPD)
    "MRTO2_basal": 0.25/60,            # Basal metabolic consumption rate for O2 (l/min STPD)

  "tau_p": 2.076,         # Time constant for the real pole
    "tau_z": 6.37,          # Time constant for the real zero
  "tau_ac": 2,            # Time constant of the chemoreceptor mechanism
  "tau_ap": 2,       # Time constant of the lung inflation afferent response, seconds
  "tau_Ts": 2,              # Time constant
    "tau_Tv": 1.5,            # Time constant
  "tau_CO2": 20,    # Time constant of the effect of CO2 on cerebral circulation, seconds
    "tau_O2": 10,     # Time constant of the effect of O2 on cerebral circulation, seconds
  "tau_w": 5,         # Time constant of the filter
  "Io_met": 0.4266,          # Is I at the central point of the sigmoid
  "tau_M": 40,               # Time constant
    "tau_met": 10,             # Time constant

  "Fi_CO2": 0.0421,       # Inspired fraction of CO2 (%)
    "Fi_O2": 21.0379,       # Inspired fraction of O2 (%)
    # "LCTV": 0.588,          # Lung to chemoreceptor vascular volume constant (l)
    "PACO2_Delay_IC": 40.4448,      # Initial CO2 convection delay (mmHg)
    # "dPa_CO2_dt_IC": -0.2465,       # Initial CO2 rate of change (mmHg/s)
    "P_atm": 760,                    # Atmospheric pressure (mmHg) # CHANGED
    "P_ws": 47,                      # Water vapor pressure (mmHg)
    "T1": 1,                        # Time constant for cardiovascular mixing (s)
    "T2": 2,                        # Time constant for cardiovascular mixing (s)
    "VL_CO2": 3,                    # Lungs storage volume for CO2 (l)
    "VL_O2": 2.5,                   # Lungs storage volume for O2 (l)
    "Z": 0.0227,                    # Molar conversion factor (l/mmol)
    "VB": 0.9,                      # Gas volume in brain (L)
  "KCSFCO2": 320,  # CO2 diffusion time constant in cerebrospinal fluid (s)
  "tauMR": 50,                    # Metabolic rate time constant (s)
    "VTCO2": 0.25,                    # Body tissue storage volume for CO2 (l) # changed from 15
    "VTO2": 0.25,                      # Body tissue storage volume for O2 (l) # changed from 6
  "MRCO2": 0.2/60,               # Minimum metabolic production rate for CO2 (l/min STPD)
    "MRO2": 0.25/60,               # Minimum metabolic consumption rate for O2 (l/min STPD)
  "tau_MRV": 50,  # Metabolic rate time constant (s)
  "lambda1": 0.4,         # Weighting factor (Dimensionless) changed
    "lambda2": 0.05,        # Weighting factor (Dimensionless) changed
    "n": 1.101,              # Power index of efficiency factor (Dimensionless)
  "P_ao": 0,                # Airway pressure (cmH2O)


  "DT_s": 2,              # Pure latency of the mechanism
    "DT_v": 0.2,            # Pure latency of the mechanism
  "PAO2_Delay_IC": 103.1223,
}


