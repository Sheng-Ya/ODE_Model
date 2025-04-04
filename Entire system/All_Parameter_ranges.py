lower = 0.8
upper = 1.2

parameters = {
    "C_sa": (0.28 * lower, 0.28 * upper),
    "L_sa": (0.00022 * lower, 0.00022 * upper),
    "R_sa": (0.06 * lower, 0.06 * upper),
    # "Vu_sa": (0.0 * lower, 10.0 * upper),
    "C_amp": (0.315 * lower, 0.315 * upper),
    "C_amv": (9.4 * lower, 9.4 * upper),
    "C_bp": (0.358 * lower, 0.358 * upper),
    "C_bv": (10.71 * lower, 10.71 * upper),
    "C_ep": (0.668 * lower, 0.668 * upper),
    "C_ev": (20 * lower, 20 * upper),
    "C_hp": (0.119 * lower, 0.119 * upper),
    "C_hv": (3.57 * lower, 3.57 * upper),
    "C_rmp": (0.21 * lower, 0.21 * upper),
    "C_rmv": (6.28 * lower, 6.28 * upper),
    "C_sp": (2.05 * lower, 2.05 * upper),
    "C_sv": (61.11 * lower, 61.11 * upper),
    # "kr_am": (24.17 * lower, 24.17 * upper),
    # "P_0": (3.93 * lower, 3.93 * upper),
    "R_amv_n": (0.0833 * lower, 0.0833 * upper),
    "R_bv_n": (0.075 * lower, 0.075 * upper),
    "R_ev_n": (0.04 * lower, 0.04 * upper),
    "R_hv_n": (0.224 * lower, 0.224 * upper),
    "R_rmv_n": (0.125 * lower, 0.125 * upper),
    "R_sv_n": (0.038 * lower, 0.038 * upper),
    # "V_tot": (5027.6 * lower, 5027.6 * upper),
    # "Vu_amp": (60.22 * lower, 60.22 * upper),
    # "Vu_bp": (68.42 * lower, 68.42 * upper),
    # "Vu_bv": (279.49 * lower, 279.49 * upper),
    # "Vu_ep": (127.72 * lower, 127.72 * upper),
    # "Vu_hp": (23 * lower, 23 * upper),
    # "Vu_hv": (93.16 * lower, 93.16 * upper),
    # "Vu_rmp": (40.1 * lower, 40.1 * upper),
    # "Vu_sp": (260.3 * lower, 260.3 * upper),
    "D1": (0.3855 * lower, 0.3855 * upper),
    "D2": (-5 * upper, -5 * lower),
    "K1_vc": (0.15 * lower, 0.15 * upper),
    "K2_vc": (0.4 * lower, 0.4 * upper),
    "Kr_vc": (0.001 * lower, 0.001 * upper),
    "Rvc_n": (0.025 * lower, 0.025 * upper),
    # "Vu_vc": (123 * lower, 123 * upper),
    # "Vvc_max": (350 * lower, 350 * upper),
    # "Vvc_min": (50 * lower, 50 * upper),
    "C_pa": (0.76 * lower, 0.76 * upper),
    "C_pp": (5.8 * lower, 5.8 * upper),
    "C_pv": (25.37 * lower, 25.37 * upper),
    "L_pa": (0.00018 * lower, 0.00018 * upper),
    "R_pa": (0.023 * lower, 0.023 * upper),
    "R_pp": (0.0894 * lower, 0.0894 * upper),
    "R_pv": (0.0056 * lower, 0.0056 * upper),
    # "Vu_pa": (0.0 * lower, 10.0 * upper),
    # "Vu_pp": (116.6775 * lower, 116.6775 * upper),
    # "Vu_pv": (114 * lower, 114 * upper),
    # "KE_lv": (0.014 * lower, 0.014 * upper),
    # "KE_rv": (0.011 * lower, 0.011 * upper),
    "Emax_la": (0.45 * lower, 0.45 * upper),
    "P0_la": (0.45 * lower, 0.45 * upper),
    # "KE_la": (0.05 * lower, 0.05 * upper),
    "Emax_ra": (0.45 * lower, 0.45 * upper),
    "P0_ra": (0.45 * lower, 0.45 * upper),
    # "KE_ra": (0.05 * lower, 0.05 * upper),
    "P0_lv": (1.5 * lower, 1.5 * upper),
    "P0_rv": (1.5 * lower, 1.5 * upper),
    # "Vu_la": (24 * lower, 24 * upper),
    # "Vu_lv": (15.908 * lower, 15.908 * upper),
    # "Vu_ra": (24 * lower, 24 * upper),
    # "Vu_rv": (38.703 * lower, 38.703 * upper),
    # "A_im": (50 * lower, 50 * upper),
    # "Tc": (0.75 * lower, 0.75 * upper),
    # "T_im": (1 * lower, 1 * upper),
    "g_abd": (3.39 * lower, 3.39 * upper),
    "g_thor": (6.8 * lower, 6.8 * upper),
    "P_abdmax_n": (-1 * upper, 0 * lower),
    "P_abdmin_n": (-2.5 * upper, -2.5 * lower),
    "P_thormax_n": (-1 * upper, 0.0 * lower),
    "P_thormin_n": (-3 * upper, 0.0 * lower),
    "VT_n": (0.73 * lower, 0.73 * upper),
    "BF": (0.25 * lower, 0.25 * upper),

    "Emax_lv": (2.392 * lower, 2.392 * upper),
    "Emax_rv": (1.412 * lower, 1.412 * upper),
    "HR": (1.2 * lower, 1.2 * upper),
    "T_resp": (4 * lower, 4 * upper),
    "TI": (1.8 * lower, 1.8 * upper),
    "VT": (0.73 * lower, 0.73 * upper),
    # "Vu_ev": (607.8 * lower, 607.8 * upper),
    # "Vu_amv": (286.4 * lower, 286.4 * upper),
    # "Vu_rmv": (190.95 * lower, 190.95 * upper),
    # "Vu_sv": (1361.6 * lower, 1361.6 * upper),
    "R_ep": (1.655 * lower, 1.655 * upper),
    "R_amp": (3.51 * lower, 3.51 * upper),
    "R_rmp": (5.27 * lower, 5.27 * upper),
    "R_sp": (2.49 * lower, 2.49 * upper),
    "R_bp": (6.57 * lower, 6.57 * upper),
    "R_hp": (19.71 * lower, 19.71 * upper),


    "LCTV": (0.588 * lower, 0.588 * upper),
    "T1": (1 * lower, 1 * upper),
    "T2": (2 * lower, 2 * upper),
    "VL_CO2": (3 * lower, 3 * upper),
    "VL_O2": (2.5 * lower, 2.5 * upper),
    "Z": (0.0227 * lower, 0.0227 * upper),

    "dc": (0.015 * lower, 0.015 * upper),
    "h": ((0.0183 / 1000) * lower, (0.0183 / 1000) * upper),
    "KCCO2": (346000 * lower, 346000 * upper),
    "KCSFCO2": (320 * lower, 320 * upper),
    "SbCO2": ((0.36 / 1000) * lower, (0.36 / 1000) * upper),
    "SCO2": (0.0043 * lower, 0.0043 * upper),

    "VTCO2": (0.25 * lower, 0.25 * upper),
    "VTO2": (0.25 * lower, 0.25 * upper),

    "E_CW": (10.545 * lower, 10.545 * upper),
    "E_L": (10.545 * lower, 10.545 * upper),
    "k_aw1": (1.85 * lower, 1.85 * upper),
    "k_aw2": (0.43 * lower, 0.43 * upper),
    "P_ao": (0.0 * lower, 1.0 * upper),
    "R_rs": (3.02 * 0.73559 * lower, 3.02 * 0.73559 * upper),

    "A0_ua": (1 * lower, 1 * upper),
    "C_ua": ((0.001 / 0.73559) * lower, (0.001 / 0.73559) * upper),
    "K_ua": ((1 / 0.73559) * lower, (1 / 0.73559) * upper),
    "Pcrit_min": ((-40 * 0.73559) * upper, (-40 * 0.73559) * lower),
    "R_CW": (0.8326 * 0.73559 * lower, 0.8326 * 0.73559 * upper),
    "R_trachea": (1000000 * 0.73559 * lower, 1000000 * 0.73559 * upper),

    "GV_dead": (0.1698 * lower, 0.1698 * upper),
    "Kbg": (17.4 * lower, 17.4 * upper),
    "KcCO2": (0.2332 * lower, 0.2332 * upper),
    "KcMRV": (1 * lower, 1 * upper),
    "KpCO2": (0.2025 * lower, 0.2025 * upper),
    "KpO2": (4.72e-9 * lower, 4.72e-9 * upper),
    "V0_dead": (0.1587 * lower, 0.1587 * upper),
    "VA_rest": (0.0673 * lower, 0.0673 * upper)
}

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

