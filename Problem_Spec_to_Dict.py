import ast
from pathlib import Path

from SALib import ProblemSpec

lower = 0.8
upper = 1.2

sp = ProblemSpec({
    'names': [
        'beta2',
        'C2',
        'K2',
        'a2',
        'alpha2',
        'KCCO2',
        'GV_dead',
        'KcCO2',
        'KcMRV',
        'KpCO2',
        'KpO2',
        'V0_dead',
        'VA_rest',
        'E_rs',
        'R_rs',
        'C_jp',
        'C_sa',
        'L_sa',
        'R_sa',
        'C_amv',
        'C_bv',
        'C_ev',
        'C_hv',
        'C_rmv',
        'C_sv',
        'kr_am',
        'P_0',
        'R_amv_n',
        'R_bv_n',
        'R_ev_n',
        'R_hv_n',
        'R_rmv_n',
        'R_sv_n',
        'K1_vc',
        'D1',
        'Vvc_min',
        'Kr_vc',
        'Rvc_n',
        'C_pa',
        'C_pp',
        'C_pv',
        'L_pa',
        'R_pa',
        'R_pp',
        'R_pv',
        'Emax_la',
        'P0_la',
        'Emax_ra',
        'P0_ra',
        'KE_la',
        'KE_ra',
        'P0_lv',
        'P0_rv',
        's',
        'fab_o',
        'fes_o',
        'fes_inf',
        'fes_max',
        'fev_o',
        'fev_inf',
        'kes',
        'kev',
        'Io_sh',
        'Io_sp',
        'Io_sv',
        'Io_v',
        'kcc_sh',
        'kcc_sp',
        'kcc_sv',
        'kcc_v',
        'Ysh_max',
        'Ysh_min',
        'Ysp_max',
        'Ysp_min',
        'Ysv_max',
        'Ysv_min',
        'Yv_max',
        'Yv_min',
        'theta_v',
        'Wb_sh',
        'Wb_sp',
        'Wb_sv',
        'Wc_sh',
        'Wc_sp',
        'Wc_sv',
        'Wc_v',
        'Wp_sp',
        'Wp_sv',
        'Wp_v',
        'Wt_sh',
        'Wt_sp',
        'Wt_sv',
        'Wt_v',
        'Emax_lv0',
        'Emax_rv0',
        'fes_min',
        'GEmax_lv',
        'GEmax_rv',
        'GR_amp',
        'GR_ep',
        'GR_rmp',
        'GR_sp',
        'GV_amv',
        'GV_ev',
        'GV_rmv',
        'GV_sv',
        'R_amp0',
        'R_ep0',
        'R_rmp0',
        'R_sp0',
        'g_ccsh',
        'g_ccsp',
        'kisc_sh',
        'kisc_sp',
        'kisc_sv',
        'PO2_sh',
        'PO2_sp',
        'PO2_sv',
        'theta_shn',
        'theta_spn',
        'theta_svn',
        'x_sh',
        'x_sp',
        'x_sv',
        'PaCO2_n',
        'f_ab_max',
        'f_ab_min',
        'k_ab',
        'P_n',
        'P_n_max',
        'f_acCO2_n',
        'f_ac_max',
        'f_ac_min',
        'k_ac',
        'K_H',
        'PaO2_ac_n',
        'G_ap',
        'GT_s',
        'GT_v',
        'T0',
        'A',
        'B',
        'C',
        'D',
        'Cvb_O2_n',
        'gb_O2',
        'MO2_bp',
        'R_bpn',
        'Cvh_O2_n',
        'Cvrm_O2_n',
        'gh_O2',
        'grm_O2',
        'Kh_CO2',
        'Krm_CO2',
        'MO2_hpn',
        'MO2_rmp',
        'R_hpn',
        'W_hn',
        'Cvam_O2_n',
        'gam_O2',
        'gM',
        'Io_met',
        'kmet',
        'MO2_ampn',
        'phi_max',
        'phi_min',
        'Kp_ao',
        'Kf_ao',
        'Kb_ao',
        'Kv_ao',
        'theta_ao_max',
        'Kp_mi',
        'Kf_mi',
        'Kb_mi',
        'Kv_mi',
        'theta_mi_max',
        'Kp_po',
        'Kf_po',
        'Kb_po',
        'Kv_po',
        'theta_po_max',
        'Kp_tr',
        'Kf_tr',
        'Kb_tr',
        'Kv_tr',
        'theta_tr_max',
        'alpha_O2',
        'R_po',
        'R_mi',
        'R_tr',
        'R_ao',
        'C_O2_param1',
        'C_O2_param2',
        'C_O2_param3',
        'PAMO2_nominal',
        'Vu_bv',
        'Vu_hv',
        'Vu_jp',
        'Vu_vc',
        'Vu_pp',
        'Vu_pv',
        'Vu_la',
        'Vu_lv',
        'Vu_ra',
        'Vu_rv',
        'tau_Emax_lv',
        'tau_Emax_rv',
        'tau_Ramp',
        'tau_Rep',
        'tau_Rrmp',
        'tau_Rsp',
        'tau_Vamv',
        'tau_Vev',
        'tau_Vrmv',
        'tau_Vsv',
        'Vu_amv0',
        'Vu_ev0',
        'Vu_rmv0',
        'Vu_sv0',
        'tau_cc',
        'tau_isc',
        'tau_p',
        'tau_z',
        'tau_ac',
        'tau_ap',
        'tau_Ts',
        'tau_Tv',
        'tau_CO2',
        'tau_O2',
        'tau_w',
        'tau_M',
        'tau_met',
        'DEmax_lv',
        'DEmax_rv',
        'DR_amp',
        'DR_ep',
        'DR_rmp',
        'DR_sp',
        'DV_amv',
        'DV_ev',
        'DV_rmv',
        'DV_sv',
        'DT_s',
        'DT_v',
        'Dmet',
        'Ta',
        'KE_lv',
        'KE_rv',
        'T1',
        'T2',
        'VL_CO2',
        'VL_O2',
        'KCSFCO2',
        'VB',
        'tauMR',
        'VTCO2',
        'VTO2',
        'tau_MRV',
        'scale_param1',
        'scale_param3',
        'scale_param4',
        'scale_param6',
        'Pa_O2_lower',
        'rise_time_atr',
        'rise_time_ven',
        'fall_time_ven',
        'ahead1',
        'theta_min',
        'r',
        'l',
        'V_nominal',
        'V_scale',
    ],
    'bounds': [
        [0.03634378256 * lower, 0.03634378256 * upper],  # beta2 [MAP]
        [77.013532273104 * lower, 77.013532273104 * upper],  # C2 [MAP]
        [218.256037687845 * lower, 218.256037687845 * upper],  # K2 [MAP]
        [2.074868042242 * lower, 2.074868042242 * upper],  # a2 [MAP]
        [0.05591 * lower, 0.05591 * upper],
        [346000 * lower, 346000 * upper],
        [0.1698 * lower, 0.1698 * upper],
        [0.2332 * lower, 0.2332 * upper],
        [1 * lower, 1 * upper],
        [0.2025 * lower, 0.2025 * upper],
        [0.00000000472 * lower, 0.00000000472 * upper],
        [0.139022611691 * lower, 0.139022611691 * upper],  # V0_dead [MAP]
        [0.0673 * lower, 0.0673 * upper],
        [18.817362115723 * lower, 18.817362115723 * upper],  # E_rs [MAP]
        [3.329726165664 * lower, 3.329726165664 * upper],  # R_rs [MAP]
        [4.173290265104 * lower, 4.173290265104 * upper],  # C_jp [MAP]
        [0.28 * lower, 0.28 * upper],
        [0.00022 * lower, 0.00022 * upper],
        [0.050245051994 * lower, 0.050245051994 * upper],  # R_sa [MAP]
        [9.4 * lower, 9.4 * upper],
        [10.71 * lower, 10.71 * upper],
        [20 * lower, 20 * upper],
        [3.57 * lower, 3.57 * upper],
        [6.28 * lower, 6.28 * upper],
        [68.204629886668 * lower, 68.204629886668 * upper],  # C_sv [MAP]
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
        [0.021950319159 * lower, 0.021950319159 * upper],  # Rvc_n [MAP]
        [5.85 * lower, 5.85 * upper],
        [5.8 * lower, 5.8 * upper],
        [25.37 * lower, 25.37 * upper],
        [0.00018 * lower, 0.00018 * upper],
        [0.019738686634 * lower, 0.019738686634 * upper],  # R_pa [MAP]
        [0.075390089858 * lower, 0.075390089858 * upper],  # R_pp [MAP]
        [0.0056 * lower, 0.0056 * upper],
        [0.381477545517 * lower, 0.381477545517 * upper],  # Emax_la [MAP]
        [0.509783597278 * lower, 0.509783597278 * upper],  # P0_la [MAP]
        [0.387109011509 * lower, 0.387109011509 * upper],  # Emax_ra [MAP]
        [0.381462937748 * lower, 0.381462937748 * upper],  # P0_ra [MAP]
        [0.05609992622 * lower, 0.05609992622 * upper],  # KE_la [MAP]
        [0.043098658992 * lower, 0.043098658992 * upper],  # KE_ra [MAP]
        [1.517954367078 * lower, 1.517954367078 * upper],  # P0_lv [MAP]
        [1.30591367592 * lower, 1.30591367592 * upper],  # P0_rv [MAP]
        [0.04 * lower, 0.04 * upper],
        [21.359942722077 * lower, 21.359942722077 * upper],  # fab_o [MAP]
        [15.434807391078 * lower, 15.434807391078 * upper],  # fes_o [MAP]
        [2.432881945042 * lower, 2.432881945042 * upper],  # fes_inf [MAP]
        [80 * lower, 80 * upper],
        [3.531127504801 * lower, 3.531127504801 * upper],  # fev_o [MAP]
        [5.86760871137 * lower, 5.86760871137 * upper],  # fev_inf [MAP]
        [0.058869060069 * lower, 0.058869060069 * upper],  # kes [MAP]
        [7.06 * lower, 7.06 * upper],
        [0.658 * lower, 0.658 * upper],
        [0.65 * lower, 0.65 * upper],
        [0.411591838192 * lower, 0.411591838192 * upper],  # Io_sv [MAP]
        [0.126 * lower, 0.126 * upper],
        [0.114 * lower, 0.114 * upper],
        [0.13 * lower, 0.13 * upper],
        [0.101736262307 * lower, 0.101736262307 * upper],  # kcc_sv [MAP]
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
        [-1.509384850803 * upper, -1.509384850803 * lower],  # Wb_sh [MAP]
        [-1.1375 * upper, -1.1375 * lower],
        [-1.10346134896 * upper, -1.10346134896 * lower],  # Wb_sv [MAP]
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
        [2.004921129836 * lower, 2.004921129836 * upper],  # Emax_lv0 [MAP]
        [1.232201870061 * lower, 1.232201870061 * upper],  # Emax_rv0 [MAP]
        [2.55747208445 * lower, 2.55747208445 * upper],  # fes_min [MAP]
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
        [12.421919448785 * lower, 12.421919448785 * upper],  # theta_svn [MAP]
        [53 * lower, 53 * upper],
        [6 * lower, 6 * upper],
        [6 * lower, 6 * upper],
        [38.1785623276 * lower, 38.1785623276 * upper],  # PaCO2_n [MAP]
        [43.317964244628 * lower, 43.317964244628 * upper],  # f_ab_max [MAP]
        [2.52 * lower, 2.52 * upper],
        [9.865923128457 * lower, 9.865923128457 * upper],  # k_ab [MAP]
        [94.058616807439 * lower, 94.058616807439 * 1.05],  # P_n [MAP]
        [112 * 0.9, 112 * upper],
        [1.4 * lower, 1.4 * upper],
        [12.3 * lower, 12.3 * upper],
        [0.835 * lower, 0.835 * upper],
        [29.27 * lower, 29.27 * upper],
        [3 * lower, 3 * upper],
        [45 * lower, 45 * upper],
        [11.76 * lower, 11.76 * upper],
        [-0.111277976334 * upper, -0.111277976334 * lower],  # GT_s [MAP]
        [0.08939442599 * lower, 0.08939442599 * upper],  # GT_v [MAP]
        [0.662131474971 * lower, 0.662131474971 * upper],  # T0 [MAP]
        [20.9 * lower, 20.9 * upper],
        [92.8 * lower, 92.8 * upper],
        [10570 * lower, 10570 * upper],
        [-5.251 * upper, -5.251 * lower],
        [0.14 * lower, 0.14 * upper],
        [10 * lower, 10 * upper],
        [0.794301887941 * lower, 0.794301887941 * upper],  # MO2_bp [MAP]
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
        [0.137542480239 * lower, 0.137542480239 * upper],  # Cvam_O2_n [MAP]
        [30 * lower, 30 * upper],
        [40 * lower, 40 * upper],
        [0.438967333221 * lower, 0.438967333221 * upper],  # Io_met [MAP]
        [0.159244607475 * lower, 0.159244607475 * upper],  # kmet [MAP]
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
        [4.046658659346 * lower, 4.046658659346 * upper],  # Kv_mi [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [2000 * lower, 2000 * upper],
        [2 * lower, 2 * upper],
        [7.27611504046 * lower, 7.27611504046 * upper],  # Kv_po [MAP]
        [1.309 * lower, 1.309 * upper],
        [2000 * lower, 2000 * upper],
        [200 * lower, 200 * upper],
        [2 * lower, 2 * upper],
        [2.958243334097 * lower, 2.958243334097 * upper],  # Kv_tr [MAP]
        [1.309 * lower, 1.309 * upper],
        [0.0000317 * lower, 0.0000317 * upper],
        [350 * lower, 350 * upper],
        [400 * lower, 400 * upper],
        [400 * lower, 400 * upper],
        [350 * lower, 350 * upper],
        [0.001475698043 * lower, 0.001475698043 * upper],  # C_O2_param1 [MAP]
        [2.6 * lower, 2.6 * upper],
        [0.0000303 * lower, 0.0000303 * upper],
        [104 * lower, 104 * upper],
        [241.933855976857 * lower, 241.933855976857 * upper],  # Vu_bv [MAP]
        [93.16 * lower, 93.16 * upper],
        [651.194627289645 * lower, 651.194627289645 * upper],  # Vu_jp [MAP]
        [123 * lower, 123 * upper],
        [116.68 * lower, 116.68 * upper],
        [114 * lower, 114 * upper],
        [20.652359606871 * lower, 20.652359606871 * upper],  # Vu_la [MAP]
        [18.471562850064 * lower, 18.471562850064 * upper],  # Vu_lv [MAP]
        [32.335683552201 * lower, 32.335683552201 * upper],  # Vu_ra [MAP]
        [44.227623158586 * lower, 44.227623158586 * upper],  # Vu_rv [MAP]
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
        [251.411004097183 * lower, 251.411004097183 * upper],  # Vu_amv0 [MAP]
        [601.497648948428 * lower, 601.497648948428 * upper],  # Vu_ev0 [MAP]
        [190.95 * lower, 190.95 * upper],
        [1202.95398997038 * lower, 1202.95398997038 * upper],  # Vu_sv0 [MAP]
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
        [0.012487192385 * lower, 0.012487192385 * upper],  # KE_lv [MAP]
        [0.009263717414 * lower, 0.009263717414 * upper],  # KE_rv [MAP]
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
        [0.0387130545 * lower, 0.0387130545 * upper],  # rise_time_atr [MAP]
        [0.335074462135 * lower, 0.335074462135 * upper],  # rise_time_ven [MAP]
        [0.497391869651 * 0.85, 0.497391869651 * 1.15],  # fall_time_ven [MAP]
        [0.952740983617 * 0.92, 0.952740983617 * 1.08],  # ahead1 [MAP]
        [0.0873 * lower, 0.0873 * upper],
        [1.104913108062 * 0.85, 1.104913108062 * 1.15],  # r [MAP]
        [1.361143921931 * 0.85, 1.361143921931 * 1.15],  # l [MAP]
        [140.676129000154 * lower, 140.676129000154 * upper],  # V_nominal [MAP]
        [42.17363461731 * lower, 42.17363461731 * upper],  # V_scale [MAP]
    ]
})

# Compute nominal values from the base value in each bounds expression, rather
# than from the midpoint of the evaluated bounds.
precision = 12


def _numeric_literals(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return [float(node.value)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return [-value for value in _numeric_literals(node.operand)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _numeric_literals(node.left) + _numeric_literals(node.right)
    return []


def _nominal_from_bound(bound):
    lower_candidates = _numeric_literals(bound.elts[0])
    upper_candidates = _numeric_literals(bound.elts[1])

    for value in lower_candidates:
        if value in upper_candidates:
            return value

    if lower_candidates:
        return lower_candidates[0]

    raise ValueError(f"Could not infer nominal value from bounds expression: {ast.unparse(bound)}")


def _nominals_from_problem_spec_source():
    source = Path(__file__).read_text()
    module = ast.parse(source)

    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "sp" for target in statement.targets):
            continue
        if not isinstance(statement.value, ast.Call) or not statement.value.args:
            continue

        problem_dict = statement.value.args[0]
        if not isinstance(problem_dict, ast.Dict):
            continue

        for key, value in zip(problem_dict.keys, problem_dict.values):
            if isinstance(key, ast.Constant) and key.value == "bounds":
                return [
                    _nominal_from_bound(bound)
                    for bound in value.elts
                ]

    raise ValueError("Could not find bounds in ProblemSpec source.")


nominal_values = [
    round(value, precision)
    for value in _nominals_from_problem_spec_source()
]

if len(nominal_values) != len(sp["names"]):
    raise ValueError("ProblemSpec names and nominal values have different lengths.")

# Build the Parameters dictionary
Parameters = {name: value for name, value in zip(sp["names"], nominal_values)}

# Pretty-print it in the exact formatting you want
# print("Parameters = {")
# for k, v in Parameters.items():
#     print(f'    "{k}": {v},')
# print("}")

print("Parameters = { " + ", ".join(f'"{k}": {v}' for k, v in Parameters.items()) + " }")
