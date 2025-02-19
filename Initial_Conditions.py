Initial_Conditions = {
    # Table 1. Systemic arteries

# cardiovascular system = {
# state
    "VT_pa": 70, # 9.264, # trenhago
    "VT_pp": 200, # 193.1305 would be trenhago
    "VT_pv": 215, # 344.590143
    "Q_pa": 0,
    "VT_la": 50, # (Vu_la + (P_la-P_thor) * C_la) where P_la = 0 at end diastole, P_thor = -4
    "VT_lv": 180, # 291.924595828,
    "VT_ra": 50,
    "VT_rv": 200,
    "VT_sv": 1361.6,
    "VT_bv": 300,
    "VT_hv": 100,
    "VT_rmv": 190.95,
    "VT_amv": 286.4,
    "VT_ev": 607.8,
    "P_sp": 66,
    "V_sa": 22,
    "P_sa": 92,
    "Q_sa": 0,
    "VT_vc": 350, # set in parameters as the max vc


    # "VT_pa": 6.84,
    # "VT_pp": 116.6775,
    # "VT_pv": 114,
    # "Q_pa": 0,
    # "VT_la": 40,
    # "VT_lv": 160,
    # "VT_ra": 44,
    # "VT_rv": 160,
    # "VT_sv": 1361.6,
    # "VT_bv": 279.49,
    # "VT_hv": 93.16,
    # "VT_rmv": 190.95,
    # "VT_amv": 286.4,
    # "VT_ev": 607.8,
    # "P_sp": 66,
    # "V_sa": 22,
    # "P_sa": 70,
    # "Q_sa": 0,
    # "VT_vc": 350,

# initial condition of inputs from other controllers
    "U": 0,
    "beta": 0,
    "BF": 0.25,
    "Vu_ev": 607.8,
    "Vu_amv": 286.4,
    "Vu_rmv": 190.95,
    "Vu_sv": 1361.6,
    "R_ep": 1.655,
    "R_amp": 3.51,
    "R_rmp": 5.27,
    "R_sp": 2.49,
    "R_bp": 6.57,
    "R_hp": 19.71,
    "HR": 1.72, # want to change to 1.2 but following from 0.58 heart period
    "Emax_lv": 2.392, # 5.2, # should change based on literature (before: 2.392)
    "Emax_rv": 1.412,
    "I": 0,

# cardiovascular controller
    "theta_change_O2_sp": 0,
    "theta_change_CO2_sp": 0,
    "theta_change_O2_sv": 0,
    "theta_change_CO2_sv": 0,
    "theta_change_O2_sh": 0,
    "theta_change_CO2_sh": 0,
    "P_tilda": 92,
    "f_ac": 8.0807,
    "f_ap": 4.4492,
    "R_ep_change": 0,
    "R_sp_change": 0,
    "R_rmp_n_change": 0,
    "R_amp_n_change": 0,
    "Vu_ev_change": 0,
    "Vu_sv_change": 0,
    "Vu_rmv_change": 0,
    "Vu_amv_change": 0,
    "Emax_lv_change": 0,
    "Emax_rv_change": 0,
    "Ts_change": 0,
    "Tv_change": 0,
    "xb_O2": 0,
    "xb_CO2": 0,
    "xh_O2": 0,
    "xh_CO2": 0,
    "Wh": 12660,
    "xrm_O2": 0,
    "xrm_CO2": 0,
    "xam_O2": 0,
    "xM": 0,
    "x_met": 0,
# initial condition of inputs from other controllers
    "Wh_lv": 10800,
    "Wh_rv": 1860,
    "Ca_O2": 0.2,
    # "Q_bp": 0,
    "Q_hp": 10,
    "Q_rmp": 10,
    "Q_amp": 10,
    # "times": [0],
    # "f_sp_history": [],
    # "f_sh_history": [],
    # "f_v_history": [],
    # "phi_met_history": [],
    "VT": 0.73, # changed from 0.4 to the original parameter
    "dP_sa_dt": 0,
    # "PaO2": 80,
    # "PaCO2": 40,
    # "MRTCO2":
    "TI": 1.8,
    "beta2": 0,
    # "previous_VE": [],

# Respiratory mechanics
    "V": 0,
    "alpha": 0,
    "Vflow_ua": 0,
# other inputs
#     "Nd": [0, 0, 0, 0, 0, 0],

# Respiratory controller
    "VE_integral": 0,
# other inputs
    "dV_dt": 0,
    # "previous_dV_dt": [0],
    "P_musc": 0,
    "WI": 0,
    "WE": 0,
    # "previous_WI": [0],
    # "previous_WE": [0],
    # "Pa_CO2": 40,
    # "Pa_O2": 80,
    "PbCO2": 48.5383,
    # "MRV": 0,

# Gas exchange
    "Pd_1_O2": 104.3637,
    "Pd_1_CO2": 39.5616,
    "Pd_2_O2": 104.3637,
    "Pd_2_CO2": 39.6736,
    "Pd_3_O2": 104.0505,
    "Pd_3_CO2": 39.8127,
    "Pd_4_O2": 103.8005,
    "Pd_4_CO2": 40.0061,
    "Pd_5_O2": 103.3579,
    "Pd_5_CO2": 40.3359,
    "Pa_O2": 100, # ignore: set 0 here, but it is another value initially from Gas_Exchange.py
    "Pa_CO2": 40, # ignore: set 0 here, but it is another value initially from Gas_Exchange.py
    "dPa_O2_dt": 0.3557,
    "dPa_CO2_dt": -0.2465,
    "PA_O2": 102.5153,
    "PA_CO2": 40.9432,
    "PvbCO2": 43.7, # # can't be 0 here. set 0 here, but it is another value initially from Gas_Exchange.py
    "PCSFCO2": 43.6,
    "MRTO2": 0.33,
    "MRTCO2": 0.3,
    "CvO2": 0.1639,
    "CvCO2": 0.5247,
    "MRV": 0,
    # other inputs from other systems
    # "dV_dt":
    # "V": 0,
    "Q_pp": 10, # 83.33,
    "Q_bp": 10, # 16.667, # or 10.5
    "Q_la": 10, # set based on simulation results and visually looking at plot (need to change)
    "VD": 0.15,
}