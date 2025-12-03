# original params
    # CARDIOVASCULAR SYSTEM
Parameters = {
    # Table 1. Systemic arteries
# systemic_arteries = {
    "C_sa": 0.28,  # Systemic arterial compliance (decreasing C_sa allows Q_sa to match closer to Q_lv) # want to change to 1.13 (harry thesis
    "L_sa": 0.00022,  # Systemic arterial inertance
    "R_sa": 0.2,  # Systemic arterial hydraulic resistance (want to increase from 0.06 to 0.2 to increase Psys). This is because P_sa decreases at a slower rater (first order)
    "Vu_sa": 1.0, # Systemic arterial unstressed volume (edited for DGSM)
# }
#
#     # Systemic peripheral and venous circulation
#     # Table 2. Compliance values
# systemic_peripheral_and_venous = {
    "C_jp": 3.72, # 3.72
    # "C_amp": 0.315,  # Active skeletal muscle peripheral compliance
    "C_amv": 4.4,    # Active skeletal muscle venous compliance
    # "C_bp": 0.358,   # Brain peripheral compliance
    "C_bv": 5.71,   # Brain venous compliance
    # "C_ep": 0.668,   # Extra-splanchnic peripheral compliance
    "C_ev": 10,      # Extra-splanchnic venous compliance
    # "C_hp": 0.119,   # Coronary peripheral compliance
    "C_hv": 1.57,    # Coronary venous compliance
    # "C_rmp": 0.21,   # Resting skeletal muscle peripheral compliance
    "C_rmv": 3.28,   # Resting skeletal muscle venous compliance
    # "C_sp": 2.05,    # Splanchnic peripheral compliance
    "C_sv": 31.11,   # Splanchnic venous compliance
    "kr_am": 24.17,  # Constant parameter
    "P_0": 3.93,     # Constant parameter
    "R_amv_n": 0.0833,  # Active skeletal muscle venous resistance
    "R_bv_n": 0.075,    # Brain venous resistance
    "R_ev_n": 0.04,     # Extra-splanchnic venous resistance
    "R_hv_n": 0.224,    # Coronary venous resistance
    "R_rmv_n": 0.125,   # Resting skeletal muscle venous resistance
    "R_sv_n": 0.038,    # Splanchnic venous resistance
    "V_tot": 5027.6,    # Total blood volume
    # "Vu_amp": 60.22,   # Active skeletal muscle peripheral unstressed volume
    # "Vu_bp": 68.42,    # Brain peripheral unstressed volume
    "Vu_bv": 279.49,   # Brain venous unstressed volume
    # "Vu_ep": 127.72,   # Extra-splanchnic peripheral unstressed volume
    # "Vu_hp": 23,       # Coronary peripheral unstressed volume
    "Vu_hv": 93.16,    # Coronary venous unstressed volume
    "Vu_jp": 579.76,
    # "Vu_rmp": 40.1,    # Resting skeletal muscle peripheral unstressed volume
    # "Vu_sp": 260.3,    # Splanchnic peripheral unstressed volume
# }
#
#     # Table 3. Vena Cava Parameters
# vena_cava = {
    "D1": 0.3855,      # Parameter for P-V curve of vena cava
    "D2": -5,          # Parameter for P-V curve of vena cava
    "K1_vc": 0.15,     # Parameter for P-V curve of vena cava
    "K2_vc": 0.4,      # Parameter for P-V curve of vena cava
    "Kr_vc": 0.0001,    # Gain for vena cava flow resistance
    "Rvc_n": 0.0025,    # Nominal vena cava flow resistance # edited (changed to 0.0025 from xxx for better left atrial pressures, changed back to 0.025)
    "Vu_vc": 123,      # Vena cava unstressed volume
    "Vvc_max": 350,     # Maximum volume of vena cava
    "Vvc_min": 50,      # Minimum volume of vena cava
# }
#
#     # Table 4. Pulmonary Circulation Parameters
# pulmonary_circulation = {
    "C_pa": 0.76, # 0.76,           # Pulmonary arterial compliances want to change to 5
    "C_pp": 15.8, # 5.8,            # Pulmonary peripheral compliances want to change to 10
    "C_pv": 25.37, # 25.37,          # Pulmonary venous compliances want to change to 15 # edited
    "L_pa": 0.00018,        # Pulmonary arterial inertance
    "R_pa": 0.023,          # Pulmonary arterial flow resistance (this value could raise RA pressure)
    "R_pp": 0.0894,         # Pulmonary peripheral flow resistance # edited to reduce CvtO2 oscillation # edited from 0.0894
    "R_pv": 0.0056,         # Pulmonary venous flow resistance # edited to remove the backflow
    "Vu_pa": 1,            # Pulmonary arterial unstressed volume # edited for dgsm
    "Vu_pp": 116.6775,     # Pulmonary peripheral unstressed volume
    "Vu_pv": 214,          # Pulmonary venous unstressed volume
# }

    # Table 5. Heart Parameters
# heart = {
#     "C_la": 4, # 19.23,          # Left atrial compliances changed
#     "C_ra": 5, # 31.25,          # Right atrial compliances changed
    "s": 0.04,
    "Ta": 2,
    "KE_lv": 0.014, # 0.014      # End-diastolic P-V relationship in left ventricle # adjusted (changes a lot depending on whether it is 0.06 or 0.05)
    "KE_rv": 0.011, # 0.011       # End-diastolic P-V relationship in right ventricle Another model use 0.027
    # "KR_lv": 0.000375,     # Viscosity of left ventricle
    # "KR_rv": 0.00014,       # Viscosity of right ventricle (this parameter affects RA pressure) (changed from 0.0014. The removes the kink in the lv)
    # "ksys": 0.075,         # Duration of systole as function of heart rate (this parameter affects RA pressure)
    "Emax_la": 0.34,    # edited
    "P0_la": 0.55,      # edited
    "KE_la": 0.05,

    "Emax_ra": 0.30,    # edited
    "P0_ra": 0.55,      # edited
    "KE_ra": 0.05,
    "P0_lv": 1.5, # 1.5        # End-diastolic P-V relationship in left ventricle # another model use 1.0
    "P0_rv": 1.5, # 1.5         # End-diastolic P-V relationship in right ventricle (check this one first, has strong effect with the smallest change in volume)
    # "R_la": 0.0025,         # Left atrial flow resistance
    # "R_ra": 0.0025,         # Right atrial flow resistance
    # "Tsys_0": 0.4,         # Duration of systole as function of heart rate (need to change Tsys0, T0, HR in initial/next conditions)
    "Vu_la": 4,           # Left atrial unstressed volume
    "Vu_lv": 5,       # Left ventricular unstressed volume
    "Vu_ra": 4,           # Right atrial unstressed volume (changing this doesn't affect atrial volume)
    "Vu_rv": 10,       # Right ventricular unstressed volume
# }

    # Table 6. Muscle Pump
# muscle_pump = {
    "A_im": 30,            # Peak value of intramuscular pressure # edited from 50
    "Tc": 0.7,            # The overall duration of muscular contraction # edited to match dgsm
    "T_im": 1.1,             # Duration of the muscular contraction-relaxation cycle # edited to match dgsm
# }

    # Table 7. Respiratory Pump
# respiratory_pump = {
    "g_abd": 3.39,          # Constant gain factor linking tidal volume changes to abdominal pressure variations
    "g_thor": 6.8,          # Constant gain factor linking tidal volume changes to intrathoracic pressure variations
    "P_abdmax_n": -1,        # Basal value of abdominal pressure at the end of expiration # edited to match dgsm
    "P_abdmin_n": -2.5,     # Basal value of abdominal pressure at the end of inspiration
    "P_thormax_n": -0,      # Basal value of intrathoracic pressure at the end of expiration
    "P_thormin_n": -0,      # Basal value of intrathoracic pressure at the end of inspiration
    "VT_n": 0.73,           # Basal value of tidal volume
# }

    # Table 8. Afferent Baroreflex Pathway
# afferent_baroreflex = {
    "f_ab_max": 47.78,      # Upper saturation level of the frequency discharge in the baroreceptor afferent fibers
    "f_ab_min": 2.52,       # Lower saturation level of the frequency discharge in the baroreceptor afferent fibers
    "k_ab": 11.76,          # Parameter related to the slope of the static function at the central point
    "P_n": 92,              # Value of baroreceptor pressure at the central point of the sigmoidal function
    "P_n_max": 112,
    "tau_p": 2.076,         # Time constant for the real pole
    "tau_z": 0.8,          # Time constant for the real zero
# }

    # Table 9. Afferent Chemoreflex Pathway
# afferent_chemoreflex = {
#     "f_ac_IC": 8.0807,      # Initial condition for the afferent activity from chemoreceptors
    "f_acCO2_n": 1.4,       # Constant parameter tuned to reproduce the CO2 static response
    "f_ac_max": 12.3,       # Upper saturation level of the frequency discharge in the chemoreceptor afferent fibers
    "f_ac_min": 0.835,      # Lower saturation level of the frequency discharge in the chemoreceptor afferent fibers
    "k_ac": 29.27,          # Parameter related to the slope of the sigmoid at the central point
    "K_H": 3,               # Constant parameter tuned to reproduce the CO2 static response
    "PaO2_ac_n": 45,        # Arterial PO2 at the central point of the sigmoid
    "PaCO2_n": 40,          # PaCO2 basal value
    "tau_ac": 2,            # Time constant of the chemoreceptor mechanism
# }

    # Table 10. Afferent activity from Pulmonary Stretch Receptors
# afferent_pulmonary_stretch_receptors = {
#     "f_ap_IC": 4.4492,  # Initial condition for the afferent activity from pulmonary stretch receptors, spikes/s
    "G_ap": 11.76,      # Constant gain factor, spikes/s/l
    "tau_ap": 2,       # Time constant of the lung inflation afferent response, seconds
# }

    # Table 11.  Cerebral Blood Flow (Blood flow local control)
# cerebral_blood_flow = {
    "A": 20.9,        # Constant parameter for cerebral blood flow regulation, dimensionless
    "B": 92.8,        # Constant parameter for cerebral blood flow regulation, dimensionless
    "C": 10570,       # Constant parameter for cerebral blood flow regulation, dimensionless
    "D": -5.251,      # Constant parameter for cerebral blood flow regulation, dimensionless
    "Cvb_O2_n": 0.14, # O2 concentration in venous blood leaving the brain under normal conditions
    "gb_O2": 10,      # Constant gain factor
    "MO2_bp": 0.925, # Oxygen consumption rate in the brain compartment # changed
    "R_bpn": 6.57,    # Constant parameter denoting the basal value of peripheral cerebrovascular conductance # edited from 6.57
    "tau_CO2": 20,    # Time constant of the effect of CO2 on cerebral circulation, seconds
    "tau_O2": 10,     # Time constant of the effect of O2 on cerebral circulation, seconds
# }

    # Table 12. Coronary and Resting Muscle Blood FLow (Blood flow local control)
# coronary_resting_muscle_blood_flow = {
    "Cvh_O2_n": 0.11,   # O2 concentration in venous blood leaving the heart under normal conditions
    "Cvrm_O2_n": 0.155, # O2 concentration in venous blood leaving the skeletal resting muscle under normal conditions
    "gh_O2": 35,        # Constant gain factor
    "grm_O2": 30,       # Constant gain factor
    "Kh_CO2": 11.11,    # Parameter related to the slope of the sigmoidal function at the central point
    "Krm_CO2": 142.8,   # Parameter related to the slope of the sigmoidal function at the central point
    "MO2_hpn": 0.4,     # Nominal value of O2 consumption rate in the heart
    "MO2_rmp": 0.86,    # Consumption rate in the resting muscle
    "R_hpn": 19.71,      # Normal peripheral resistance in coronary compartment # edited from 19.71
    "tau_w": 5,         # Time constant of the filter
    "W_hn": 12660,       # Nominal value of the average power of the cardiac pump
# }

    # Table 13. Active Muscle Blood Flow (Blood Flow Local Control)
# active_muscle_blood_flow = {
    "Cvam_O2_n": 0.1555,       # O2 concentration in venous blood leaving the active skeletal muscle under normal conditions
    "Dmet": 4,                 # Pure delay
    "gam_O2": 30,              # Constant gain factor
    "gM": 40,                  # Static gain
    "Io_met": 0.4266,          # Is I at the central point of the sigmoid
    "kmet": 0.18,              # Parameter related to the slope of the sigmoid at the central point
    "MO2_ampn": 0.516,         # Nominal oxygen consumption rate
    "phi_max": 20,             # Upper saturation of the static sigmoidal characteristic
    "phi_min": -1.87,          # Lower saturation of the static sigmoidal characteristic
    "tau_M": 40,               # Time constant
    "tau_met": 10,             # Time constant
# }

    # Table 14. CNS Ischemic Response
# cns_ischemic_response = {
    "g_ccsh": 1,                # Constant gain factor tuned to reproduce experimental results
    "g_ccsp": 1.5,              # Constant gain factor tuned to reproduce experimental results
    "g_ccsv": 0.2,                # Constant gain factor tuned to reproduce experimental results # edited to match dgsm
    "kisc_sh": 6,              # Parameter related to the slope of the static function at the central point for heart
    "kisc_sp": 2,              # Parameter related to the slope of the static function at the central point for peripheral resistance
    "kisc_sv": 2,              # Parameter related to the slope of the static function at the central point for unstressed volume of veins
    "PO2_sh": 45,              # Value of PO2 at the central point of the sigmoidal function for heart
    "PO2_sp": 30,              # Value of PO2 at the central point of the sigmoidal function for peripheral resistance
    "PO2_sv": 30,              # Value of PO2 at the central point of the sigmoidal function for unstressed volume of veins
    "tau_cc": 20,              # Time constant
    "tau_isc": 30,             # Time constant of the mechanism
    "theta_shn": 3.6,           # Offset term in basal condition for heart
    "theta_spn": 13.32,        # Offset term in basal condition for peripheral resistance
    "theta_svn": 13.32,        # Offset term in basal condition for unstressed volume of veins
    "x_sh": 53,                # Saturation of the hypoxic response for heart
    "x_sp": 6,                 # Saturation of the hypoxic response for peripheral resistance
    "x_sv": 6,                 # Saturation of the hypoxic response for unstressed volume of veins
# }

    # Table 15. Metabolic Regulation
# metabolic_regulation = {
    "AT": 1/60,                   # Anaerobic threshold
# }

    # Table 16: Parameters of Efferent Pathways
# efferent_pathways = {
    "fab_o": 25,               # Central value in the curve of fab
    "fes_o": 16.11,            # Constant parameter
    "fes_inf": 2.1,            # Constant parameter
    "fes_max": 80,             # Saturation level above which the sympathetic activity cannot increase # changed
    "fev_o": 3.2,              # Constant parameter
    "fev_inf": 6.3,            # Constant parameter
    "kes": 0.0675,             # Constant parameter
    "kev": 7.06,               # Constant parameter
    "Io_sh": 0.658,            # Value of exercise intensity at the central point of the sigmoid (heart)
    "Io_sp": 0.65,             # Value of exercise intensity at the central point of the sigmoid (peripheral resistance)
    "Io_sv": 0.45,             # Value of exercise intensity at the central point of the sigmoid (unstressed volume of veins)
    "Io_v": 0.22,             # Value of exercise intensity at the central point of the sigmoid (edited)
    "kcc_sh": 0.114,           # Parameter related to the slope of the characteristic at the central point (heart)
    "kcc_sp": 0.13,            # Parameter related to the slope of the characteristic at the central point (peripheral resistance)
    "kcc_sv": 0.09,            # Parameter related to the slope of the characteristic at the central point (unstressed volume of veins)
    "kcc_v": 0.0162,           # Parameter related to the slope of the characteristic at the central point
    "Ysh_max": 20,              # Upper saturation of the central command response (heart) # edited from 9
    "Ysh_min": -0.0283,        # Lower saturation of the central command response (heart)
    "Ysp_max": 5.5,            # Upper saturation of the central command response (peripheral resistance)
    "Ysp_min": -0.037,         # Lower saturation of the central command response (peripheral resistance)
    "Ysv_max": 64.9,           # Upper saturation of the central command response (unstressed volume of veins)
    "Ysv_min": -0.437,         # Lower saturation of the central command response (unstressed volume of veins)
    "Yv_max": 1.9,             # Upper saturation of the central command response
    "Yv_min": -0.0008,         # Lower saturation of the central command response
    "theta_v": -0.68,          # Offset term
    "Wb_sh": -1.75,            # Synaptic weight tuned to reproduce physiological results (heart)
    "Wb_sp": -1.1375,          # Synaptic weight tuned to reproduce physiological results (peripheral resistance)
    "Wb_sv": -1.1375,          # Synaptic weight tuned to reproduce physiological results (unstressed volume of veins)
    "Wc_sh": 1,                # Synaptic weight tuned to reproduce physiological results (heart)
    "Wc_sp": 1.716,            # Synaptic weight tuned to reproduce physiological results (peripheral resistance)
    "Wc_sv": 1.716,            # Synaptic weight tuned to reproduce physiological results (unstressed volume of veins)
    "Wc_v": 0.2,               # Synaptic weight tuned to reproduce physiological results
    "Wp_sh": -0.2,                # Synaptic weight tuned to reproduce physiological results (heart) # edited to match DGSM
    "Wp_sp": -0.3997,          # Synaptic weight tuned to reproduce physiological results (peripheral resistance)
    "Wp_sv": -0.3997,          # Synaptic weight tuned to reproduce physiological results (unstressed volume of veins)
    "Wp_v": -0.103,            # Synaptic weight tuned to reproduce physiological results
    "Wt_sh": 0.4,              # Synaptic weight tuned to reproduce physiological results (heart)
    "Wt_sp": 0.4,              # Synaptic weight tuned to reproduce physiological results (peripheral resistance)
    "Wt_sv": 0.4,              # Synaptic weight tuned to reproduce physiological results (unstressed volume of veins)
    "Wt_v": 0.4,
# }

    # Table 17: Effectors for Reflex Control: Resistances, Unstressed Volumes, and Cardiac Elastances
# parameters_reflex_control = {
    "DEmax_lv": 2,       # Pure latency of the mechanism
    "DEmax_rv": 2,       # Pure latency of the mechanism
    "DR_amp": 2,          # Pure latency of the mechanism
    "DR_ep": 2,           # Pure latency of the mechanism
    "DR_rmp": 2,           # Pure latency of the mechanism
    "DR_sp": 2,           # Pure latency of the mechanism
    "DV_amv": 5,        # Pure latency of the mechanism
    "DV_ev": 5,           # Pure latency of the mechanism
    "DV_rmv": 5,           # Pure latency of the mechanism
    "DV_sv": 5,           # Pure latency of the mechanism
    "Emax_lv0": 2.392,   # Basal level of maximum end-systolic elastance of the left ventricle
    "Emax_rv0": 1.412,   # Basal level of maximum end-systolic elastance of the right ventricle
    "fes_min": 2.66,     # Threshold for sympathetic stimulation
    "GEmax_lv": 0.475,   # Constant gain factor
    "GEmax_rv": 0.282,   # Constant gain factor
    "GR_amp": 4.47,       # Constant gain factor # edited
    "GR_ep": 1.94,        # Constant gain factor
    "GR_rmp": 2.47,        # Constant gain factor
    "GR_sp": 0.695,       # Constant gain factor
    "GV_amv": -28.29,     # Constant gain factor # edited
    "GV_ev": -74.21,      # Constant gain factor
    "GV_rmv": -28.29,      # Constant gain factor
    "GV_sv": -265.4,      # Constant gain factor
    "R_amp0": 3.510,     # Basal level of active skeletal peripheral resistance
    "R_ep0": 1.655,      # Basal level of extra-splanchnic peripheral resistance # edited from 1.655
    "R_rmp0": 5.270,      # Basal level of resting skeletal peripheral resistance # edited from 5.270
    "R_sp0": 2.49,       # Basal level of splanchnic peripheral resistance # edited from 2.49
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
    "Vu_amv0": 286.4,    # Basal level of active skeletal muscle venous unstressed volume
    "Vu_ev0": 607.8,     # Basal level of extra-splanchnic venous unstressed volume
    "Vu_rmv0": 190.95,    # Basal level of resting skeletal muscle venous unstressed volume
    "Vu_sv0": 961.6,    # Basal level of splanchnic venous unstressed volume
# }

# Table 18: Parameters of Effectors for Reflex Control: Heart Period
# parameters_heart_period = {
    "DT_s": 2,              # Pure latency of the mechanism
    "DT_v": 0.2,            # Pure latency of the mechanism
    # "fsh_IC": 3.8576,         # Initial condition for the efferent sympathetic cardiac activity
    # "fv_IC": 4.2748,         # Initial condition for the efferent vagal activity
    "GT_s": -0.13,          # Constant gain factor
    "GT_v": 0.09,           # Constant gain factor
    "T0": 0.58,              # Heart period in the absence of cardiac innervation # want to change to 0.83333 from 0.58
    "tau_Ts": 2,              # Time constant
    "tau_Tv": 1.5,            # Time constant
# }

# Table 19: Parameters of the Upper Airways
# parameters_upper_airways = {
#     "A0_ua": 1,               # Maximum area of opening in upper airway
#     "b_ua": 1,                 # Upper airway mechanics constant
#     "C_ua": 0.001,             # Upper airway compliance
#     "K_ua": 1,                 # Proportionality coefficient
#     "Pcrit_min": -40,         # Critical upper airway pressure
#     "R_AW": 0.82128,           # Airway wall resistance
#     "R_CW": 0.8326,            # Chest wall resistance
#     "R_L": 1.3661,             # Lung transmural resistance
#     "R_trachea": 1000000,      # Upper airway wall resistance
    "R_rs": 3.02,              # Overall resistance
# }

# Table 20: Parameters of the Pulmonary Mechanics
# parameters_pulmonary_mechanics = {
#     "E_CW": 10.545,           # Chest wall elastance (cmH2O/l) # changed
#     "E_L": 10.545,            # Lung transmural elastance (cmH2O/l) # changed
    "E_rs": 21.9, # 21.9,             # Overall elastance (cmH2O/l) # changed
    # "k_aw1": 1.85,            # Constant for upper airway pressure (cmH2O·s/l)
    # "k_aw2": 0.43,            # Constant for upper airway pressure (cmH2O·s^2/l^2)
    "P_ao": 0,                # Airway pressure (cmH2O)
# }

# Table 21: Parameters of Ventilation Controller
# parameters_ventilation_controller = {
    "GV_dead": 0.1698,        # Constant gain for dead space volume
    # "Kbg": 17.4,             # Blood gas dissociation constant
    "KcCO2": 0.2332,         # Constant gain of CO2 central chemoreceptors
    "KcMRV": 1,              # Constant gain of central response to exercise (neural drive)
    "KpCO2": 0.2025,         # Constant gain of CO2 peripheral chemoreceptors
    "KpO2": 4.72e-9,         # Constant gain of O2 peripheral chemoreceptors
    "V0_dead": 0.1587,        # Offset value of dead space volume
    "VA_rest": 0.0673,        # Basal value of alveolar ventilation
# }

# Table 22: Parameters of Breathing Pattern Optimizer
# parameters_breathing_pattern_optimizer = {
    "lambda1": 0.3,         # Weighting factor (Dimensionless) edited
    "lambda2": 0.05,        # Weighting factor (Dimensionless) edited
    "n": 1.101,              # Power index of efficiency factor (Dimensionless)
    "Pmax": 50,              # Maximum inspiratory pressure (cmH2O) edited
    "Pmax_dot": 500,        # Maximum pressure rate during inspiration (cmH2O/s)
# }

# Table 23: Parameters of the Gas Exchange and Mixing
# parameters_gas_exchange_mixing = {
#     "a1": 0.3836,           # Parameter for O2 dissociation in blood (Dimensionless)
    "a2": 1.819,            # Parameter for CO2 dissociation in blood (Dimensionless) # changed
    # "alpha1": 0.03198,      # O2 dissociation constant (mmHg^-1)
    "alpha2": 0.05591,      # CO2 dissociation constant (mmHg^-1)
    # "beta1": 0.008275,      # O2 Bohr-Haldane parameter (mmHg^-1)
    "beta2": 0.03255,       # CO2 Bohr-Haldane parameter (mmHg^-1)
    # "C1": 9,                # Max concentration of hemoglobin-bound oxygen (mmol/l)
    "C2": 87,               # Max carbon dioxide concentration (mmol/l)
    # "Pd_CO2_IC": [39.5616, 39.6736, 39.8127, 40.0061, 40.3359], # Initial CO2 dead space conditions (mmHg)
    # "Pd_O2_IC": [104.3637, 104.2258, 104.0505, 103.8005, 103.3579], # Initial O2 dead space conditions (mmHg)
    "Fi_CO2": 0.0421,       # Inspired fraction of CO2 (%)
    "Fi_O2": 21.0379,       # Inspired fraction of O2 (%)
    "K1": 13,               # Parameter for O2 dissociation equation (mmHg)
    "K2": 194.4,            # Parameter for CO2 dissociation equation (mmHg)
    # "LCTV": 0.588,          # Lung to chemoreceptor vascular volume constant (l)
    "PACO2_Delay_IC": 40.4448,      # Initial CO2 convection delay (mmHg)
    # "dPa_CO2_dt_IC": -0.2465,       # Initial CO2 rate of change (mmHg/s)
    # "PACO2_IC": 40.9432,            # Initial Condition for CO2 convection (mmHg)
    # "d2Pa_CO2_dt2_IC": 40.3928,   # Second order CO2 rate of change (mmHg/s^2)
    "PAO2_Delay_IC": 103.1223,      # Initial O2 convection delay (mmHg)
    # "dPa_O2_dt_IC": 0.3557,         # Initial O2 rate of change (mmHg/s)
    # "PAO2_IC": 102.5153,            # Initial Condition for O2 convection (mmHg)
    # "d2Pa_O2_dt2_IC": 103.1435,   # Second order O2 rate of change (mmHg/s^2)
    "P_atm": 760,                    # Atmospheric pressure (mmHg) # CHANGED
    "P_ws": 47,                      # Water vapor pressure (mmHg)
    "T1": 1,                        # Time constant for cardiovascular mixing (s)
    "T2": 2,                        # Time constant for cardiovascular mixing (s)
    "VL_CO2": 3,                    # Lungs storage volume for CO2 (l)
    "VL_O2": 2.5,                   # Lungs storage volume for O2 (l)
    "Z": 0.0227,                    # Molar conversion factor (l/mmol)
    "VB": 0.9,                      # Gas volume in brain (L)
# }

# Table 24: Parameters of the Brain Compartment
# parameters_brain_compartment = {
    "dc": 0.015,                    # Depth of central receptor below medulla surface (cm)
    # "h": 0.0183/1000,                    # Cerebral blood flow constant (ml/(100g·s))
    "KCCO2": 346000,                # CO2 central receptor constant (s·cm^-2·l^-1)
    "KCSFCO2": 20,                 # CO2 diffusion time constant in cerebrospinal fluid (s), changed to be faster to see if limit cycle is reached
    "MRBCO2": 0.0009,               # Metabolic production rate of CO2 (1/s STPD)
    "MRBO2": 0.000925,              # Metabolic production rate of O2 (1/s STPD)
    # "PbCO2IC": 48.5338,             # Initial condition for brain CO2 partial pressure (mmHg)
    # "SbCO2": 0.36/1000,                  # Dissociation slope for CO2 in the brain (ml·(100g·y)/mmHg) # convert to L
    # "SCO2": 0.0043,                 # Dissociation slope for CO2 in blood (mmHg^-1)
# }

# Table 25: Parameters of the Gas Transport: Body Tissues Compartment
# parameters_body_tissues_compartment = {
#     "Cv_CO2_IC": 0.5247,            # Initial mixed venous CO2 concentration (ml/ml)
#     "Cv_O2_IC": 0.1639,             # Initial mixed venous O2 concentration (ml/ml)
    "MRCO2": 0.2/60,               # Minimum metabolic production rate for CO2 (l/min STPD)
    "MRO2": 0.25/60,               # Minimum metabolic consumption rate for O2 (l/min STPD) brain O2 minused in gas exchange
    "MRTCO2_basal": 0.2/60,            # Basal metabolic production rate for CO2 (l/min STPD)
    "MRTO2_basal": 0.25/60,            # Basal metabolic consumption rate for O2 (l/min STPD)
    "tauMR": 50,                    # Metabolic rate time constant (s)
    "VTCO2": 0.25,                    # Body tissue storage volume for CO2 (l) # changed from 15
    "VTO2": 0.25,                      # Body tissue storage volume for O2 (l) # changed from 6
# }

# Table 26: Gas Transport: Metabolism Dynamic
# gas_transport = {
    "tau_MRV": 50,            # Metabolic rate time constant (s)

# added params
    "Kp_ao": 1000,
    "Kf_ao": 5000,
    "Kb_ao": 2,
    "Kv_ao": 5,
#     "Kp_ao": 500,
#     "Kf_ao": 50,
#     "Kb_ao": 2,
#     "Kv_ao": 4,
    "theta_ao_max": 1.309,
    "Kp_mi": 3000,
    "Kf_mi": 500,
    "Kb_mi": 2,
    "Kv_mi": 7,
    "theta_mi_max": 1.309,
    "Kp_po": 3000,
    "Kf_po": 2000,
    "Kb_po": 5,
    "Kv_po": 10,
    "theta_po_max": 1.309,
    # "Kp_tr": 200,
    # "Kf_tr": 1000,
    # "Kb_tr": 1,
    # "Kv_tr": 7,
    "Kp_tr": 3000,
    "Kf_tr": 500,
    "Kb_tr": 2,
    "Kv_tr": 7,
    "theta_tr_max": 1.309,
    "alpha_O2": 0.0000317,
    "R_po": 350,
    "R_mi": 350,
    "R_tr": 350,
    "R_ao": 350,
    "C_O2_param1": 0.00134,
    "C_O2_param2": 2.6,
    "C_O2_param3": 3.03e-5,
    "PAMO2_nominal": 104,
    "scale_param1": 4.9,
    "scale_param2": 1.5,
    "scale_param3": 0.3,
    "scale_param4": 26.6,
    "scale_param5": 0.5,
    "scale_param6": 1.2,
    "scale_param7": 30,
    "scale_param8": 1.6,
    "shift_param1": 4,
    "shift_param2": 0.3,
    "shift_param3": 4,
    "shift_param4": 0.3,
    "Pa_O2_lower": 80,
    "rise_time_atr": 0.05,
    "fall_time_atr": 0.2,
    "rise_time_ven": 0.15,
    "fall_time_ven": 0.3,
    "ahead1": 0.9,
    "theta_min": 0.0872665,
    "delta_P": 0.05,
}
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 94.323814, 'C_O2_param1': 0.0015608842, 'C_O2_param2': 3.5407846, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 12.6021595, 'C_hv': 1.57, 'C_jp': 5.1168885, 'C_pa': 0.76, 'C_pp': 21.228539, 'C_pv': 25.669655, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 25.770163, 'Cvam_O2_n': 0.08815333, 'Cvb_O2_n': 0.09052261, 'Cvh_O2_n': 0.057520334, 'Cvrm_O2_n': 0.20011526, 'D': -7.3836823, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 13.06373, 'Emax_la': 0.35, 'Emax_lv0': 0.87741095, 'Emax_ra': 0.35, 'Emax_rv0': 0.40517095, 'Fi_CO2': 0.0421, 'Fi_O2': 30.029568, 'GEmax_lv': 0.24551241, 'GEmax_rv': 0.273231, 'GR_amp': 4.47, 'GR_ep': 2.6729898, 'GR_rmp': 2.47, 'GR_sp': 0.7780109, 'GT_s': -0.11474573, 'GT_v': 0.12126079, 'GV_amv': -28.29, 'GV_dead': 0.2471073, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -263.48398, 'G_ap': 13.952332, 'Io_met': 0.42811677, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.5851593, 'Io_v': 0.22, 'K1_vc': 0.16082393, 'K2': 178.99307, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.06445175, 'KE_lv': 0.010495906, 'KE_ra': 0.0708307, 'KE_rv': 0.011586004, 'K_H': 3.6883354, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.11893098, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.15837711, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 1785.5186, 'Kp_tr': 130.59839, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 8.27662, 'Kv_tr': 3.8860948, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.45851168, 'MO2_bp': 1.0967921, 'MO2_hpn': 0.4, 'MO2_rmp': 0.5043618, 'P0_la': 0.55, 'P0_lv': 1.6788163, 'P0_ra': 0.55, 'P0_rv': 1.4722053, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 96.13526, 'P_n_max': 112.0, 'P_thormax_n': -4.0, 'P_thormin_n': -4, 'PaCO2_n': 36.039318, 'PaO2_ac_n': 35.239395, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 4.7661476, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 6.312542, 'R_bv_n': 0.075, 'R_ep0': 2.3037422, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.24469315, 'R_mi': 246.57578, 'R_pa': 0.033236817, 'R_po': 378.93, 'R_pp': 0.10952646, 'R_pv': 0.09700368, 'R_rmp0': 4.5679336, 'R_rmv_n': 0.125, 'R_rs': 3.562436, 'R_sa': 0.07515636, 'R_sp0': 3.6745343, 'R_sv_n': 0.038, 'R_tr': 195.38086, 'Rvc_n': 0.06512741, 'T0': 0.40323877, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.1136509, 'Ta': 2.0, 'Tc': 0.3979315, 'V0_dead': 0.19210498, 'VA_rest': 0.06628334, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 4818.3115, 'Vu_amv0': 214.70573, 'Vu_bv': 386.22208, 'Vu_ev0': 602.7552, 'Vu_hv': 49.593784, 'Vu_jp': 489.68472, 'Vu_la': 4.0, 'Vu_lv': 16.171349, 'Vu_pa': 1.0, 'Vu_pp': 134.53569, 'Vu_pv': 165.71053, 'Vu_ra': 4.0, 'Vu_rmv0': 125.05529, 'Vu_rv': 52.665028, 'Vu_sa': 1.0, 'Vu_sv0': 885.3988, 'Vu_vc': 142.35129, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.7213833, 'Wb_sp': -1.035797, 'Wb_sv': -1.5311289, 'Wc_sh': 0.800884, 'Wc_sp': 2.3501709, 'Wc_sv': 1.716, 'Wc_v': 0.2924614, 'Wp_sh': -0.20147176, 'Wp_sp': -0.3997, 'Wp_sv': -0.52766645, 'Wp_v': -0.08721046, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 70.47197, 'Ysv_min': -0.4764506, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.6896983, 'ahead1': 0.8807701, 'alpha2': 0.07291157, 'alpha_O2': 3.17e-05, 'beta2': 0.021709256, 'dc': 0.015, 'delta_P': 0.4173715, 'f_ab_max': 35.04449, 'f_ab_min': 2.6098804, 'f_acCO2_n': 1.0468378, 'f_ac_max': 9.138153, 'f_ac_min': 0.6669395, 'fab_o': 28.704922, 'fall_time_atr': 0.1, 'fall_time_ven': 0.2542506, 'fes_inf': 1.6007478, 'fes_max': 80.0, 'fes_min': 3.0173924, 'fes_o': 23.786322, 'fev_inf': 6.0673943, 'fev_o': 2.1744254, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 1.4082268, 'g_ccsp': 2.1704328, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 14.526971, 'gh_O2': 35.0, 'grm_O2': 16.781862, 'k_ab': 15.048384, 'k_ac': 34.63521, 'kcc_sh': 0.12545459, 'kcc_sp': 0.13, 'kcc_sv': 0.10884218, 'kcc_v': 0.0162, 'kes': 0.0984378, 'kev': 9.713776, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.25839624, 'kr_am': 24.17, 'phi_max': 28.367245, 'phi_min': -2.6751063, 'rise_time_atr': 0.05, 'rise_time_ven': 0.15850285, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.6444576, 'scale_param3': 0.2020339, 'scale_param4': 20.790846, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.7610633, 'theta_min': 0.11597809, 'theta_po_max': 0.9498662, 'theta_shn': 3.5987911, 'theta_spn': 9.053085, 'theta_svn': 17.63625, 'theta_tr_max': 1.4860394, 'theta_v': -0.96205395, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}
# Slide 179
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 80.76854, 'C_O2_param1': 0.0016280396, 'C_O2_param2': 2.5562308, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 5.8942447, 'C_hv': 1.57, 'C_jp': 2.9988003, 'C_pa': 0.76, 'C_pp': 15.75479, 'C_pv': 22.739174, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 17.608536, 'Cvam_O2_n': 0.095513925, 'Cvb_O2_n': 0.087925166, 'Cvh_O2_n': 0.10401794, 'Cvrm_O2_n': 0.10403119, 'D': -4.0966907, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 17.581713, 'Emax_la': 0.35, 'Emax_lv0': 1.5210254, 'Emax_ra': 0.35, 'Emax_rv0': 0.6959497, 'Fi_CO2': 0.0421, 'Fi_O2': 23.213263, 'GEmax_lv': 0.5452949, 'GEmax_rv': 0.17742495, 'GR_amp': 4.47, 'GR_ep': 1.1082402, 'GR_rmp': 2.47, 'GR_sp': 0.73157346, 'GT_s': -0.16193494, 'GT_v': 0.07344954, 'GV_amv': -28.29, 'GV_dead': 0.15540025, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -258.99387, 'G_ap': 6.633357, 'Io_met': 0.30002865, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.24283352, 'Io_v': 0.22, 'K1_vc': 0.18483122, 'K2': 175.61687, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.06725279, 'KE_lv': 0.008432211, 'KE_ra': 0.057989143, 'KE_rv': 0.0060636434, 'K_H': 4.0538077, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.17710899, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.18988557, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 1770.57, 'Kp_tr': 56.63762, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 5.6740303, 'Kv_tr': 8.524703, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.57756364, 'MO2_bp': 0.56853515, 'MO2_hpn': 0.4, 'MO2_rmp': 0.9713148, 'P0_la': 0.55, 'P0_lv': 1.2341449, 'P0_ra': 0.55, 'P0_rv': 1.5433319, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 91.78269, 'P_n_max': 112.0, 'P_thormax_n': -4.0, 'P_thormin_n': -4, 'PaCO2_n': 39.22745, 'PaO2_ac_n': 59.572975, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 4.25357, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 7.1840553, 'R_bv_n': 0.075, 'R_ep0': 1.6321366, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.17849647, 'R_mi': 413.2936, 'R_pa': 0.012838589, 'R_po': 340.87598, 'R_pp': 0.05801987, 'R_pv': 0.10470716, 'R_rmp0': 6.7150893, 'R_rmv_n': 0.125, 'R_rs': 2.9914567, 'R_sa': 0.055540897, 'R_sp0': 3.1551092, 'R_sv_n': 0.038, 'R_tr': 194.17595, 'Rvc_n': 0.0571389, 'T0': 0.8682399, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.4489796, 'Ta': 2.0, 'Tc': 0.7676254, 'V0_dead': 0.22848578, 'VA_rest': 0.06632323, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 4801.8657, 'Vu_amv0': 327.16235, 'Vu_bv': 327.64154, 'Vu_ev0': 589.28424, 'Vu_hv': 117.40228, 'Vu_jp': 392.97214, 'Vu_la': 4.0, 'Vu_lv': 23.79085, 'Vu_pa': 1.0, 'Vu_pp': 132.71786, 'Vu_pv': 99.86373, 'Vu_ra': 4.0, 'Vu_rmv0': 126.17735, 'Vu_rv': 39.975357, 'Vu_sa': 1.0, 'Vu_sv0': 1498.2147, 'Vu_vc': 169.69576, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.7166406, 'Wb_sp': -1.0694853, 'Wb_sv': -1.2029309, 'Wc_sh': 0.6392687, 'Wc_sp': 1.9398042, 'Wc_sv': 1.716, 'Wc_v': 0.18703963, 'Wp_sh': -0.2733672, 'Wp_sp': -0.3997, 'Wp_sv': -0.49786747, 'Wp_v': -0.14651664, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 38.813023, 'Ysv_min': -0.64345855, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.6373777, 'ahead1': 0.9047692, 'alpha2': 0.07542447, 'alpha_O2': 3.17e-05, 'beta2': 0.030226678, 'dc': 0.015, 'delta_P': 0.41370863, 'f_ab_max': 37.89126, 'f_ab_min': 2.8646665, 'f_acCO2_n': 0.8779206, 'f_ac_max': 18.335514, 'f_ac_min': 1.0464523, 'fab_o': 13.340106, 'fall_time_atr': 0.1, 'fall_time_ven': 0.30860725, 'fes_inf': 3.0327964, 'fes_max': 80.0, 'fes_min': 1.7915539, 'fes_o': 18.716782, 'fev_inf': 6.5948324, 'fev_o': 2.9958644, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 0.58243525, 'g_ccsp': 1.8800584, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 8.701386, 'gh_O2': 35.0, 'grm_O2': 33.27961, 'k_ab': 14.092456, 'k_ac': 35.695946, 'kcc_sh': 0.14656834, 'kcc_sp': 0.13, 'kcc_sv': 0.082692236, 'kcc_v': 0.0162, 'kes': 0.07129275, 'kev': 6.9514503, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.2364808, 'kr_am': 24.17, 'phi_max': 14.854409, 'phi_min': -2.5393617, 'rise_time_atr': 0.05, 'rise_time_ven': 0.15581879, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 2.122722, 'scale_param3': 0.30646732, 'scale_param4': 17.770975, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.7665671, 'theta_min': 0.052193936, 'theta_po_max': 0.8066221, 'theta_shn': 2.2286088, 'theta_spn': 11.596245, 'theta_svn': 12.99261, 'theta_tr_max': 0.85718316, 'theta_v': -0.48158357, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 85.10838, 'C_O2_param1': 0.0015223391, 'C_O2_param2': 3.883741, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 14.493739, 'C_hv': 1.57, 'C_jp': 5.1864114, 'C_pa': 0.76, 'C_pp': 15.0610485, 'C_pv': 18.667507, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 45.452503, 'Cvam_O2_n': 0.20636033, 'Cvb_O2_n': 0.13687047, 'Cvh_O2_n': 0.14720291, 'Cvrm_O2_n': 0.13442713, 'D': -3.1749704, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 14.239739, 'Emax_la': 0.35, 'Emax_lv0': 0.70103645, 'Emax_ra': 0.35, 'Emax_rv0': 0.7715704, 'Fi_CO2': 0.0421, 'Fi_O2': 22.480019, 'GEmax_lv': 0.39943781, 'GEmax_rv': 0.31339112, 'GR_amp': 4.47, 'GR_ep': 2.0076497, 'GR_rmp': 2.47, 'GR_sp': 0.6877934, 'GT_s': -0.16471796, 'GT_v': 0.0747155, 'GV_amv': -28.29, 'GV_dead': 0.16088337, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -222.24026, 'G_ap': 7.0848827, 'Io_met': 0.37183547, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.39252296, 'Io_v': 0.22, 'K1_vc': 0.12724183, 'K2': 211.37552, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.060274694, 'KE_lv': 0.013556033, 'KE_ra': 0.07232089, 'KE_rv': 0.0108511355, 'K_H': 3.6826472, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.18964656, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.15442632, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 4160.3076, 'Kp_tr': 112.9173, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 5.557269, 'Kv_tr': 4.8010945, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.34565225, 'MO2_bp': 0.5015912, 'MO2_hpn': 0.4, 'MO2_rmp': 1.2711611, 'P0_la': 0.55, 'P0_lv': 1.3452486, 'P0_ra': 0.55, 'P0_rv': 1.5237777, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 89.31407, 'P_n_max': 112.0, 'P_thormax_n': -4.0, 'P_thormin_n': -4, 'PaCO2_n': 43.698254, 'PaO2_ac_n': 51.393208, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 2.7897751, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 8.686378, 'R_bv_n': 0.075, 'R_ep0': 1.5203091, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.17502481, 'R_mi': 297.11093, 'R_pa': 0.02829779, 'R_po': 177.51392, 'R_pp': 0.050717693, 'R_pv': 0.0862459, 'R_rmp0': 4.7436166, 'R_rmv_n': 0.125, 'R_rs': 2.967241, 'R_sa': 0.05421166, 'R_sp0': 2.3786528, 'R_sv_n': 0.038, 'R_tr': 385.55704, 'Rvc_n': 0.07383141, 'T0': 0.7268808, 'T1': 1.0, 'T2': 2.0, 'T_im': 0.72233367, 'Ta': 2.0, 'Tc': 0.83903265, 'V0_dead': 0.14957346, 'VA_rest': 0.076326676, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 4323.253, 'Vu_amv0': 219.10909, 'Vu_bv': 208.79568, 'Vu_ev0': 712.65674, 'Vu_hv': 126.25206, 'Vu_jp': 325.64365, 'Vu_la': 4.0, 'Vu_lv': 11.560448, 'Vu_pa': 1.0, 'Vu_pp': 114.36946, 'Vu_pv': 121.49066, 'Vu_ra': 4.0, 'Vu_rmv0': 264.2398, 'Vu_rv': 53.62917, 'Vu_sa': 1.0, 'Vu_sv0': 1087.4319, 'Vu_vc': 118.8508, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -2.2288718, 'Wb_sp': -1.4646281, 'Wb_sv': -1.6313516, 'Wc_sh': 1.4993825, 'Wc_sp': 2.4375226, 'Wc_sv': 1.716, 'Wc_v': 0.19526505, 'Wp_sh': -0.26044932, 'Wp_sp': -0.3997, 'Wp_sv': -0.2368723, 'Wp_v': -0.13257082, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 88.73009, 'Ysv_min': -0.297605, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.8852309, 'ahead1': 0.80531275, 'alpha2': 0.07650908, 'alpha_O2': 3.17e-05, 'beta2': 0.044033743, 'dc': 0.015, 'delta_P': 0.43999022, 'f_ab_max': 54.645206, 'f_ab_min': 3.4100587, 'f_acCO2_n': 0.98936963, 'f_ac_max': 13.542758, 'f_ac_min': 0.7584193, 'fab_o': 15.290993, 'fall_time_atr': 0.1, 'fall_time_ven': 0.40928057, 'fes_inf': 2.018002, 'fes_max': 80.0, 'fes_min': 2.0902553, 'fes_o': 20.04489, 'fev_inf': 8.695317, 'fev_o': 4.55201, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 0.84645844, 'g_ccsp': 1.7608838, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 6.254319, 'gh_O2': 35.0, 'grm_O2': 32.377117, 'k_ab': 14.84149, 'k_ac': 26.998575, 'kcc_sh': 0.1123353, 'kcc_sp': 0.13, 'kcc_sv': 0.0982813, 'kcc_v': 0.0162, 'kes': 0.099026084, 'kev': 9.373619, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.22532362, 'kr_am': 24.17, 'phi_max': 28.00714, 'phi_min': -2.6058562, 'rise_time_atr': 0.05, 'rise_time_ven': 0.22144873, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.23304, 'scale_param3': 0.17774893, 'scale_param4': 28.343447, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.2155957, 'theta_min': 0.123867124, 'theta_po_max': 1.6611198, 'theta_shn': 3.4649527, 'theta_spn': 7.333361, 'theta_svn': 19.65172, 'theta_tr_max': 1.1121043, 'theta_v': -0.7006232, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}

# nominal parameter set for input into lhcs emulator training
# Parameters = {'A': 20.9, 'AT': 0.016666666666666666, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 87.0, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.6, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 10.0, 'C_hv': 1.57, 'C_jp': 3.72, 'C_pa': 0.76, 'C_pp': 15.8, 'C_pv': 25.37, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 31.11, 'Cvam_O2_n': 0.1555, 'Cvb_O2_n': 0.14, 'Cvh_O2_n': 0.11, 'Cvrm_O2_n': 0.155, 'D': -5.251, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 21.9, 'Emax_la': 0.35, 'Emax_lv0': 1.4, 'Emax_ra': 0.35, 'Emax_rv0': 0.7, 'Fi_CO2': 0.0421, 'Fi_O2': 21.0379, 'GEmax_lv': 0.475, 'GEmax_rv': 0.282, 'GR_amp': 4.47, 'GR_ep': 1.94, 'GR_rmp': 2.47, 'GR_sp': 0.695, 'GT_s': -0.13, 'GT_v': 0.09, 'GV_amv': -28.29, 'GV_dead': 0.1698, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -265.4, 'G_ap': 11.76, 'Io_met': 0.4266, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.45, 'Io_v': 0.22, 'K1_vc': 0.15, 'K2': 194.40000000000003, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.05, 'KE_lv': 0.014, 'KE_ra': 0.05, 'KE_rv': 0.011, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.2332, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.2025, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 3000.0, 'Kp_tr': 100.0, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 10.0, 'Kv_tr': 7.0, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.516, 'MO2_bp': 0.925, 'MO2_hpn': 0.4, 'MO2_rmp': 0.86, 'P0_la': 0.55, 'P0_lv': 1.5, 'P0_ra': 0.55, 'P0_rv': 1.5, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 92.0, 'P_n_max': 112.0, 'P_thormax_n': -2.0, 'P_thormin_n': -6.0, 'PaCO2_n': 40.0, 'PaO2_ac_n': 45.0, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 3.51, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 6.57, 'R_bv_n': 0.075, 'R_ep0': 1.655, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.224, 'R_mi': 350.0, 'R_pa': 0.023, 'R_po': 350.0, 'R_pp': 0.0894, 'R_pv': 0.1, 'R_rmp0': 5.27, 'R_rmv_n': 0.125, 'R_rs': 3.02, 'R_sa': 0.06, 'R_sp0': 2.49, 'R_sv_n': 0.038, 'R_tr': 350.0, 'Rvc_n': 0.05, 'T0': 0.58, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.1, 'Ta': 5.0, 'Tc': 0.7, 'V0_dead': 0.1587, 'VA_rest': 0.067, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 5027.6, 'Vu_amv0': 286.4, 'Vu_bv': 279.49, 'Vu_ev0': 607.8, 'Vu_hv': 93.16, 'Vu_jp': 579.76, 'Vu_la': 4.0, 'Vu_lv': 15.908, 'Vu_pa': 1.0, 'Vu_pp': 116.6775, 'Vu_pv': 114.0, 'Vu_ra': 4.0, 'Vu_rmv0': 190.95, 'Vu_rv': 38.703, 'Vu_sa': 1.0, 'Vu_sv0': 1361.6, 'Vu_vc': 123.0, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.75, 'Wb_sp': -1.1375, 'Wb_sv': -1.1375, 'Wc_sh': 1.0, 'Wc_sp': 1.716, 'Wc_sv': 1.716, 'Wc_v': 0.2, 'Wp_sh': -0.2, 'Wp_sp': -0.3997, 'Wp_sv': -0.3997, 'Wp_v': -0.103, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 64.9, 'Ysv_min': -0.437, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.819, 'ahead1': 0.8, 'alpha2': 0.05591, 'alpha_O2': 3.17e-05, 'beta2': 0.03255, 'dc': 0.015, 'delta_P': 0.3, 'f_ab_max': 47.78, 'f_ab_min': 2.52, 'f_acCO2_n': 1.4, 'f_ac_max': 12.3, 'f_ac_min': 0.835, 'fab_o': 25.0, 'fall_time_atr': 0.1, 'fall_time_ven': 0.3, 'fes_inf': 2.1, 'fes_max': 80.0, 'fes_min': 2.66, 'fes_o': 16.11, 'fev_inf': 6.3, 'fev_o': 3.2, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 1.0, 'g_ccsp': 1.5, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 10.0, 'gh_O2': 35.0, 'grm_O2': 30.0, 'k_ab': 11.76, 'k_ac': 29.27, 'kcc_sh': 0.114, 'kcc_sp': 0.13, 'kcc_sv': 0.09, 'kcc_v': 0.0162, 'kes': 0.0675, 'kev': 7.06, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.18, 'kr_am': 24.17, 'phi_max': 20.0, 'phi_min': -1.87, 'rise_time_atr': 0.05, 'rise_time_ven': 0.15, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.5, 'scale_param3': 0.3, 'scale_param4': 26.6, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.309, 'theta_min': 0.0872665, 'theta_po_max': 1.309, 'theta_shn': 3.6, 'theta_spn': 13.32, 'theta_svn': 13.32, 'theta_tr_max': 1.309, 'theta_v': -0.68, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}

# checking physiological
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 87.0, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.6, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 13.892639, 'C_hv': 1.57, 'C_jp': 4.542198, 'C_pa': 0.76, 'C_pp': 8.940501, 'C_pv': 13.178656, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 42.11085, 'Cvam_O2_n': 0.21954548, 'Cvb_O2_n': 0.17397921, 'Cvh_O2_n': 0.07200153, 'Cvrm_O2_n': 0.11212631, 'D': -3.1803095, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 21.9, 'Emax_la': 0.35, 'Emax_lv0': 1.5310963, 'Emax_ra': 0.35, 'Emax_rv0': 0.45778954, 'Fi_CO2': 0.0421, 'Fi_O2': 21.0379, 'GEmax_lv': 0.44759282, 'GEmax_rv': 0.31378803, 'GR_amp': 4.47, 'GR_ep': 2.4702032, 'GR_rmp': 2.47, 'GR_sp': 0.96651375, 'GT_s': -0.08758687, 'GT_v': 0.11623258, 'GV_amv': -28.29, 'GV_dead': 0.1698, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -224.61212, 'G_ap': 8.381655, 'Io_met': 0.30170512, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.3332137, 'Io_v': 0.22, 'K1_vc': 0.22498398, 'K2': 194.4, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.05560146, 'KE_lv': 0.013733092, 'KE_ra': 0.06393655, 'KE_rv': 0.010777749, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.2332, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.2025, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 2927.5989, 'Kp_tr': 79.8303, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 8.837868, 'Kv_tr': 5.9578586, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.67558056, 'MO2_bp': 1.2787504, 'MO2_hpn': 0.4, 'MO2_rmp': 0.7690501, 'P0_la': 0.55, 'P0_lv': 0.8228914, 'P0_ra': 0.55, 'P0_rv': 1.0831838, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 88.1423, 'P_n_max': 112.0, 'P_thormax_n': -2.0, 'P_thormin_n': -2.0, 'PaCO2_n': 40.0, 'PaO2_ac_n': 45.0, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 3.6612408, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 6.5594926, 'R_bv_n': 0.075, 'R_ep0': 2.2719216, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.18640628, 'R_mi': 372.25244, 'R_pa': 0.015163942, 'R_po': 443.8721, 'R_pp': 0.11075763, 'R_pv': 0.05409203, 'R_rmp0': 2.7399774, 'R_rmv_n': 0.125, 'R_rs': 3.02, 'R_sa': 0.06469082, 'R_sp0': 2.3793685, 'R_sv_n': 0.038, 'R_tr': 286.1981, 'Rvc_n': 0.07359674, 'T0': 0.7549434, 'T1': 1.0, 'T2': 2.0, 'T_im': 0.58490735, 'Ta': 2.0, 'Tc': 0.66407293, 'V0_dead': 0.1587, 'VA_rest': 0.067, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 5116.083, 'Vu_amv0': 382.4417, 'Vu_bv': 237.20724, 'Vu_ev0': 662.9004, 'Vu_hv': 112.61012, 'Vu_jp': 494.64233, 'Vu_la': 4.0, 'Vu_lv': 22.43135, 'Vu_pa': 1.0, 'Vu_pp': 153.26457, 'Vu_pv': 60.825928, 'Vu_ra': 4.0, 'Vu_rmv0': 268.21466, 'Vu_rv': 43.780434, 'Vu_sa': 1.0, 'Vu_sv0': 1566.4963, 'Vu_vc': 125.19409, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.7293574, 'Wb_sp': -0.71830624, 'Wb_sv': -1.0223228, 'Wc_sh': 0.68143606, 'Wc_sp': 1.921212, 'Wc_sv': 1.716, 'Wc_v': 0.2722284, 'Wp_sh': -0.2877577, 'Wp_sp': -0.3997, 'Wp_sv': -0.3826874, 'Wp_v': -0.08776803, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 80.70727, 'Ysv_min': -0.59853643, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.819, 'ahead1': 0.9015429, 'alpha2': 0.05591, 'alpha_O2': 3.17e-05, 'beta2': 0.03255, 'dc': 0.015, 'delta_P': 0.19487058, 'f_ab_max': 67.93988, 'f_ab_min': 2.6202273, 'f_acCO2_n': 1.3399563, 'f_ac_max': 9.862197, 'f_ac_min': 0.82340735, 'fab_o': 21.19918, 'fall_time_atr': 0.1, 'fall_time_ven': 0.26389775, 'fes_inf': 1.3142627, 'fes_max': 80.0, 'fes_min': 1.7923621, 'fes_o': 16.384644, 'fev_inf': 7.162251, 'fev_o': 4.7826858, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 0.6245575, 'g_ccsp': 1.7971339, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 12.4334545, 'gh_O2': 35.0, 'grm_O2': 18.733038, 'k_ab': 16.388168, 'k_ac': 41.934887, 'kcc_sh': 0.09189838, 'kcc_sp': 0.13, 'kcc_sv': 0.13114133, 'kcc_v': 0.0162, 'kes': 0.088870525, 'kev': 5.3482375, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.13250487, 'kr_am': 24.17, 'phi_max': 23.55283, 'phi_min': -2.201891, 'rise_time_atr': 0.05, 'rise_time_ven': 0.17267132, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.955512, 'scale_param3': 0.3, 'scale_param4': 26.6, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.7448641, 'theta_min': 0.11986012, 'theta_po_max': 1.4636664, 'theta_shn': 3.8263187, 'theta_spn': 7.163615, 'theta_svn': 10.237498, 'theta_tr_max': 0.8199331, 'theta_v': -0.65057033, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 87.0, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.6, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 13.163554, 'C_hv': 1.57, 'C_jp': 4.61329, 'C_pa': 0.76, 'C_pp': 18.490892, 'C_pv': 33.993782, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 36.278053, 'Cvam_O2_n': 0.21129906, 'Cvb_O2_n': 0.07232412, 'Cvh_O2_n': 0.14000537, 'Cvrm_O2_n': 0.1007577, 'D': -3.0938704, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 21.9, 'Emax_la': 0.35, 'Emax_lv0': 1.7550619, 'Emax_ra': 0.35, 'Emax_rv0': 0.65024006, 'Fi_CO2': 0.0421, 'Fi_O2': 21.0379, 'GEmax_lv': 0.43845433, 'GEmax_rv': 0.25649977, 'GR_amp': 4.47, 'GR_ep': 1.4715011, 'GR_rmp': 2.47, 'GR_sp': 0.9345286, 'GT_s': -0.113909245, 'GT_v': 0.062487748, 'GV_amv': -28.29, 'GV_dead': 0.1698, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -229.70033, 'G_ap': 9.616291, 'Io_met': 0.43808952, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.5942973, 'Io_v': 0.22, 'K1_vc': 0.18734257, 'K2': 194.4, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.06065554, 'KE_lv': 0.013143627, 'KE_ra': 0.055604097, 'KE_rv': 0.007842522, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.2332, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.2025, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 2006.6179, 'Kp_tr': 84.77101, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 11.842118, 'Kv_tr': 4.8609242, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.41274327, 'MO2_bp': 1.2287687, 'MO2_hpn': 0.4, 'MO2_rmp': 0.6129666, 'P0_la': 0.55, 'P0_lv': 1.3667512, 'P0_ra': 0.55, 'P0_rv': 1.6101573, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 88.23875, 'P_n_max': 112.0, 'P_thormax_n': -2.0, 'P_thormin_n': -2.0, 'PaCO2_n': 40.0, 'PaO2_ac_n': 45.0, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 2.3477528, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 7.3956084, 'R_bv_n': 0.075, 'R_ep0': 2.269029, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.16626075, 'R_mi': 337.31006, 'R_pa': 0.028598027, 'R_po': 331.3358, 'R_pp': 0.062099658, 'R_pv': 0.06196163, 'R_rmp0': 3.5504122, 'R_rmv_n': 0.125, 'R_rs': 3.02, 'R_sa': 0.046750594, 'R_sp0': 3.4472036, 'R_sv_n': 0.038, 'R_tr': 452.78085, 'Rvc_n': 0.052481472, 'T0': 0.8655106, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.3889598, 'Ta': 2.0, 'Tc': 0.4495121, 'V0_dead': 0.1587, 'VA_rest': 0.067, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 4923.0767, 'Vu_amv0': 166.87639, 'Vu_bv': 166.44005, 'Vu_ev0': 811.4276, 'Vu_hv': 134.33644, 'Vu_jp': 586.95026, 'Vu_la': 4.0, 'Vu_lv': 15.537728, 'Vu_pa': 1.0, 'Vu_pp': 64.30236, 'Vu_pv': 148.93652, 'Vu_ra': 4.0, 'Vu_rmv0': 150.31042, 'Vu_rv': 47.915684, 'Vu_sa': 1.0, 'Vu_sv0': 1721.978, 'Vu_vc': 61.57784, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.2398196, 'Wb_sp': -0.75327176, 'Wb_sv': -0.8139902, 'Wc_sh': 0.7834839, 'Wc_sp': 1.4700422, 'Wc_sv': 1.716, 'Wc_v': 0.123167425, 'Wp_sh': -0.124091074, 'Wp_sp': -0.3997, 'Wp_sv': -0.25713247, 'Wp_v': -0.12623586, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 74.5403, 'Ysv_min': -0.6074521, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.819, 'ahead1': 0.8752577, 'alpha2': 0.05591, 'alpha_O2': 3.17e-05, 'beta2': 0.03255, 'dc': 0.015, 'delta_P': 0.3102968, 'f_ab_max': 63.705585, 'f_ab_min': 2.202113, 'f_acCO2_n': 0.9538434, 'f_ac_max': 16.444267, 'f_ac_min': 1.1572869, 'fab_o': 17.233936, 'fall_time_atr': 0.1, 'fall_time_ven': 0.3788514, 'fes_inf': 2.125986, 'fes_max': 80.0, 'fes_min': 1.9882374, 'fes_o': 23.263817, 'fev_inf': 9.041499, 'fev_o': 2.4199934, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 0.62262034, 'g_ccsp': 1.1561137, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 13.5231905, 'gh_O2': 35.0, 'grm_O2': 20.970594, 'k_ab': 9.870779, 'k_ac': 37.915375, 'kcc_sh': 0.11552594, 'kcc_sp': 0.13, 'kcc_sv': 0.12216191, 'kcc_v': 0.0162, 'kes': 0.08549118, 'kev': 3.6128669, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.12688, 'kr_am': 24.17, 'phi_max': 20.319191, 'phi_min': -2.543785, 'rise_time_atr': 0.05, 'rise_time_ven': 0.15121184, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.2005804, 'scale_param3': 0.3, 'scale_param4': 26.6, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 0.83195937, 'theta_min': 0.120470606, 'theta_po_max': 1.2478832, 'theta_shn': 2.826099, 'theta_spn': 12.036194, 'theta_svn': 14.867234, 'theta_tr_max': 1.5456536, 'theta_v': -0.9473225, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}
# bad partial drivative
# Parameters = {'A': 20.9, 'AT': 0.016666668, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 87.0, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.6, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.71, 'C_ev': 9.01367, 'C_hv': 1.57, 'C_jp': 3.3507938, 'C_pa': 0.76, 'C_pp': 20.536943, 'C_pv': 28.005367, 'C_rmv': 3.28, 'C_sa': 0.28, 'C_sv': 43.379185, 'Cvam_O2_n': 0.14701654, 'Cvb_O2_n': 0.12123883, 'Cvh_O2_n': 0.14456436, 'Cvrm_O2_n': 0.08475335, 'D': -3.293794, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 21.9, 'Emax_la': 0.35, 'Emax_lv0': 0.98830634, 'Emax_ra': 0.35, 'Emax_rv0': 0.97966325, 'Fi_CO2': 0.0421, 'Fi_O2': 21.0379, 'GEmax_lv': 0.547915, 'GEmax_rv': 0.40630218, 'GR_amp': 4.47, 'GR_ep': 1.9280238, 'GR_rmp': 2.47, 'GR_sp': 0.78412205, 'GT_s': -0.14353418, 'GT_v': 0.09874813, 'GV_amv': -28.29, 'GV_dead': 0.1698, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -314.45694, 'G_ap': 13.076185, 'Io_met': 0.22917163, 'Io_sh': 0.658, 'Io_sp': 0.65, 'Io_sv': 0.4858791, 'Io_v': 0.22, 'K1_vc': 0.13631189, 'K2': 194.4, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.06394197, 'KE_lv': 0.0075745145, 'KE_ra': 0.05820069, 'KE_rv': 0.011013837, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.2332, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.2025, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 100.0, 'Kp_po': 1914.1511, 'Kp_tr': 64.90742, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.0, 'Kv_po': 5.933391, 'Kv_tr': 4.9173083, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.31925493, 'MO2_bp': 0.7985471, 'MO2_hpn': 0.4, 'MO2_rmp': 0.51843137, 'P0_la': 0.55, 'P0_lv': 1.3026407, 'P0_ra': 0.55, 'P0_rv': 1.5517969, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 129.97798, 'P_n_max': 112.0, 'P_thormax_n': -2.0, 'P_thormin_n': -2, 'PaCO2_n': 40.0, 'PaO2_ac_n': 45.0, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 2.895741, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 5.665506, 'R_bv_n': 0.075, 'R_ep0': 0.9812416, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.33427525, 'R_mi': 443.18622, 'R_pa': 0.017963883, 'R_po': 346.9936, 'R_pp': 0.12998922, 'R_pv': 0.106910944, 'R_rmp0': 6.0224957, 'R_rmv_n': 0.125, 'R_rs': 3.02, 'R_sa': 0.085191086, 'R_sp0': 2.1256695, 'R_sv_n': 0.038, 'R_tr': 357.66846, 'Rvc_n': 0.041879185, 'T0': 0.65934116, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.228312, 'Ta': 2.0, 'Tc': 0.8072287, 'V0_dead': 0.1587, 'VA_rest': 0.067, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 5388.689, 'Vu_amv0': 257.87442, 'Vu_bv': 302.38882, 'Vu_ev0': 783.1693, 'Vu_hv': 50.359196, 'Vu_jp': 800.5119, 'Vu_la': 4.0, 'Vu_lv': 8.626895, 'Vu_pa': 1.0, 'Vu_pp': 94.7375, 'Vu_pv': 111.0555, 'Vu_ra': 4.0, 'Vu_rmv0': 136.99194, 'Vu_rv': 39.807945, 'Vu_sa': 1.0, 'Vu_sv0': 2015.4071, 'Vu_vc': 149.2693, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.9902505, 'Wb_sp': -1.248818, 'Wb_sv': -0.71638304, 'Wc_sh': 0.83062667, 'Wc_sp': 1.1785134, 'Wc_sv': 1.716, 'Wc_v': 0.19114566, 'Wp_sh': -0.2556052, 'Wp_sp': -0.3997, 'Wp_sv': -0.33244815, 'Wp_v': -0.07488816, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 57.378582, 'Ysv_min': -0.54744196, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.819, 'ahead1': 0.9276693, 'alpha2': 0.05591, 'alpha_O2': 3.17e-05, 'beta2': 0.03255, 'dc': 0.015, 'delta_P': 0.39631826, 'f_ab_max': 49.79187, 'f_ab_min': 2.8124347, 'f_acCO2_n': 0.94512486, 'f_ac_max': 9.766112, 'f_ac_min': 1.1302038, 'fab_o': 15.887719, 'fall_time_atr': 0.1, 'fall_time_ven': 0.19500749, 'fes_inf': 1.4663808, 'fes_max': 80.0, 'fes_min': 2.2765086, 'fes_o': 19.219563, 'fev_inf': 5.744877, 'fev_o': 2.5285296, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 0.53415096, 'g_ccsp': 2.073711, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 10.461564, 'gh_O2': 35.0, 'grm_O2': 21.005718, 'k_ab': 8.7707815, 'k_ac': 26.8884, 'kcc_sh': 0.074289314, 'kcc_sp': 0.13, 'kcc_sv': 0.12901378, 'kcc_v': 0.0162, 'kes': 0.06803719, 'kev': 8.400656, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.10546247, 'kr_am': 24.17, 'phi_max': 19.11744, 'phi_min': -2.2088487, 'rise_time_atr': 0.05, 'rise_time_ven': 0.21623023, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 0.9579094, 'scale_param3': 0.3, 'scale_param4': 26.6, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.4692627, 'theta_min': 0.054078624, 'theta_po_max': 1.4857466, 'theta_shn': 4.3426313, 'theta_spn': 14.252314, 'theta_svn': 12.292168, 'theta_tr_max': 0.9894643, 'theta_v': -0.9462644, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}

# Best so far
# Parameters = {'A': 20.9, 'AT': 0.016666666666666666, 'A_im': 30.0, 'B': 92.8, 'C': 10570.0, 'C2': 86.975, 'C_O2_param1': 0.00134, 'C_O2_param2': 2.59, 'C_O2_param3': 3.03e-05, 'C_amv': 4.4, 'C_bv': 5.75, 'C_ev': 10.0, 'C_hv': 1.57, 'C_jp': 3.76, 'C_pa': 0.76, 'C_pp': 15.715, 'C_pv': 25.450000000000003, 'C_rmv': 3.28, 'C_sa': 0.285, 'C_sv': 31.13, 'Cvam_O2_n': 0.145, 'Cvb_O2_n': 0.145, 'Cvh_O2_n': 0.11, 'Cvrm_O2_n': 0.15, 'D': -5.251, 'D1': 0.3855, 'DEmax_lv': 2.0, 'DEmax_rv': 2.0, 'DR_amp': 2.0, 'DR_ep': 2.0, 'DR_rmp': 2.0, 'DR_sp': 2.0, 'DT_s': 2.0, 'DT_v': 0.2, 'DV_amv': 5.0, 'DV_ev': 5.0, 'DV_rmv': 5.0, 'DV_sv': 5.0, 'Dmet': 4.0, 'E_rs': 24.189999999999998, 'Emax_la': 0.35, 'Emax_lv0': 1.3, 'Emax_ra': 0.32, 'Emax_rv0': 0.7, 'Fi_CO2': 0.0421, 'Fi_O2': 18.645, 'GEmax_lv': 0.475, 'GEmax_rv': 0.282, 'GR_amp': 4.47, 'GR_ep': 2.035, 'GR_rmp': 2.47, 'GR_sp': 0.695, 'GT_s': -0.13, 'GT_v': 0.095, 'GV_amv': -28.29, 'GV_dead': 0.16999999999999998, 'GV_ev': -74.21, 'GV_rmv': -28.29, 'GV_sv': -264.975, 'G_ap': 11.695, 'Io_met': 0.43, 'Io_sh': 0.6950000000000001, 'Io_sp': 0.685, 'Io_sv': 0.45, 'Io_v': 0.22, 'K1_vc': 0.15, 'K2': 195.035, 'KCCO2': 346000.0, 'KCSFCO2': 20.0, 'KE_la': 0.05, 'KE_lv': 0.015, 'KE_ra': 0.05500000000000001, 'KE_rv': 0.015, 'K_H': 3.0, 'Kb_ao': 2.0, 'Kb_mi': 2.0, 'Kb_po': 5.0, 'Kb_tr': 2.0, 'KcCO2': 0.23, 'KcMRV': 1.0, 'Kf_ao': 5000.0, 'Kf_mi': 500.0, 'Kf_po': 2000.0, 'Kf_tr': 500.0, 'Kh_CO2': 11.11, 'KpCO2': 0.195, 'KpO2': 4.72e-09, 'Kp_ao': 1000.0, 'Kp_mi': 98.625, 'Kp_po': 3000.0, 'Kp_tr': 98.03999999999999, 'Kr_vc': 0.001, 'Krm_CO2': 142.8, 'Kv_ao': 5.0, 'Kv_mi': 7.02, 'Kv_po': 10.0, 'Kv_tr': 6.875, 'L_pa': 0.00018, 'L_sa': 0.00022, 'MO2_ampn': 0.516, 'MO2_bp': 0.925, 'MO2_hpn': 0.385, 'MO2_rmp': 0.86, 'P0_la': 0.55, 'P0_lv': 1.455, 'P0_ra': 0.55, 'P0_rv': 1.415, 'PAMO2_nominal': 104.0, 'PO2_sh': 45.0, 'PO2_sp': 30.0, 'PO2_sv': 30.0, 'P_0': 3.93, 'P_abdmax_n': -1.0, 'P_abdmin_n': -2.5, 'P_n': 94.83, 'P_n_max': 112.0, 'P_thormax_n': -2.0, 'P_thormin_n': -2.0, 'PaCO2_n': 40.71, 'PaO2_ac_n': 45.120000000000005, 'Pa_O2_lower': 80.0, 'Pmax': 100.0, 'Pmax_dot': 1000.0, 'R_amp0': 3.51, 'R_amv_n': 0.0833, 'R_ao': 350.0, 'R_bpn': 6.55, 'R_bv_n': 0.075, 'R_ep0': 1.655, 'R_ev_n': 0.04, 'R_hpn': 19.71, 'R_hv_n': 0.224, 'R_mi': 368.21500000000003, 'R_pa': 0.023, 'R_po': 350.0, 'R_pp': 0.0894, 'R_pv': 0.1, 'R_rmp0': 5.27, 'R_rmv_n': 0.125, 'R_rs': 2.15, 'R_sa': 0.065, 'R_sp0': 2.415, 'R_sv_n': 0.038, 'R_tr': 350.0, 'Rvc_n': 0.05, 'T0': 0.63, 'T1': 1.0, 'T2': 2.0, 'T_im': 1.1, 'Ta': 2.0, 'Tc': 0.7, 'V0_dead': 0.14500000000000002, 'VA_rest': 0.065, 'VB': 0.9, 'VL_CO2': 3.0, 'VL_O2': 2.5, 'VTCO2': 0.25, 'VTO2': 0.25, 'VT_n': 0.73, 'V_tot': 5039.950000000001, 'Vu_amv0': 291.13, 'Vu_bv': 270.02, 'Vu_ev0': 610.085, 'Vu_hv': 93.16, 'Vu_jp': 579.81, 'Vu_la': 4.0, 'Vu_lv': 15.908, 'Vu_pa': 1.0, 'Vu_pp': 122.495, 'Vu_pv': 114.0, 'Vu_ra': 4.0, 'Vu_rmv0': 190.02499999999998, 'Vu_rv': 36.82, 'Vu_sa': 1.0, 'Vu_sv0': 1394.27, 'Vu_vc': 121.10499999999999, 'Vvc_max': 350.0, 'Vvc_min': 50.0, 'W_hn': 12660.0, 'Wb_sh': -1.7850000000000001, 'Wb_sp': -1.12, 'Wb_sv': -1.09, 'Wc_sh': 0.95, 'Wc_sp': 1.716, 'Wc_sv': 1.716, 'Wc_v': 0.2, 'Wp_sh': -0.2, 'Wp_sp': -0.415, 'Wp_sv': -0.3997, 'Wp_v': -0.105, 'Wt_sh': 0.4, 'Wt_sp': 0.4, 'Wt_sv': 0.4, 'Wt_v': 0.4, 'Ysh_max': 20.0, 'Ysh_min': -0.0283, 'Ysp_max': 5.5, 'Ysp_min': -0.037, 'Ysv_max': 64.9, 'Ysv_min': -0.437, 'Yv_max': 1.9, 'Yv_min': -0.0008, 'a2': 1.815, 'ahead1': 0.8500000000000001, 'alpha2': 0.055, 'alpha_O2': 3.17e-05, 'beta2': 0.035, 'dc': 0.015, 'delta_P': 0.3, 'f_ab_max': 47.03, 'f_ab_min': 2.485, 'f_acCO2_n': 1.365, 'f_ac_max': 12.94, 'f_ac_min': 0.86, 'fab_o': 24.97, 'fall_time_atr': 0.1, 'fall_time_ven': 0.29, 'fes_inf': 2.21, 'fes_max': 80.0, 'fes_min': 2.555, 'fes_o': 16.3, 'fev_inf': 6.155, 'fev_o': 3.39, 'gM': 40.0, 'g_abd': 3.39, 'g_ccsh': 1.03, 'g_ccsp': 1.5, 'g_ccsv': 0.2, 'g_thor': 6.8, 'gam_O2': 30.0, 'gb_O2': 10.0, 'gh_O2': 35.0, 'grm_O2': 29.520000000000003, 'k_ab': 11.855, 'k_ac': 29.2, 'kcc_sh': 0.11, 'kcc_sp': 0.13, 'kcc_sv': 0.095, 'kcc_v': 0.0162, 'kes': 0.065, 'kev': 6.83, 'kisc_sh': 6.0, 'kisc_sp': 2.0, 'kisc_sv': 2.0, 'kmet': 0.18, 'kr_am': 24.17, 'phi_max': 20.0, 'phi_min': -1.96, 'rise_time_atr': 0.05, 'rise_time_ven': 0.155, 's': 0.04, 'scale_param1': 4.9, 'scale_param2': 1.5, 'scale_param3': 0.3, 'scale_param4': 26.11, 'scale_param5': 0.5, 'scale_param6': 1.2, 'scale_param7': 30.0, 'scale_param8': 1.6, 'shift_param1': 4.0, 'shift_param2': 0.3, 'shift_param3': 4.0, 'shift_param4': 0.3, 'tauMR': 50.0, 'tau_CO2': 20.0, 'tau_Emax_lv': 8.0, 'tau_Emax_rv': 8.0, 'tau_M': 40.0, 'tau_MRV': 50.0, 'tau_O2': 10.0, 'tau_Ramp': 2.0, 'tau_Rep': 2.0, 'tau_Rrmp': 2.0, 'tau_Rsp': 2.0, 'tau_Ts': 2.0, 'tau_Tv': 1.5, 'tau_Vamv': 20.0, 'tau_Vev': 20.0, 'tau_Vrmv': 20.0, 'tau_Vsv': 20.0, 'tau_ac': 2.0, 'tau_ap': 2.0, 'tau_cc': 20.0, 'tau_isc': 30.0, 'tau_met': 10.0, 'tau_p': 2.076, 'tau_w': 5.0, 'tau_z': 0.8, 'theta_ao_max': 1.309, 'theta_mi_max': 1.31, 'theta_min': 0.0872665, 'theta_po_max': 1.309, 'theta_shn': 3.6399999999999997, 'theta_spn': 13.934999999999999, 'theta_svn': 13.32, 'theta_tr_max': 1.29, 'theta_v': -0.67, 'x_sh': 53.0, 'x_sp': 6.0, 'x_sv': 6.0}

# Parameters = {'A': 21.083691406249997, 'AT': 0.018017578125, 'A_im': 28.001953125, 'B': 99.99562499999999, 'C': 11389.587890625, 'C2': 81.31611328125, 'C_O2_param1': 0.00114292578125, 'C_O2_param2': 2.0825390625, 'C_O2_param3': 2.792689453125e-05, 'C_amv': 3.9007031250000006, 'C_bv': 5.824869140625, 'C_ev': 10.912109375, 'C_hv': 1.3090488281250001, 'C_jp': 4.4516484375, 'C_pa': 0.6927578125, 'C_pp': 17.2102734375, 'C_pv': 22.223525390625003, 'C_rmv': 3.231953125, 'C_sa': 0.3240234375, 'C_sv': 26.194376953125, 'Cvam_O2_n': 0.16834697265624998, 'Cvb_O2_n': 0.14341796875, 'Cvh_O2_n': 0.130904296875, 'Cvrm_O2_n': 0.1507314453125, 'D': -5.647901757812501, 'D1': 0.38858701171875, 'E_rs': 20.21900390625, 'Emax_la': 0.293603515625, 'Emax_lv0': 1.3778515624999998, 'Emax_ra': 0.31479492187499997, 'Emax_rv0': 0.7151757812499999, 'GEmax_lv': 0.49234863281249996, 'GEmax_rv': 0.23369648437499999, 'GR_amp': 4.088478515625, 'GR_ep': 1.81685546875, 'GR_rmp': 2.697220703125, 'GR_sp': 0.8023720703124999, 'GT_s': -0.13982617187500002, 'GT_v': 0.102462890625, 'GV_amv': -25.112900390625, 'GV_dead': 0.1492051171875, 'GV_ev': -59.788330078125, 'GV_rmv': -26.925228515625, 'GV_sv': -231.0327734375, 'G_ap': 9.915609374999999, 'Io_met': 0.45984480468749994, 'Io_sh': 0.728812109375, 'Io_sp': 0.555419921875, 'Io_sv': 0.37116210937500005, 'Io_v': 0.23130078125000003, 'K1_vc': 0.16661132812499999, 'K2': 201.48117187500003, 'KCCO2': 299438.671875, 'KE_la': 0.054892578125, 'KE_ra': 0.04006835937500001, 'K_H': 2.9748046875, 'Kb_ao': 2.083984375, 'Kb_mi': 1.815234375, 'Kb_po': 4.7197265625, 'Kb_tr': 1.687890625, 'KcCO2': 0.270502890625, 'KcMRV': 1.0998046875, 'Kf_ao': 4581.0546875, 'Kf_mi': 506.54296875, 'Kf_po': 1737.890625, 'Kf_tr': 504.98046875, 'Kh_CO2': 13.269072265624999, 'KpCO2': 0.18173583984375002, 'KpO2': 5.250078125e-09, 'Kp_ao': 808.7890625, 'Kp_mi': 86.07421875, 'Kp_po': 3088.4765625, 'Kp_tr': 106.11328125, 'Kr_vc': 0.0009365234375, 'Krm_CO2': 114.93726562500001, 'Kv_ao': 4.7392578125, 'Kv_mi': 7.3787109375, 'Kv_po': 11.041015625, 'Kv_tr': 7.0287109375000005, 'L_pa': 0.00016435546875, 'L_sa': 0.00019606640625000003, 'MO2_ampn': 0.56568515625, 'MO2_bp': 0.9461376953125001, 'MO2_hpn': 0.462265625, 'MO2_rmp': 0.97741015625, 'P0_la': 0.44118164062500004, 'P0_lv': 1.2940429687500001, 'P0_ra': 0.49961914062500007, 'P0_rv': 1.68779296875, 'PAMO2_nominal': 108.9359375, 'PO2_sh': 46.3271484375, 'PO2_sp': 32.900390625, 'PO2_sv': 28.353515625, 'P_0': 4.710626953125001, 'P_n': 94.00576171875001, 'P_n_max': 131.66015625, 'P_thormax_n': -2.8271484375, 'P_thormin_n': -7.4443359375, 'PaCO2_n': 41.44140625, 'PaO2_ac_n': 40.0517578125, 'Pa_O2_lower': 88.734375, 'Pmax': 118.41796875, 'Pmax_dot': 1049.0234375, 'R_amp0': 3.5065722656249996, 'R_amv_n': 0.08260041015625, 'R_ao': 364.833984375, 'R_bpn': 5.282947265625, 'R_bv_n': 0.0782080078125, 'R_ep0': 1.4723681640625, 'R_ev_n': 0.0353046875, 'R_hpn': 21.546263671875, 'R_hv_n': 0.25169375, 'R_mi': 397.099609375, 'R_pa': 0.0251697265625, 'R_po': 395.595703125, 'R_pp': 0.0757979296875, 'R_pv': 0.10193359375000001, 'R_rmp0': 6.059470703124999, 'R_rmv_n': 0.11374511718750001, 'R_rs': 3.4287617187499997, 'R_sa': 0.04981640625, 'R_sp0': 2.722951171875, 'R_sv_n': 0.043677734375, 'R_tr': 345.830078125, 'Rvc_n': 0.054052734375, 'T0': 0.49084765625, 'T_im': 1.0576757812500002, 'Tc': 0.77123046875, 'V0_dead': 0.17280322265625, 'VA_rest': 0.05554980468750001, 'VT_n': 0.827380859375, 'Vu_bv': 280.745521484375, 'Vu_hv': 110.5001328125, 'Vu_jp': 644.643296875, 'Vu_la': 4.00234375, 'Vu_lv': 14.034460156249999, 'Vu_pa': 1.0298828125, 'Vu_pp': 96.64634326171874, 'Vu_pv': 130.63242187499998, 'Vu_ra': 3.6554687500000003, 'Vu_rv': 32.6329787109375, 'Vu_sa': 1.0666015625, 'Vu_vc': 131.6244140625, 'Vvc_max': 320.400390625, 'W_hn': 10308.50390625, 'Wb_sh': -2.014208984375, 'Wb_sp': -0.94976806640625, 'Wb_sv': -1.04174560546875, 'Wc_sh': 1.0212890625, 'Wc_sp': 1.9971960937499997, 'Wc_sv': 1.7692898437499998, 'Wc_v': 0.16113281250000003, 'Wp_sh': -0.21628906250000002, 'Wp_sp': -0.32998669921875, 'Wp_sv': -0.34606837890625003, 'Wp_v': -0.1092966796875, 'Wt_sh': 0.449765625, 'Wt_sp': 0.435546875, 'Wt_sv': 0.414296875, 'Wt_v': 0.458671875, 'Ysh_max': 16.94140625, 'Ysh_min': -0.024137910156249998, 'Ysp_max': 4.487011718750001, 'Ysp_min': -0.042470507812499995, 'Ysv_max': 65.24224609375001, 'Ysv_min': -0.38706933593750004, 'Yv_max': 2.05177734375, 'Yv_min': -0.0006860937500000001, 'a2': 1.83125693359375, 'ahead1': 0.8692822265625, 'alpha2': 0.059519034179687506, 'alpha_O2': 2.657970703125e-05, 'beta2': 0.032661254882812504, 'dc': 0.0149091796875, 'delta_P': 0.31353515625, 'f_ab_max': 50.924894531250004, 'f_ab_min': 2.0558671875, 'f_acCO2_n': 1.2493359375, 'f_ac_max': 12.975058593750001, 'f_ac_min': 0.7738427734375, 'fab_o': 27.9443359375, 'fall_time_atr': 0.11701171875, 'fall_time_ven': 0.27322265625, 'fes_inf': 2.22755859375, 'fes_max': 72.109375, 'fes_min': 2.43712109375, 'fes_o': 13.287603515625, 'fev_inf': 7.2314648437499995, 'fev_o': 3.7043749999999998, 'gM': 45.4453125, 'g_ccsh': 1.0302734375, 'g_ccsp': 1.3028320312500001, 'g_ccsv': 0.17230468750000003, 'g_thor': 6.6127343750000005, 'gam_O2': 27.861328125, 'gb_O2': 11.650390625, 'gh_O2': 38.2880859375, 'grm_O2': 32.150390625, 'k_ab': 13.066921875, 'k_ac': 31.127958984375, 'kcc_sh': 0.103468359375, 'kcc_sp': 0.123576171875, 'kcc_sv': 0.08330273437499999, 'kcc_v': 0.013171992187499999, 'kes': 0.05514697265625001, 'kev': 7.177207031249999, 'kisc_sh': 5.933203125, 'kisc_sp': 1.673046875, 'kisc_sv': 1.974609375, 'kmet': 0.15345703125, 'kr_am': 25.251041015625002, 'phi_max': 20.91796875, 'phi_min': -2.0317988281250003, 'rise_time_atr': 0.04540039062500001, 'rise_time_ven': 0.153486328125, 's': 0.0328359375, 'scale_param1': 4.66744140625, 'scale_param2': 1.31923828125, 'scale_param3': 0.35349609375, 'scale_param4': 23.8101171875, 'scale_param5': 0.45048828125, 'scale_param6': 1.225078125, 'scale_param7': 31.763671875, 'shift_param1': 3.97265625, 'shift_param2': 0.30802734374999996, 'shift_param3': 4.76484375, 'shift_param4': 0.32197265625, 'theta_ao_max': 1.4984470703125, 'theta_mi_max': 1.4693013671875, 'theta_min': 0.10071440400390624, 'theta_po_max': 1.5429326171875, 'theta_shn': 3.8032031250000005, 'theta_spn': 12.573351562500001, 'theta_svn': 11.3246015625, 'theta_tr_max': 1.2703947265625, 'theta_v': -0.6742890625, 'x_sh': 45.722851562500004, 'x_sp': 6.132421875, 'x_sv': 5.194921875}