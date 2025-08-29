    # CARDIOVASCULAR SYSTEM
Parameters = {
    # Table 1. Systemic arteries
# systemic_arteries = {
    "C_sa": 0.28,  # Systemic arterial compliance (decreasing C_sa allows Q_sa to match closer to Q_lv) # want to change to 1.13 (harry thesis
    "L_sa": 0.00066,  # Systemic arterial inertance
    "R_sa": 0.2,  # Systemic arterial hydraulic resistance (want to increase from 0.06 to 0.2 to increase Psys). This is because P_sa decreases at a slower rater (first order)
    "Vu_sa": 1, # Systemic arterial unstressed volume (edited for DGSM)
# }
#
#     # Systemic peripheral and venous circulation
#     # Table 2. Compliance values
# systemic_peripheral_and_venous = {
    "C_jp": 3.72, # 3.72
    "C_amp": 0.315,  # Active skeletal muscle peripheral compliance
    "C_amv": 9.4,    # Active skeletal muscle venous compliance
    "C_bp": 0.358,   # Brain peripheral compliance
    "C_bv": 10.71,   # Brain venous compliance
    "C_ep": 0.668,   # Extra-splanchnic peripheral compliance
    "C_ev": 20,      # Extra-splanchnic venous compliance
    "C_hp": 0.119,   # Coronary peripheral compliance
    "C_hv": 3.57,    # Coronary venous compliance
    "C_rmp": 0.21,   # Resting skeletal muscle peripheral compliance
    "C_rmv": 6.28,   # Resting skeletal muscle venous compliance
    "C_sp": 2.05,    # Splanchnic peripheral compliance
    "C_sv": 61.11,   # Splanchnic venous compliance
    "kr_am": 24.17,  # Constant parameter
    "P_0": 3.93,     # Constant parameter
    "R_amv_n": 0.0833,  # Active skeletal muscle venous resistance
    "R_bv_n": 0.075,    # Brain venous resistance
    "R_ev_n": 0.04,     # Extra-splanchnic venous resistance
    "R_hv_n": 0.224,    # Coronary venous resistance
    "R_rmv_n": 0.125,   # Resting skeletal muscle venous resistance
    "R_sv_n": 0.038,    # Splanchnic venous resistance
    "V_tot": 5027.6,    # Total blood volume
    "Vu_amp": 60.22,   # Active skeletal muscle peripheral unstressed volume
    "Vu_bp": 68.42,    # Brain peripheral unstressed volume
    "Vu_bv": 279.49,   # Brain venous unstressed volume
    "Vu_ep": 127.72,   # Extra-splanchnic peripheral unstressed volume
    "Vu_hp": 23,       # Coronary peripheral unstressed volume
    "Vu_hv": 93.16,    # Coronary venous unstressed volume
    "Vu_rmp": 40.1,    # Resting skeletal muscle peripheral unstressed volume
    "Vu_sp": 260.3,    # Splanchnic peripheral unstressed volume
# }
#
#     # Table 3. Vena Cava Parameters
# vena_cava = {
    "D1": 0.3855,      # Parameter for P-V curve of vena cava
    "D2": -5,          # Parameter for P-V curve of vena cava
    "K1_vc": 0.15,     # Parameter for P-V curve of vena cava
    "K2_vc": 0.4,      # Parameter for P-V curve of vena cava
    "Kr_vc": 0.001,    # Gain for vena cava flow resistance
    "Rvc_n": 0.075,    # Nominal vena cava flow resistance # edited (changed to 0.0025 from xxx for better left atrial pressures, changed back to 0.025)
    "Vu_vc": 123,      # Vena cava unstressed volume
    "Vvc_max": 350,     # Maximum volume of vena cava
    "Vvc_min": 50,      # Minimum volume of vena cava
# }
#
#     # Table 4. Pulmonary Circulation Parameters
# pulmonary_circulation = {
    "C_pa": 0.76, # 0.76,           # Pulmonary arterial compliances want to change to 5
    "C_pp": 5.8, # 5.8,            # Pulmonary peripheral compliances want to change to 10
    "C_pv": 20.5, # 25.37,          # Pulmonary venous compliances want to change to 15 # edited
    "L_pa": 0.00018,        # Pulmonary arterial inertance
    "R_pa": 0.023,          # Pulmonary arterial flow resistance (this value could raise RA pressure)
    "R_pp": 0.3,         # Pulmonary peripheral flow resistance # edited to reduce CvtO2 oscillation # edited from 0.0894
    "R_pv": 0.06,         # Pulmonary venous flow resistance # edited to remove the backflow
    "Vu_pa": 1,            # Pulmonary arterial unstressed volume # edited for dgsm
    "Vu_pp": 116.6775,     # Pulmonary peripheral unstressed volume
    "Vu_pv": 114,          # Pulmonary venous unstressed volume
# }

    # Table 5. Heart Parameters
# heart = {
#     "C_la": 4, # 19.23,          # Left atrial compliances changed
#     "C_ra": 5, # 31.25,          # Right atrial compliances changed
    "s": 0.04,
    "Ta": 5,
    "KE_lv": 0.014, # 0.014      # End-diastolic P-V relationship in left ventricle # adjusted (changes a lot depending on whether it is 0.06 or 0.05)
    "KE_rv": 0.011, # 0.011       # End-diastolic P-V relationship in right ventricle Another model use 0.027
    # "KR_lv": 0.000375,     # Viscosity of left ventricle
    # "KR_rv": 0.00014,       # Viscosity of right ventricle (this parameter affects RA pressure) (changed from 0.0014. The removes the kink in the lv)
    # "ksys": 0.075,         # Duration of systole as function of heart rate (this parameter affects RA pressure)
    "Emax_la": 0.25,    # edited
    "P0_la": 0.55,      # edited
    "KE_la": 0.05,

    "Emax_ra": 0.25,    # edited
    "P0_ra": 0.55,      # edited
    "KE_ra": 0.05,
    "P0_lv": 1.5, # 1.5        # End-diastolic P-V relationship in left ventricle # another model use 1.0
    "P0_rv": 1.5, # 1.5         # End-diastolic P-V relationship in right ventricle (check this one first, has strong effect with the smallest change in volume)
    # "R_la": 0.0025,         # Left atrial flow resistance
    # "R_ra": 0.0025,         # Right atrial flow resistance
    # "Tsys_0": 0.4,         # Duration of systole as function of heart rate (need to change Tsys0, T0, HR in initial/next conditions)
    "Vu_la": 4,           # Left atrial unstressed volume # adjusted to the shi paper Increasing these heart unstressed volumes decreases the maximum flow and pressures
    "Vu_lv": 5,       # Left ventricular unstressed volume # adjusted to the shi paper
    "Vu_ra": 4,           # Right atrial unstressed volume # adjusted to the shi paper
    "Vu_rv": 10,       # Right ventricular unstressed volume # adjusted to the shi paper
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
    "P_abdmax_n": 0,        # Basal value of abdominal pressure at the end of expiration
    "P_abdmin_n": -2.5,     # Basal value of abdominal pressure at the end of inspiration
    "P_thormax_n": -0,      # Basal value of intrathoracic pressure at the end of expiration
    "P_thormin_n": -0,      # Basal value of intrathoracic pressure at the end of inspiration
    "VT_n": 0.45,           # Basal value of tidal volume changed
# }

    # Table 8. Afferent Baroreflex Pathway
# afferent_baroreflex = {
    "f_ab_max": 47.78,      # Upper saturation level of the frequency discharge in the baroreceptor afferent fibers
    "f_ab_min": 2.52,       # Lower saturation level of the frequency discharge in the baroreceptor afferent fibers
    "k_ab": 11.76,          # Parameter related to the slope of the static function at the central point
    "P_n": 92,              # Value of baroreceptor pressure at the central point of the sigmoidal function
    "P_n_max": 122,
    "tau_p": 2.076,         # Time constant for the real pole
    "tau_z": 0.8,          # Time constant for the real zero
# }

    # Table 9. Afferent Chemoreflex Pathway
# afferent_chemoreflex = {
    "f_ac_IC": 8.0807,      # Initial condition for the afferent activity from chemoreceptors
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
    "f_ap_IC": 4.4492,  # Initial condition for the afferent activity from pulmonary stretch receptors, spikes/s
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
    "R_bpn": 10.57,    # Constant parameter denoting the basal value of peripheral cerebrovascular conductance # edited from 6.57
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
    "R_hpn": 25.71,      # Normal peripheral resistance in coronary compartment # edited from 19.71
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
    "Ysv_min": -0.028,         # Lower saturation of the central command response (unstressed volume of veins)
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
    "GR_amp": 3.47,       # Constant gain factor # edited
    "GR_ep": 1.94,        # Constant gain factor
    "GR_rmp": 2.47,        # Constant gain factor
    "GR_sp": 0.695,       # Constant gain factor
    "GV_amv": -35.29,     # Constant gain factor # edited
    "GV_ev": -74.21,      # Constant gain factor
    "GV_rmv": -58.29,      # Constant gain factor
    "GV_sv": -265.4,      # Constant gain factor
    "R_amp0": 3.510,     # Basal level of active skeletal peripheral resistance
    "R_ep0": 5.655,      # Basal level of extra-splanchnic peripheral resistance # edited from 1.655
    "R_rmp0": 10.270,      # Basal level of resting skeletal peripheral resistance # edited from 5.270
    "R_sp0": 5.49,       # Basal level of splanchnic peripheral resistance # edited from 2.49
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
    "Vu_sv0": 1361.6,    # Basal level of splanchnic venous unstressed volume
# }

# Table 18: Parameters of Effectors for Reflex Control: Heart Period
# parameters_heart_period = {
    "DT_s": 2,              # Pure latency of the mechanism
    "DT_v": 0.2,            # Pure latency of the mechanism
    "fsh_IC": 3.8576,         # Initial condition for the efferent sympathetic cardiac activity
    "fv_IC": 4.2748,         # Initial condition for the efferent vagal activity
    "GT_s": -0.13,          # Constant gain factor
    "GT_v": 0.09,           # Constant gain factor
    "T0": 0.58,              # Heart period in the absence of cardiac innervation # want to change to 0.83333 from 0.58
    "tau_Ts": 2,              # Time constant
    "tau_Tv": 1.5,            # Time constant
# }

# Table 19: Parameters of the Upper Airways
# parameters_upper_airways = {
    "A0_ua": 1,               # Maximum area of opening in upper airway
    "b_ua": 1,                 # Upper airway mechanics constant
    "C_ua": 0.001,             # Upper airway compliance
    "K_ua": 1,                 # Proportionality coefficient
    "Pcrit_min": -40,         # Critical upper airway pressure
    "R_AW": 0.82128,           # Airway wall resistance
    "R_CW": 0.8326,            # Chest wall resistance
    "R_L": 1.3661,             # Lung transmural resistance
    "R_trachea": 1000000,      # Upper airway wall resistance
    "R_rs": 3.02,              # Overall resistance # changed
# }

# Table 20: Parameters of the Pulmonary Mechanics
# parameters_pulmonary_mechanics = {
    "E_CW": 10.5,           # Chest wall elastance (cmH2O/l) # changed
    "E_L": 10.5,            # Lung transmural elastance (cmH2O/l) # changed
    "E_rs": 21.9, # 21.9,             # Overall elastance (cmH2O/l) # changed
    "k_aw1": 1.85,            # Constant for upper airway pressure (cmH2O·s/l)
    "k_aw2": 0.43,            # Constant for upper airway pressure (cmH2O·s^2/l^2)
    "P_ao": 0,                # Airway pressure (cmH2O)
# }

# Table 21: Parameters of Ventilation Controller
# parameters_ventilation_controller = {
    "GV_dead": 0.1698,        # Constant gain for dead space volume
    "Kbg": 17.4,             # Blood gas dissociation constant
    "KcCO2": 0.2332,         # Constant gain of CO2 central chemoreceptors
    "KcMRV": 1,              # Constant gain of central response to exercise (neural drive)
    "KpCO2": 0.2025,         # Constant gain of CO2 peripheral chemoreceptors
    "KpO2": 4.72e-9,         # Constant gain of O2 peripheral chemoreceptors
    "V0_dead": 0.1587,        # Offset value of dead space volume
    "VA_rest": 0.067,        # Basal value of alveolar ventilation
# }

# Table 22: Parameters of Breathing Pattern Optimizer
# parameters_breathing_pattern_optimizer = {
    "lambda1": 0.3,         # Weighting factor (Dimensionless) edited
    "lambda2": 0.05,        # Weighting factor (Dimensionless) edited
    "n": 1.101,              # Power index of efficiency factor (Dimensionless)
    "Pmax": 100,              # Maximum inspiratory pressure (cmH2O) edited
    "Pmax_dot": 1000,        # Maximum pressure rate during inspiration (cmH2O/s)
# }

# Table 23: Parameters of the Gas Exchange and Mixing
# parameters_gas_exchange_mixing = {
    "a1": 0.3836,           # Parameter for O2 dissociation in blood (Dimensionless)
    "a2": 1.819,            # Parameter for CO2 dissociation in blood (Dimensionless) # changed
    "alpha1": 0.03198,      # O2 dissociation constant (mmHg^-1)
    "alpha2": 0.05591,      # CO2 dissociation constant (mmHg^-1)
    "beta1": 0.008275,      # O2 Bohr-Haldane parameter (mmHg^-1)
    "beta2": 0.03255,       # CO2 Bohr-Haldane parameter (mmHg^-1)
    "C1": 9,                # Max concentration of hemoglobin-bound oxygen (mmol/l)
    "C2": 87,               # Max carbon dioxide concentration (mmol/l)
    "Pd_CO2_IC": [39.5616, 39.6736, 39.8127, 40.0061, 40.3359], # Initial CO2 dead space conditions (mmHg)
    "Pd_O2_IC": [104.3637, 104.2258, 104.0505, 103.8005, 103.3579], # Initial O2 dead space conditions (mmHg)
    "Fi_CO2": 0.0421,       # Inspired fraction of CO2 (%)
    "Fi_O2": 21.0379,       # Inspired fraction of O2 (%)
    "K1": 13,               # Parameter for O2 dissociation equation (mmHg)
    "K2": 194.4,            # Parameter for CO2 dissociation equation (mmHg) # Changed
    "LCTV": 0.588,          # Lung to chemoreceptor vascular volume constant (l)
    "PACO2_Delay_IC": 40.4448,      # Initial CO2 convection delay (mmHg)
    "dPa_CO2_dt_IC": -0.2465,       # Initial CO2 rate of change (mmHg/s)
    "PACO2_IC": 40.9432,            # Initial Condition for CO2 convection (mmHg)
    "d2Pa_CO2_dt2_IC": 40.3928,   # Second order CO2 rate of change (mmHg/s^2)
    "PAO2_Delay_IC": 103.1223,      # Initial O2 convection delay (mmHg)
    "dPa_O2_dt_IC": 0.3557,         # Initial O2 rate of change (mmHg/s)
    "PAO2_IC": 102.5153,            # Initial Condition for O2 convection (mmHg)
    "d2Pa_O2_dt2_IC": 103.1435,   # Second order O2 rate of change (mmHg/s^2)
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
    "h": 0.0183/1000,                    # Cerebral blood flow constant (ml/(100g·s))
    "KCCO2": 346000,                # CO2 central receptor constant (s·cm^-2·l^-1)
    "KCSFCO2": 20,                 # CO2 diffusion time constant in cerebrospinal fluid (s), changed to be faster to see if limit cycle is reached
    "MRBCO2": 0.0009,               # Metabolic production rate of CO2 (1/s STPD)
    "MRBO2": 0.000925,              # Metabolic production rate of O2 (1/s STPD)
    "PbCO2IC": 48.5338,             # Initial condition for brain CO2 partial pressure (mmHg)
    "SbCO2": 0.36/1000,                  # Dissociation slope for CO2 in the brain (ml·(100g·y)/mmHg) # convert to L
    "SCO2": 0.0043,                 # Dissociation slope for CO2 in blood (mmHg^-1)
# }

# Table 25: Parameters of the Gas Transport: Body Tissues Compartment
# parameters_body_tissues_compartment = {
    "Cv_CO2_IC": 0.5247,            # Initial mixed venous CO2 concentration (ml/ml)
    "Cv_O2_IC": 0.1639,             # Initial mixed venous O2 concentration (ml/ml)
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
    "Kp_ao": 300,
    "Kf_ao": 2000,
    "Kb_ao": 0.2,
    "Kv_ao": 20,
    "theta_ao_max": 1.309,
    "Kp_mi": 1000,
    "Kf_mi": 800,
    "Kb_mi": 2,
    "Kv_mi": 3.5,
    "theta_mi_max": 1.309,
    "Kp_po": 800,
    "Kf_po": 800,
    "Kb_po": 1,
    "Kv_po": 10,
    "theta_po_max": 1.309,
    "Kp_tr": 2000,
    "Kf_tr": 800,
    "Kb_tr": 2,
    "Kv_tr": 7,
    "theta_tr_max": 1.309,
    "alpha_O2": 0.0000317,
    "R_po": 350,
    "R_mi": 350,
    "R_tr": 350,
    "R_ao": 350,
    "C_O2_param1": 0.00134,
    "C_O2_param2": 150,
    "C_O2_param3": 3.03e-5,
    "PAMO2_nominal": 104
}