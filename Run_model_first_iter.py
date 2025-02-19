import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from Exp_inputs import Exp_inputs
from Cardiovascular_controller import cardiovascular_controller
from Cardiovascular_system import cardiovascular_system
from Gas_Exchange import gas_exchange
from Initial_Conditions import Initial_Conditions
from Parameters import Parameters
from Resp_Control_Breath_Optimiser import breath_optimiser
from Resp_Control_Ventilation import resp_control_vent
from Respiratory_Mechanics import respiratory_mechanics


# Resp control breath optimiser
t = 0.001
initial_Nd_guess = np.array([0, 0, 0, 0, 1, 2])  # Example initial values for a0, a1, a2, tau, t1, t2
time_history = []
bounds = [(0, None), (0, None), (0, None), (0, None), (0.1, None), (0.1, None)]

# Optimize
result = minimize(breath_optimiser, initial_Nd_guess, args=(t, time_history, Parameters, Exp_inputs, Initial_Conditions), method='SLSQP', bounds=bounds)



# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech):
    """

    """
    time_history = []

    # Indices for slicing
    idx_gas = num_gas
    idx_cardio = idx_gas + num_cardio
    idx_cardio_contr = idx_cardio + num_cardio_control
    idx_resp_contr = idx_cardio_contr + num_resp_control
    idx_resp_mech = idx_resp_contr + num_resp_mech

    # Extract each subsystem's state variables
    gas_state = Initial_Conditions_numpy[:idx_gas]
    cardio_state = Initial_Conditions_numpy[idx_gas:idx_cardio]
    cardio_contr_state = Initial_Conditions_numpy[idx_cardio:idx_cardio_contr]
    resp_contr_state = Initial_Conditions_numpy[idx_cardio_contr:idx_resp_contr]
    resp_mech_state = Initial_Conditions_numpy[idx_resp_contr:idx_resp_mech]

    # Cardiovascular dynamics
    d_gas = gas_exchange(t, gas_state, Parameters, time_history, Exp_inputs, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict)
    d_cardio = cardiovascular_system(t, cardio_state, Parameters, Exp_inputs, Initial_Conditions_dict, Initial_Conditions_dict)
    d_cardio_contr = cardiovascular_controller(t, cardio_contr_state, Parameters, time_history, Exp_inputs, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict)
    d_resp_vent = resp_control_vent(t, resp_contr_state, Parameters, time_history, Exp_inputs, Initial_Conditions_dict)
    d_resp_mech = respiratory_mechanics(t, resp_mech_state, Parameters, Exp_inputs)

    d_combined = np.concatenate((d_gas, d_cardio + d_cardio_contr, d_resp_vent, d_resp_mech))

    return d_combined


def simulate():

    # Time span
    t_span = (0, 0.001)  # Simulate for x seconds
    t_eval = np.linspace(t_span[0], t_span[1], 2)

    # gas exchange
    required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                         "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                         "PvbCO2", "PCSFCO2", "MRTO2", "MRTCO2", "CvO2", "CvCO2", "MRV"]
    IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
    num_gas = len(required_gas_keys)

    # cardiovascular system
    required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                               "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "V_sa", "P_sa", "Q_sa", "VT_vc" ]
    IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)
    num_cardio = len(required_cardio_keys)

    # cardiovascular controller
    required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv",
                             "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change",
                             "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                             "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                             "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2", "xam_O2",
                             "xM", "x_met", "beta"]

    IC_cardio_contr = np.array([Initial_Conditions[key] for key in required_cardio_control_keys], dtype=float)
    num_cardio_control = len(required_cardio_control_keys)

    # resp control ventilation
    required_resp_control_keys = ["VE_integral"]
    IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
    num_resp_control = len(required_resp_control_keys)


    # resp mechanics
    required_resp_mech_keys = ["V", "alpha"]
    IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
    num_resp_mech = len(required_resp_mech_keys)

    IC_overall = np.concatenate((IC_gas, IC_cardio, IC_cardio_contr, IC_resp_contr, IC_resp_mech))

    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_overall, method="RK23", t_eval=t_eval, max_step=0.01, rtol=1e-3,
                             atol=1e-6, args=(Parameters, Initial_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech))

    return ODE_solution


if __name__ == "__main__":
    solution = simulate()
    print(solution.y)
