import numpy as np
import bisect

import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from scipy.signal import find_peaks

from Initial_Conditions import Initial_Conditions
from Next_Conditions import Next_Conditions

from Parameter_Ranges import parameters
from GSA_Cardiovascular_system import cardiovascular_system
from joblib import Parallel, delayed


target_values = np.arange(0, 10000, 30)

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech):
    """

    """
    i = Initial_Conditions_dict["i"].item()
    if t != 0:
        latest_nonzero_value = Initial_Conditions_dict["all_time"][i - 1]
        if t < latest_nonzero_value:
            index = bisect.bisect_left(Initial_Conditions_dict["time_history"], t)
            num_removed = i - index
            Initial_Conditions_dict["time_history"][index:i + 1] = np.full((num_removed + 1,), 1e6)
        else:
            num_removed = 0
    else:
        num_removed = 0

    # Indices for slicing
    idx_cardio = num_cardio
    idx_cardio_contr = idx_cardio + num_cardio_control
    idx_gas = idx_cardio_contr + num_gas
    idx_resp_mech = idx_gas + num_resp_mech
    idx_resp_contr = idx_resp_mech + num_resp_control

    # Extract each subsystem's state variables
    cardio_state = Initial_Conditions_numpy[:idx_cardio]
    cardio_contr_state = Initial_Conditions_numpy[idx_cardio:idx_cardio_contr]
    gas_state = Initial_Conditions_numpy[idx_cardio_contr:idx_gas]
    resp_mech_state = Initial_Conditions_numpy[idx_gas:idx_resp_mech]
    resp_contr_state = Initial_Conditions_numpy[idx_resp_mech:idx_resp_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    d_cardio = cardiovascular_system(t, cardio_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    # d_cardio_contr = cardiovascular_controller(t, cardio_contr_state, Parameters, Initial_Conditions_dict["time_history"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    # d_gas = gas_exchange(t, gas_state, Parameters, Initial_Conditions_dict["time_history"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    # d_resp_mech = respiratory_mechanics(t, resp_mech_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    # d_resp_vent = resp_control_vent(t, resp_contr_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)

    # d_combined = np.concatenate((d_cardio, d_cardio_contr, d_gas, d_resp_mech, d_resp_vent))
    # d_combined = np.concatenate((d_cardio, d_cardio_contr, d_gas, d_resp_mech))

    if num_removed == 0:
        Initial_Conditions_dict["time_history"][i] = t
        Initial_Conditions_dict["all_time"][i] = t
        Initial_Conditions_dict["i"][0] = i + 1
        i = i + 1
    else:
        Initial_Conditions_dict["time_history"][i - num_removed] = t
        Initial_Conditions_dict["all_time"][i - num_removed] = t
        Initial_Conditions_dict["i"][0] = i - num_removed + 1
        i = i - num_removed + 1

    # just for checking progress of code
    if t != 0:
        if i > 2:
            last_nonzero_value1 = Initial_Conditions_dict["time_history"][i - 1]
            last_nonzero_value2 = Initial_Conditions_dict["time_history"][i - 2]
            if t > 0.00001:
                if last_nonzero_value1 < last_nonzero_value2:
                    print("ISSUE")
            diff = np.abs(last_nonzero_value1 - target_values)
            if np.any(diff < 0.001):
                print(last_nonzero_value1)

    return d_cardio


t_span = (0,30) # Simulate for 30 seconds for just the cardiovascular system for global sensitivity

# t_eval = np.arange(t_span[0], t_span[1], 0.01) # set as the number of times calculated in solution.t

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PvbCO2", "PCSFCO2", "MRTO2", "MRTCO2", "Cv_O2", "Cv_CO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc"]
IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)
num_cardio = len(required_cardio_keys)

# cardiovascular controller
required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv",
                         "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change",
                         "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                         "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                         "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2", "xam_O2",
                         "xM", "x_met"]

IC_cardio_contr = np.array([Initial_Conditions[key] for key in required_cardio_control_keys], dtype=float)
num_cardio_control = len(required_cardio_control_keys)

# resp control ventilation
required_resp_control_keys = ["VE_integral"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)


# resp mechanics
required_resp_mech_keys = ["V", "Vflow_ua"]
IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
num_resp_mech = len(required_resp_mech_keys)

# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))
IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_mech, IC_resp_contr))
# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_mech))

def simulate_cpu(Parameters, storage):
    local_updates = {key: np.array(value, copy=True) for key, value in storage.items()}
    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_cardio, t_eval=t_eval, max_step = 0.003, method="RK23", rtol=1e-3,
                             atol=1e-6, args=(Parameters, local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech))

    index = np.where(local_updates["time_history"] == 1e6)[0][0] - 1
    P_sa = local_updates["P_sa"][:index]
    # V_lv_smooth = np.convolve(local_updates["V_lv"][:index], np.ones(20) / 20, mode='same')

    peaks, _ = find_peaks(P_sa, distance=int(500))  # Adjust distance based on heart rate
    # troughs, _ = find_peaks(-P_sa, distance=int(500))  # Find minima (inverted peaks)

    # last_5_troughs = troughs[-6:-1]  # Get indices of last 5 minima
    # last_5_min = P_sa[last_5_troughs]  # Get actual minimum values

    last_5_peaks = peaks[-6:-1]  # Get indices of last 5 max
    last_5_max = P_sa[last_5_peaks]  # Get actual max values

    mean_P_sys = np.mean(last_5_max)

    # diff = last_5_max - last_5_min
    # mean_diff = np.mean(diff)
    #
    # HR = local_updates["HR"][:index]
    # last_5_HR = HR[-5:]
    # mean_HR = np.mean(last_5_HR)
    # mean_HR = Parameters["HR"] # HR was kept constant from the initial conditions

    # CO = mean_diff * mean_HR

    return mean_P_sys


def parallel_simulations(param_samples, storage, n_jobs):
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, storage) for params in param_samples)
    return results



if __name__ == "__main__":
    t_eval = np.linspace(0, t_span[1], t_span[1] * 1000)
    # sample from a simulation (do this for initial training of emulator but use saltelli sampling for GSA)
    param_keys = list(parameters.keys())
    lhd = LatinHypercube(list(parameters.values()))
    X = lhd.sample(1000)

    param_samples = [dict(zip(param_keys, row)) for row in X]
    print(f"Number of samples created: {len(X)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=-1)

    print(Result)

    np.save('X_samples_P_sys_1000_fixed.npy', X)
    np.save('Result_P_sys_1000_fixed.npy', Result)

    # # X = np.load('X_samples_900.npy')
    # # Result = np.load('Result_900.npy')
    #
    #
    # # compare emulators
    # ae = AutoEmulate()
    # ae.setup(X, Result)
    #
    # best_emulator = ae.compare()
    #
    # # cross-validation results
    # ae.summarise_cv()
    # ae.plot_cv()
    # #
    # # test set results for the best emulator
    # ae.evaluate(best_emulator)
    # ae.plot_eval(best_emulator)
    # #
    # # refit on full data and emulate!
    # emulator = ae.refit(best_emulator)
    # emulator.predict(X)
    #
    # # global sensitivity analysis
    # si = ae.sensitivity_analysis(emulator)
    # print(si)
    # ae.plot_sensitivity_analysis(si)