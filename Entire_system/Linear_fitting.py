import numpy as np
import bisect

import pandas as pd
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from All_Parameter_ranges import parameters as parameters_change

from line_profiler import LineProfiler
from collections import deque

import Resp_Control_Breath_Optimiser
from All_Cardiovascular_controller import cardiovascular_controller
from All_Cardiovascular_system import cardiovascular_system
from All_Gas_exchange import gas_exchange
from Parameters import Parameters as Old_Parameters
from All_Respiratory_controller import resp_control_vent
from Respiratory_Mechanics import respiratory_mechanics


from Selected_Conditions import Selected_Conditions as previous_Selected_Conditions
from Initial_Conditions_after_running_again import Initial_Conditions
from All_Next_Conditions import Next_Conditions

# output_file1 = "Selected_Conditions_new.py"
# output_file2 = "Initial_Conditions_new.py"
# output_file3 = "Next_Conditions_new.py"


target_values = np.arange(0, 10000, 10)
t_span = (0, 60) # Simulate for 30 seconds for just the cardiovascular system for global sensitivity

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Current_Parameters, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Parameters):
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
    idx_resp_contr = idx_gas + num_resp_control
    # idx_resp_mech = idx_resp_contr + num_resp_mech

    # Extract each subsystem's state variables
    cardio_state = Initial_Conditions_numpy[:idx_cardio]
    cardio_contr_state = Initial_Conditions_numpy[idx_cardio:idx_cardio_contr]
    gas_state = Initial_Conditions_numpy[idx_cardio_contr:idx_gas]
    resp_contr_state = Initial_Conditions_numpy[idx_gas:idx_resp_contr]
    # resp_mech_state = Initial_Conditions_numpy[idx_resp_contr:idx_resp_mech]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    d_cardio = cardiovascular_system(t, cardio_state, Current_Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i, t_span[0], Parameters)
    d_cardio_contr = cardiovascular_controller(t, cardio_contr_state, Current_Parameters, Initial_Conditions_dict["time_history"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i, t_span[0], previous_Selected_Conditions, Parameters)
    d_gas = gas_exchange(t, gas_state, Current_Parameters, Initial_Conditions_dict["time_history"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i, t_span[0], previous_Selected_Conditions, Parameters)
    d_resp_vent = resp_control_vent(t, resp_contr_state, Current_Parameters, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i, t_span[0], Parameters)
    # d_resp_mech = respiratory_mechanics(t, resp_mech_state, Current_Parameters, Initial_Conditions_dict, num_removed, i)

    d_combined = np.concatenate((d_cardio, d_cardio_contr, d_gas, d_resp_vent))

    # if np.any(np.isnan(d_combined)) or np.any(np.isinf(d_combined)):
    #     print(f"NaN or Inf detected at t = {t}")

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
            if np.any(diff < 0.00001):
                for key, value in Current_Parameters.items():
                    if value not in [nominal_values[key]]:
                        changed_parameter = key
                        break
                    else:
                        changed_parameter = "no_idea"
                print(last_nonzero_value1, changed_parameter)

    return d_combined

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]
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


# # resp mechanics
# required_resp_mech_keys = ["Vflow_ua"]
# IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
# num_resp_mech = len(required_resp_mech_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_contr))

def simulate_cpu(Current_Parameters, storage):
    list_keys = {
        'HR1', 'Vu_ev1', 'Vu_sv1', 'Vu_rmv1', 'Vu_amv1', 'Emax_lv1', 'Emax_rv1'
        'Pa_O2_history', 'Pa_CO2_history', 'Pb_CO2_history',
        'PamO2', 'PamCO2', 'PmbCO2', 'Nd', 'finish_breath_time'
    }

    local_updates = {
        key: value if key in list_keys else np.array(value, copy=True)
        for key, value in storage.items()
    }
    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_overall, t_eval=t_eval, max_step = 0.003, method="RK23", rtol=1e-3,
                             atol=1e-6, args=(Current_Parameters, local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Old_Parameters))

    index = np.where(local_updates["all_time"] == 1e6)[0][0] - 1
    P_sa_smooth = local_updates["P_sa"][:index]

    peaks, _ = find_peaks(P_sa_smooth, distance=int(500))  # Adjust distance based on heart rate
    troughs, _ = find_peaks(-P_sa_smooth, distance=int(500))  # Find minima (inverted peaks)

    last_5_troughs = troughs[-6:-1]  # Get indices of last 5 minima
    last_5_min = P_sa_smooth[last_5_troughs]  # Get actual minimum values

    last_5_peaks = peaks[-6:-1]  # Get indices of last 5 max
    last_5_max = P_sa_smooth[last_5_peaks]  # Get actual max values

    last_5_HR = local_updates["HR"][:index][-5:]

    return np.mean(last_5_HR), np.mean(last_5_max), np.mean(last_5_min)

def parallel_simulations(param_samples, storage, n_jobs):
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, storage) for params in param_samples)
    return results


if __name__ == "__main__":

    t_eval = np.linspace(0, t_span[1], t_span[1] * 1000)

    # Compute nominal (midpoint) values
    nominal_values = {k: (v[0] + v[1]) / 2 for k, v in parameters_change.items()}
    nominal_values["Kbg"] = 17.4
    nominal_values["KcMRV"] = 1
    nominal_values["KpCO2"] = 0.2025
    nominal_values["KpO2"] = 4.72e-09
    nominal_values["Tc"] = 0.75
    nominal_values["T_im"] = 1
    # nominal_values["C2"] = 40
    # nominal_values["K2"] = 25

    param_keys = list(parameters_change.keys())

    # Create OAT samples
    param_samples = []
    for key, (low, high) in parameters_change.items():
        # Low sample
        low_sample = [nominal_values[k] if k != key else low for k in param_keys]
        param_samples.append(low_sample)

        # High sample
        high_sample = [nominal_values[k] if k != key else high for k in param_keys]
        param_samples.append(high_sample)

    # Convert to NumPy array (same type/shape as lhd.sample(1000))
    X_oat = np.array(param_samples)

    param_samples = [dict(zip(param_keys, row)) for row in X_oat]
    print(f"Number of samples created: {len(X_oat)}")

    Result = parallel_simulations(param_samples, Next_Conditions, n_jobs=-1)

    print(Result)

    np.save('X_samples_HR_P_sys_P_dia.npy', X_oat)
    np.save('Result_HR_P_sys_P_dia.npy', Result)
