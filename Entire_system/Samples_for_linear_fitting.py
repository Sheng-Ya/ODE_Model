import os

import numpy as np
import tqdm_joblib

from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm

from All_Parameter_ranges import parameters as parameters_change

from All_Cardiovascular_controller import cardiovascular_controller
from All_Cardiovascular_system import cardiovascular_system
from All_Gas_exchange import gas_exchange
from Parameters import Parameters as Old_Parameters
from All_Respiratory_controller import resp_control_vent


from Selected_Conditions import Selected_Conditions as previous_Selected_Conditions
from Initial_Conditions_after_running_again import Initial_Conditions
from All_Next_Conditions import Next_Conditions


target_values = np.arange(0, 10000, 10)
t_span = (0, 60) # Simulate for 30 seconds for just the cardiovascular system for global sensitivity

time_saved = 0.005
BUFFER_LIMIT = 10000

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Current_Parameters, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, Parameters):
    """

    """

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    if t != 0:
        all_time = Initial_Conditions_dict["all_time"]
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:

            index = -1  # Set a default value for safety

            # Iterating through the buffer in circular order
            for j in range(BUFFER_LIMIT):
                logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
                if all_time[logical_index] == t or all_time[logical_index] < t:
                    index = (logical_index + 1) % BUFFER_LIMIT
                    break

            num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (
                        actual_index - index)

            for j in range(num_removed + 1):
                Initial_Conditions_dict["all_time"][(index + j) % BUFFER_LIMIT] = 1e6

            if num_removed > 5:
                raise ValueError(f"num_removed should not be greater than 5, got {num_removed}")
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
    d_cardio = cardiovascular_system(t, cardio_state, Current_Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, t_span[0], i, BUFFER_LIMIT, Parameters)
    d_cardio_contr = cardiovascular_controller(t, cardio_contr_state, Current_Parameters, Initial_Conditions_dict["all_time"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, t_span[0], previous_Selected_Conditions, i, BUFFER_LIMIT, Parameters)
    d_gas = gas_exchange(t, gas_state, Current_Parameters, Initial_Conditions_dict["all_time"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, t_span[0], previous_Selected_Conditions, i, BUFFER_LIMIT, Parameters)
    d_resp_vent = resp_control_vent(t, resp_contr_state, Current_Parameters, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, t_span[0], i, BUFFER_LIMIT, Parameters)
    # d_resp_mech = respiratory_mechanics(t, resp_mech_state, Current_Parameters, Initial_Conditions_dict, num_removed, i)

    d_combined = np.concatenate((d_cardio, d_cardio_contr, d_gas, d_resp_vent))

    Initial_Conditions_dict["all_time"][(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1

    # # Debugging check for progress
    # if t != 0:
    #     diff = np.abs(t - target_values)
    #     if np.any(diff < 0.0001):
    #         print(t)

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
        'HR1', 'Vu_ev1', 'Vu_sv1', 'Vu_rmv1', 'Vu_amv1', 'Emax_lv1', 'Emax_rv1',
        'Pa_O2_history', 'Pa_CO2_history', 'Pb_CO2_history',
        'PamO2', 'PamCO2', 'PmbCO2', 'Nd', 'finish_breath_time',
        "current_times", "P_musc_current", "V_current", "dV_dt_current",
    }

    local_updates = {
        key: value if key in list_keys else np.array(value, copy=True)
        for key, value in storage.items()
    }

    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_overall, t_eval=t_eval, max_step = 0.003, method="RK23", rtol=1e-3,
                             atol=1e-6, args=(Current_Parameters, local_updates, num_gas, num_cardio, num_cardio_control, num_resp_control, Old_Parameters))

    if ODE_solution.status == -1:
        # Integration failed or early termination
        return 0.0, 0.0, 0.0

    i_buffer = local_updates["i"].item() % BUFFER_LIMIT


    P_sa = np.concatenate((local_updates["P_sa_store"][i_buffer:], local_updates["P_sa_store"][:i_buffer]))

    peaks, _ = find_peaks(P_sa, distance=int(500))  # Adjust distance based on heart rate
    troughs, _ = find_peaks(-P_sa, distance=int(500))  # Find minima (inverted peaks)

    last_10_troughs = troughs[-10:-1]  # Get indices of last 5 minima
    last_10_min = P_sa[last_10_troughs]  # Get actual minimum values

    last_10_peaks = peaks[-10:-1]  # Get indices of last 5 max
    last_10_max = P_sa[last_10_peaks]  # Get actual max values


    # Get past 10 HR
    HR = np.concatenate((local_updates["HR_store"][i_buffer:], local_updates["HR_store"][:i_buffer]))

    # Initialize list of segments
    past_10_flat_segments = []

    # Start from the end and track the current segment value
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            # New segment found
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break

    return np.mean(past_10_flat_segments), np.mean(last_10_max), np.mean(last_10_min)
#
# def parallel_simulations(param_samples, sample_descriptions, storage, n_jobs):
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(lambda params, desc: (desc, simulate_cpu(params, storage)))(params, desc)
#         for params, desc in zip(param_samples, sample_descriptions)
#     )
#     return results

# def chunked(iterable, n):
#     """Yield successive n-sized chunks from iterable."""
#     for i in range(0, len(iterable), n):
#         yield iterable[i:i + n]

def chunked(iterable1, iterable2, n):
    """Yield successive n-sized chunks from two iterables."""
    for i in range(0, len(iterable1), n):
        yield iterable1[i:i + n], iterable2[i:i + n]


def parallel_simulations(param_samples, sample_descriptions, storage, n_jobs, chunk_size=240, save_path='Result_DGSM_chunked.npy'):
    results_all = []

    # If file exists from previous run, remove it to start fresh
    if os.path.exists(save_path):
        os.remove(save_path)

    # for i, chunk in enumerate(chunked(param_samples, chunk_size)):
    #     with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i+1)*chunk_size}", total=len(chunk))):
    #         results_chunk = Parallel(n_jobs=n_jobs)(delayed(simulate_cpu)(params, storage) for params in chunk)

    for i, (param_chunk, desc_chunk) in enumerate(chunked(param_samples, sample_descriptions, chunk_size)):

        with tqdm_joblib.tqdm_joblib(tqdm(desc=f"Sim {i * chunk_size}-{(i + 1) * chunk_size}", total=len(param_chunk))):
            results_chunk = Parallel(n_jobs=n_jobs)(
                delayed(lambda params, desc: (desc, simulate_cpu(params, storage)))(params, desc)
                for params, desc in zip(param_chunk, desc_chunk)
            )

        results_all.extend(results_chunk)
        np.save(save_path, np.array(results_all, dtype=object))  # dtype=object to store tuples properly


    return results_all


if __name__ == "__main__":

    t_eval = np.arange(0, t_span[1], 0.001)

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
    sample_descriptions = []

    nominal_sample = [nominal_values[k] for k in param_keys]
    param_samples.append(nominal_sample)
    sample_descriptions.append("NOMINAL")


    for key, (high, low) in parameters_change.items():
        # High sample
        high_sample = [nominal_values[k] if k != key else high for k in param_keys]
        param_samples.append(high_sample)
        sample_descriptions.append(f"{key} HIGH")

        # Low sample
        low_sample = [nominal_values[k] if k != key else low for k in param_keys]
        param_samples.append(low_sample)
        sample_descriptions.append(f"{key} LOW")

        # 25% sample
        q25 = low + 0.25 * (high - low)
        q25_sample = [nominal_values[k] if k != key else q25 for k in param_keys]
        param_samples.append(q25_sample)
        sample_descriptions.append(f"{key} Q25")

        # 75% sample
        q75 = low + 0.75 * (high - low)
        q75_sample = [nominal_values[k] if k != key else q75 for k in param_keys]
        param_samples.append(q75_sample)
        sample_descriptions.append(f"{key} Q75")

    # Convert to NumPy array (same type/shape as lhd.sample(1000))
    X_oat = np.array(param_samples)
    np.save('X_samples_HR_P_sys_P_dia_box_linear.npy', X_oat)

    # Print the order
    print("Simulation order:")
    for i, desc in enumerate(sample_descriptions):
        print(f"{i + 1:02d}: {desc}")

    param_samples = [dict(zip(param_keys, row)) for row in X_oat]
    print(f"Number of samples created: {len(X_oat)}")

    Result = parallel_simulations(param_samples, sample_descriptions, Next_Conditions, n_jobs=-1)

    print(Result)

    np.save('Result_HR_P_sys_P_dia_box_linear.npy', np.array(Result, dtype=object))

