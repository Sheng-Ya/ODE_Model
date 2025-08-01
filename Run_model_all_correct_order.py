import numpy as np
import bisect

import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from line_profiler import LineProfiler
from collections import deque

import Resp_Control_Breath_Optimiser
from All_derivatives import model_derivatives
from Cardiovascular_controller import cardiovascular_controller
from Cardiovascular_system_new import cardiovascular_system
from Gas_Exchange import gas_exchange
from Parameters import Parameters
from Resp_Control_Ventilation import resp_control_vent

# from Respiratory_Mechanics import respiratory_mechanics

# from Entire_system.All_Cardiovascular_controller import cardiovascular_controller
# from Entire_system.All_Cardiovascular_system import cardiovascular_system
# from Entire_system.All_Gas_exchange import gas_exchange
# from Entire_system.All_Parameter_ranges import Parameters
# from Entire_system.All_Respiratory_controller import resp_control_vent

from Selected_Conditions import Selected_Conditions as previous_Selected_Conditions
# from Initial_Conditions import Initial_Conditions
# from Next_Conditions import Next_Conditions
from Initial_Conditions_after_running_again import Initial_Conditions
from Next_Conditions_all_derivatives import Next_Conditions

# output_file1 = "Selected_Conditions_new.py"
# output_file2 = "Initial_Conditions_new.py"
# output_file3 = "Next_Conditions_new.py"


target_values = np.arange(0, 10000, 10)
t_span = (0, 100) # Simulate for 30 seconds for just the cardiovascular system for global sensitivity

time_saved = 0.005
BUFFER_LIMIT = 20000

min_time = 10 # Minimum time in seconds before checking
max_time = 1100 # Maximum time limit to avoid infinite loops
time_step = 10  # Chunk size per solve

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    if i > 1: # t != 0:
        latest_nonzero_index = (i - 1) % BUFFER_LIMIT
        latest_nonzero_value = all_time[latest_nonzero_index]
        if t < latest_nonzero_value:
            # num_removed = 6
            index = -1  # Set a default value for safety

            # Iterating through the buffer in circular order
            for j in range(BUFFER_LIMIT):
                logical_index = (latest_nonzero_index - j - 1) % BUFFER_LIMIT  # Traversing backwards
                if all_time[logical_index] < t:
                    index = (logical_index + 1) % BUFFER_LIMIT
                    break

            num_removed = (actual_index - index) if (actual_index - index) >= 0 else BUFFER_LIMIT + (actual_index - index)

            for j in range(num_removed):
                all_time[(index + j) % BUFFER_LIMIT] = 0

            # if num_removed != 6:
            #     print(f"num_removed should be 6, got {num_removed}")
                # raise ValueError(f"num_removed should be 6, got {num_removed}")
        else:
            num_removed = 0
    else:
        num_removed = 0



    # Indices for slicing
    idx_resp_contr = num_cardio + num_cardio_control + num_gas + num_resp_control

    # Extract each subsystem's state variables
    resp_contr_state = Initial_Conditions_numpy[:idx_resp_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    derivatives_all = model_derivatives(t, resp_contr_state, Parameters, Initial_Conditions_dict, num_removed, t_span[0], i, BUFFER_LIMIT, all_time)


    # Initial_Conditions_dict["check_time"].append(t)
    # AAAAAA = list(Initial_Conditions_dict["f_sp_store"])
    # AAAAAAAA = list(Initial_Conditions_dict["P_sa"])
    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

    # AA = list(Initial_Conditions_dict["all_time"])
    # AAAAAAA = list(Initial_Conditions_dict["check_time"])

    # Debugging check for progress
    if t != 0:
        diff = np.abs(t - target_values)
        if np.any(diff < 0.0001):
            print(t)

    return derivatives_all

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc",
                         "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt"]
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
# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))
# IC_overall = IC_cardio


def simulate():
    # Initial setup
    IC_current = IC_overall.copy()

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        (0, max_time),
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(Parameters, Next_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control)
    )

    if ODE_solution.status == -1:
        return 0.0, 0.0, 0.0

    # Post-processing: use buffer to get recent data
    i_buffer = Next_Conditions["i"].item() % BUFFER_LIMIT

    P_sa = np.concatenate((Next_Conditions["P_sa_store"][i_buffer:], Next_Conditions["P_sa_store"][:i_buffer]))
    peaks, _ = find_peaks(P_sa, distance=int(500))
    troughs, _ = find_peaks(-P_sa, distance=int(500))

    last_10_troughs = troughs[-10:-1]
    last_10_min = P_sa[last_10_troughs]

    last_10_peaks = peaks[-10:-1]
    last_10_max = P_sa[last_10_peaks]

    HR = np.concatenate((Next_Conditions["HR_store"][i_buffer:], Next_Conditions["HR_store"][:i_buffer]))

    past_10_flat_segments = []
    prev_value = None
    for j in range(len(HR) - 1, -1, -1):
        current_value = HR[j]
        if current_value != prev_value:
            past_10_flat_segments.append(current_value)
            prev_value = current_value
            if len(past_10_flat_segments) == 10:
                break

    # np.savez(f'HR_vs_time.npz', HR=Next_Conditions["HR_check"], time=Next_Conditions["time_history"], HR_average = Next_Conditions["HR"])

    return ODE_solution, np.mean(past_10_flat_segments), np.mean(last_10_max), np.mean(last_10_min), IC_current, Next_Conditions, ODE_solution.t, ODE_solution.y


# def simulate():
#     # Solve ODE
#
#     IC_current = IC_overall.copy()
#     t0 = 0
#     total_time = 0
#
#     all_t = []
#     all_y = []
#
#     while total_time < max_time:
#         t_span_local = (t0, t0 + time_step)
#         t_eval_local = np.arange(t0, (t0 + time_step), 0.001)
#
#         ODE_solution = solve_ivp(
#             combined_system,
#             t_span_local,
#             IC_current,
#             t_eval=t_eval_local,
#             max_step=0.003,
#             method="RK45",
#             rtol=1e-3,
#             atol=1e-6,
#             args=(Parameters, Next_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control, time_saved)
#         )
#
#         if ODE_solution.status == -1:
#             return 0.0, 0.0, 0.0
#
#
#         # Append to full history
#         all_t.append(ODE_solution.t)
#         all_y.append(ODE_solution.y)
#
#
#         i_buffer = Next_Conditions["i"].item() % BUFFER_LIMIT
#
#         P_sa = np.concatenate((Next_Conditions["P_sa_store"][i_buffer:], Next_Conditions["P_sa_store"][:i_buffer]))
#
#         peaks, _ = find_peaks(P_sa, distance=int(500))  # Adjust distance based on heart rate
#         troughs, _ = find_peaks(-P_sa, distance=int(500))  # Find minima (inverted peaks)
#
#         last_10_troughs = troughs[-10:-1]  # Get indices of last 5 minima
#         last_10_min = P_sa[last_10_troughs]  # Get actual minimum values
#
#         last_10_peaks = peaks[-10:-1]  # Get indices of last 5 max
#         last_10_max = P_sa[last_10_peaks]  # Get actual max values
#
#         # Get past 10 HR
#         HR = np.concatenate((Next_Conditions["HR_store"][i_buffer:], Next_Conditions["HR_store"][:i_buffer]))
#
#         # Initialize list of segments
#         past_10_flat_segments = []
#
#         # Start from the end and track the current segment value
#         prev_value = None
#         for j in range(len(HR) - 1, -1, -1):
#             current_value = HR[j]
#             if current_value != prev_value:
#                 # New segment found
#                 past_10_flat_segments.append(current_value)
#                 prev_value = current_value
#                 if len(past_10_flat_segments) == 10:
#                     break
#
#
#
#         # Update IC and time
#         IC_current = ODE_solution.y[:, -1]
#         t0 += time_step
#         total_time += time_step
#
#         # Only check convergence after the minimum time has passed
#         if total_time >= min_time and len(past_10_flat_segments) >= 10 and t0>200:
#             minHR = np.min(past_10_flat_segments)
#             maxHR = np.max(past_10_flat_segments)
#
#             print(minHR, maxHR)
#
#             if abs(maxHR - minHR) < 0.05:
#                 break
#
#     # Concatenate time and state arrays
#     t_full = np.concatenate(all_t)
#     y_full = np.hstack(all_y)
#
#     return ODE_solution, np.mean(past_10_flat_segments), np.mean(last_10_max), np.mean(last_10_min), IC_current, Next_Conditions, t_full, y_full


if __name__ == "__main__":

    # lp = LineProfiler()
    # lp.add_function(Resp_Control_Breath_Optimiser.objective)
    #
    # lp.add_function(model_derivatives)
    # lp.enable()
    solution, HR, Psys, Pdia, save_IC, save_Next, t_full, y_full = simulate()
    print("ODE Status:", solution.status)
    print("ODE Message:", solution.message)

    np.save(f'IC_final.npy', save_IC)  # individual chunks
    np.save(f'Next_final.npy', save_Next)  # individual chunks
    # lp.disable()
    # lp.print_stats()

    # Save chunk incrementally (appending)
    # np.save(f'IC_final.npy', save_IC)  # individual chunks
    # np.save(f'Next_final.npy', save_Next)  # individual chunks

    time = solution.t
    state_variables = solution.y

    state_variable_names = (
            required_cardio_keys +
            required_cardio_control_keys +
            required_gas_keys +
            required_resp_control_keys
    )

    index = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 1
    print(HR)
    print(len(Next_Conditions["time_history"][:index]))


    i = Next_Conditions["i"].item() % BUFFER_LIMIT
    sorted_times = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))


    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # plt.figure(figsize=(12, 6))
    # for i, label in enumerate(required_cardio_keys):
    #     plt.plot(t_full, y_full[i])
    #     # color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
    #     # plt.plot(time, state_variables[i], label=label, color=color, linestyle='-', markersize=4)
    #
    # plt.xlabel("Time [s]")
    # plt.ylabel("State value")
    # plt.title("State Variables Over Entire Simulation")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh"][:index], label="Heart sympathetic", color="m")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh_delay2"][:index], label="Delay Heart sympathetic", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v"][:index], label="Vagal firing", color="c")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v_delay02"][:index], label="Delay Vagal firing", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Tv_change"][:index], label="Tv_change", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Tv"][:index], label="sigma_Tv", color="c")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # # plt.show()
    ax2 = ax1.twinx()
    #
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR", color="r")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Q_lv"][47500:index],
             label="Q_lv (leaving LV)")
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Q_la"][47500:index], label="Q_la (into LA)")
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Q_ra"][47500:index], label="Q_ra (into RA)")
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Q_rv"][47500:index],
             label="Q_rv (leaving RV/into pul art)")
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Qi_lv"][47500:index], label="Qi_lv")
    ax1.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["Qi_rv"][47500:index], label="Qi_rv")

    # Add labels and legend
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Flow (mL/s)")
    ax1.legend(loc="upper left")
    plt.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["theta_mi"][47500:index], label="theta_mi",
             color='c')
    ax2.plot(Next_Conditions["time_history"][47500:index], Next_Conditions["theta_tr"][47500:index], label="theta_tr",
             color='y')
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pa"][:index], label="Q_pa", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rv"][:index], label="Q_rv", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pp"][:index], label="Q_pp", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_vc"][:index], label="Q_vc", color="b")
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_la"][:index], label="VT_la", color="r")

    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color='g')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_pv"][:index], label="P_pv", color='k')

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR", color="r")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pa"][:index], label="Q_pa", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rv"][:index], label="Q_rv", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pp"][:index], label="Q_pp", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_ra"][:index], label="Q_ra", color="r")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_la"][:index], label="Q_la", color="r")


    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    #
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_vc"][:index], label="P_vc", color='g')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_ra"][:index], label="P_ra", color='k')

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR", color="r")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["CvtCO2"][:index], label="CvtCO2", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["CvbCO2"][:index], label="CvbCO2", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_CO2"][:index], label="Cv_CO2", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["QT"][:index], label="QT", color="r")


    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_ab"][:index], label="Baroreceptor firing", color="aquamarine")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["check_store_all_time"][:index], label="Mean baroreceptor firing", color="g")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_ap"][:index], label="Lung stretch receptor firing", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_ac"][:index], label="Chemoreceptor firing", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Nt"][:index], label="Respiratory neuromuscular drive", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v"][:index], label="Vagal firing", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh"][:index], label="Heart sympathetic", color="m")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["prev_flat_bit_store"][:index], label="Heart sympathetic", color="r")

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh_delay2"][:index],
    #          label="Delay Heart sympathetic", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v"][:index], label="Vagal firing", color="c")
    #
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v_delay02"][:index],
    #          label="Delay Vagal firing", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Tv_change"][:index], label="Tv_change",
             color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Tv"][:index], label="sigma_Tv", color="c")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change",
             color='g')

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sv"][:index], label="theta_sv", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sp"][:index], label="theta_sp", color="r")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_v"][:index], label="theta_v", color="aquamarine")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # # plt.show()
    ax2 = ax1.twinx()
    #
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_sh"][:index], label="theta_sh", color="m")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR", color="r")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    plt.show()


    variables_to_plot = [
        # "f_sp_history", "f_sh_history", "f_v_history",
        # "xb_CO2", "P_sp", "P_bv", "Q_bp", "beta","U2", "T", "xb_O2", "Cvb_O2"
        # "PamCO2", "VE_integral"
        # "phi", "phi_atr"
        # "f_sp_history", "f_sh_history", "f_v_history", "phi_met_history", "f_sv_history",
        # "Vflow_ua", "P_ua", "P_musc", "dV_dt", "V",
        # "Pd_5_O2"
        "PA_O2", "PA_CO2", "PA_CO2_delay", "PA_O2_delay", "Pa_O2", "Pa_CO2", "finish_breath_time_plot", "Ca_O2", "Cv_O2", "Ca_CO2", "Cv_CO2", "PvtO2", "VAflow", "f_ac_history", "Q_pp", "PvtCO2", "dV_dt"
        # , "VT", "VE_flow", "VAflow", "Q_pp", "V", "PA_O2_old", "PA_CO2_old","Cv_CO2", "Ca_CO2", "Cv_O2",
        # "Ca_O2", "dPA_CO2_dt", "dPA_O2_dt",
        # "dCvO2_dt", "dCvCO2_dt", "PA_CO2", "QT", "PA_O2",  # "V", "Cv_O2", "Ca_O2"
        # "Vu_ev", "Vu_amv", "Vu_rmv", "Vu_sv", "R_ep", "R_amp", "R_rmp", "R_sp",
        # "R_bp", "R_hp", "Emax_lv", "Emax_rv", "I", "phi_met", "Nt",
        # "Vu_sv_change", "prev_flat_bit", "Pa_O2", "HR"
    ]

    for key in variables_to_plot:
        if key in Next_Conditions:  # Check if the key exists in updates
            plt.figure(figsize=(8, 4))  # Create a new figure for each variable
            plt.plot(Next_Conditions["time_history"][:index], Next_Conditions[key][:index], label=key, linewidth=2)
            plt.xlabel("Time (s)")
            plt.ylabel(key)
            plt.title(f"Plot of {key} over Time")
            plt.legend()
            plt.grid(True)
            plt.show()


    # Plot all state variables
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(state_variable_names):
        # if label != "beta":
        if label in ["VT_pa", "VT_pp", "VT_pv", "Q_pa",
        "VT_la", "VT_lv", "VT_ra", "VT_rv",
        "VT_sv", "VT_bv", "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", 'VT_vc',
        "theta_ao", "dtheta_ao_dt", "theta_po", 'dtheta_po_dt', "theta_mi", 'dtheta_mi_dt', "theta_tr", 'dtheta_tr_dt',

     # Cardio controller state variables
        "theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh",
        "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", 'R_ep_change', "R_sp_change",
        "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change",
        "Emax_rv_change", "Ts_change", "Tv_change", 'xb_O2', "xb_CO2", "xh_O2", 'xh_CO2', "Wh", 'xrm_O2', 'xrm_CO2', 'xam_O2', "xM", "x_met"]:  # Skip "Wh"
            continue
        color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
        plt.plot(time, state_variables[i], label=label, color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()



    # HR = np.concatenate((Next_Conditions["HR_store"][i:], Next_Conditions["HR_store"][:i]))
    #
    # # Initialize list of segments
    # flat_segments = []
    #
    # # Start from the end and track the current segment value
    # prev_value = None
    # for j in range(len(HR) - 1, -1, -1):
    #     current_value = HR[j]
    #     if current_value != prev_value:
    #         # New segment found
    #         flat_segments.append(current_value)
    #         prev_value = current_value
    #         if len(flat_segments) == 10:
    #             break
    #
    # print("Last 10 unique flat segments:", flat_segments)
    #
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, HR, label="HR", color="r")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()
    #
    #
    #
    #
    #
    # P_sa = np.concatenate((Next_Conditions["P_sa_store"][i:], Next_Conditions["P_sa_store"][:i]))
    #
    # peaks, _ = find_peaks(P_sa, distance=int(500))  # Adjust distance based on heart rate
    # troughs, _ = find_peaks(-P_sa, distance=int(500))  # Find minima (inverted peaks)
    #
    # last_10_troughs = troughs[-10:-1]  # Get indices of last 5 minima
    # last_10_min = P_sa[last_10_troughs]  # Get actual minimum values
    #
    # last_10_peaks = peaks[-10:-1]  # Get indices of last 5 max
    # last_10_max = P_sa[last_10_peaks]  # Get actual max values
    #
    # print(np.mean(last_10_max), np.mean(last_10_min))
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(sorted_times, P_sa, label="P_sa")
    #
    # ax1.scatter(sorted_times[troughs], P_sa[troughs], color='r', marker='o', label="Detected Minima")
    # ax1.scatter(sorted_times[peaks], P_sa[peaks], color='g', marker='x', label="Detected Maxima")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()






    # start_index = index - 10000
    #
    # # Create a new dictionary for the delays
    # selected_conditions = {key: Next_Conditions[key] for key in ["f_sp", "f_sh", "f_v", "f_sv", "phi_met", "time_history", "PA_O2", "PA_CO2"]}
    #
    # # Save to a new Python file
    # with open(output_file1, 'w') as f:
    #     f.write('import numpy as np\n\n')
    #     f.write('Selected_Conditions = {\n')
    #     for key, value in selected_conditions.items():
    #         f.write(f"    '{key}': np.array({value[start_index:index].tolist()}),\n")
    #     f.write('}\n')
    #
    #
    # # new initial state variables
    # final_values = state_variables[:, -1]  # last time point
    #
    # # Open the new file for writing
    # with open(output_file2, "w") as f:
    #     f.write("Initial_Conditions = {\n")
    #
    #     for name, value in zip(state_variable_names, final_values):
    #         f.write(f'    "{name}": {value},\n')  # adjust format as needed
    #
    #     f.write("}\n")
    #
    # # Output file path
    # output_file = "Next_Conditions_new.py"
    #
    # with open(output_file3, "w") as f:
    #     f.write("import numpy as np\n\n")
    #     f.write("Next_Conditions = {\n")
    #
    #     # Ensure 'i' and all other non populated arrays are set correctly
    #     # f.write(f'    "i": np.array([{index}]),\n\n')
    #     f.write(f'    "i": np.array([{0}]),\n')
    #     f.write(f'    "time_since_beat": np.pad(np.array([{Next_Conditions["time_since_beat"][index - 1]}, {Next_Conditions["time_since_beat"][index]}]), (0, 1200000 - 2), mode="constant", constant_values=1e6),\n')
    #     f.write(f'    "Nd": {Next_Conditions["Nd"][-5:]},\n\n')
    #
    #     for key, array in Next_Conditions.items():
    #         if key in ["i", "time_since_beat", "Nd"]:
    #             continue  # Already handled
    #         if isinstance(array, np.ndarray) and array.ndim == 1 and len(array) > index:
    #             last_value = array[index]
    #             f.write(
    #                 f'    "{key}": np.pad(np.array([{last_value}]), (0, 1200000 - 1), mode="constant", constant_values=1e6),\n')
    #         else:
    #             if isinstance(array, (list, np.ndarray)):
    #                 values = list(array)
    #             else:
    #                 values = [array]
    #
    #             f.write(f'    "{key}": {values},\n')
    #
    #     f.write("}\n")

    # index_start = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 100000
    # # state variables excel
    # if index > 100000:
    #     index_start = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 100000
    #
    #     # Transpose state variables so that each row is a time point and columns are variable values
    #     df = pd.DataFrame(data=solution.y[:, -100000:].T, columns=state_variable_names)
    #     df.insert(0, "time", solution.t[-100000:])
    #     df.to_csv("C:/Users/vanes/Documents/state_variables_output.csv", index=False)
    # else:
    #     index_start = 0
    #     df = pd.DataFrame(data=solution.y[:,:].T, columns=state_variable_names)
    #     df.insert(0, "time", solution.t[:])
    #     df.to_csv("C:/Users/vanes/Documents/state_variables_output.csv", index=False)

    index_start = 0
    # Next_Conditions excel
    # Build a dictionary of shortened arrays
    # data = {
    #     key: val[index_start:index + 1]
    #     for key, val in Next_Conditions.items()
    #     if (
    #             isinstance(val, np.ndarray)
    #             and val.ndim >= 1  # Only arrays with at least 1 dimension
    #             and len(val) > index
    #     )
    # }
    #
    # # Ensure time_history is first
    # columns = ["time_history"] + [k for k in data if k != "time_history"]
    # nextdf = pd.DataFrame({k: data[k] for k in columns})
    #
    # nextdf.to_parquet("C:/Users/vanes/Documents/Next_Conditions_Output.parquet", index=False)
    # nextdf.to_csv("C:/Users/vanes/Documents/Next_Conditions_Output.csv", index=False)


    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][(index - 10000):index], Next_Conditions["theta_po"][(index - 10000):index],
    #          label="theta_po")
    # ax1.plot(Next_Conditions["time_history"][(index - 10000):index], Next_Conditions["theta_ao"][(index - 10000):index],
    #          label="theta_ao")
    # ax1.plot(Next_Conditions["time_history"][(index - 10000):index], Next_Conditions["theta_mi"][(index - 10000):index],
    #          label="theta_mi")
    # ax1.plot(Next_Conditions["time_history"][(index - 10000):index], Next_Conditions["theta_tr"][(index - 10000):index],
    #          label="theta_tr")
    # ax1.legend(loc="upper left")
    # plt.show()

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_pa"][:index], label="P_pa")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_pp"][:index], label="P_pp")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_rv"][:index], label="P_rv")
    #
    # # Add labels and legend
    # ax1.set_ylabel("Pressure (mmHg)")
    # ax1.set_xlabel("Time (s)")
    # ax1.legend(loc="upper left")
    #
    # ax2 = ax1.twinx()
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rv"][:index], label="Q_rv (leaving RV/into pul art)", color="r")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pa"][:index], label="Q_pa", color="k")
    #
    # # ax2.set_ylabel("Flow (mL/s)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    # plt.show()
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][(index-5000):index], Next_Conditions["theta_ao"][(index-5000):index], label="theta_po")
    # plt.show()





    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["A"][:index], label="A")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sa"][:index], label="Q_sa")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sp"][:index], label="Q_sp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pp"][:index], label="Q_pp")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_ep"][:index], label="Q_ep")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sp"][:index], label="Q_sp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_bp"][:index], label="Q_bp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_hp"][:index], label="Q_hp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rmp"][:index], label="Q_rmp")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_amp"][:index], label="Q_amp")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_lv"][:index], label="Q_lv (leaving LV)")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_la"][:index], label="Q_la (into LA)")
    #
    # plt.xlabel("Time (s)")
    # # plt.ylabel("f")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sa"][:index], label="Q_sa", color="r")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_lv"][:index], label="Q_lv", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pmax_lv"][:index], label="Pmax_lv", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa", color="b")

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_CO2"][:index], label="PA_CO2", color="k")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pb_CO2"][:index], label="Pb_CO2", color="c")

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V", color="k")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VE_flow"][:index], label="VE_flow", color="b")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # ax2 = ax1.twinx()
    #
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ts_change"][:index], label="Ts_change", color='g')
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["theta_ao"][:index], label="theta_ao", color="r")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi"][:index], label="phi", color="aquamarine")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi_atr"][:index], label="phi_atr", color="c")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="VT_lv", color="y")
    #
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR", color="r")
    # # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["sigma_Ts"][:index], label="sigma_Ts", color="y")
    #
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    plt.show()






    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR Averaged")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR"][:index], label="HR")

    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Emax_lv"][:index], label="Emax_lv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Emax_rv"][:index], label="Emax_rv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["I"][:index], label="I")
    #
    #
    #
    # Add labels and legend
    plt.ylabel("")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR"][:index], label="HR")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Emax_lv"][:index], label="Emax_lv")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Emax_rv"][:index], label="Emax_rv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["I"][:index], label="I")
    #
    #
    #
    # Add labels and legend
    plt.ylabel("")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.grid(True)
    plt.show()




    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt", color="c")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V", color="k")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    #
    # ax2 = ax1.twinx()
    #
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_O2"][:index], label="Pd_5_O2", color="g")
    # # ax2.set_ylabel("Flow (mL/s)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    # plt.show()

    # # get max's plot with Pmax_la instead of P_la
    # plt.plot(local_updates["time_history"][:index], local_updates["P_lv"][:index], label="LV")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Pressure (mmHg)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # # last_beat_time = Next_Conditions["time_history"][index] - (1/Next_Conditions["HR"][index])
    # P_sa_smooth = Next_Conditions["P_sa"][:index]
    #
    # # last_beat_index = bisect.bisect_left(Next_Conditions["time_history"][:index], last_beat_time)
    # # no_of_points = index - last_beat_index
    #
    # peaks, _ = find_peaks(P_sa_smooth, distance=int(500))  # Adjust distance based on heart rate
    # troughs, _ = find_peaks(-P_sa_smooth, distance=int(500))  # Find minima (inverted peaks)
    #
    # last_5_troughs = troughs[-6:-1]  # Get indices of last 5 minima
    # last_5_min = P_sa_smooth[last_5_troughs]  # Get actual minimum values
    #
    # last_5_peaks = peaks[-6:-1]  # Get indices of last 5 max
    # last_5_max = P_sa_smooth[last_5_peaks]  # Get actual max values
    #
    # diff = last_5_max - last_5_min
    # mean_diff = np.mean(diff)
    #
    # last_5_HR = Next_Conditions["HR"][:index][-5:]

    # print(diff)
    # print(mean_diff)
    # print(last_5_HR)
    # print(np.mean(last_5_HR))

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa")
    # ax1.plot(Next_Conditions["time_history"][:index], P_sa_smooth, label="P_sa_smooth")
    # # ax1.plot(Next_Conditions["time_history"][:index], -V_lv_smooth, label="V_lv_smooth")
    #
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["T"][:index], label="T")
    #
    #
    # ax1.scatter(Next_Conditions["time_history"][troughs], P_sa_smooth[troughs], color='r', marker='o', label="Detected Minima")
    # ax1.scatter(Next_Conditions["time_history"][peaks], P_sa_smooth[peaks], color='g', marker='x', label="Detected Maxima")
    #
    # # for i in range(0, len(Next_Conditions["V_lv"][:index]), int(500)):
    # #     plt.axvline(x=Next_Conditions["time_history"][i], color='r', alpha=0.5)  # Dashed red line
    #
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_lv"][:index], label="V_lv")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["cO2_diff"][:index], label="cO2_diff")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_O2"][:index], label="Cv_O2", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_O2"][:index], label="Ca_O2", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_CO2"][:index], label="Ca_CO2", color="r")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_CO2"][:index], label="Cv_CO2", color="k")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["FO2"][:index], label="FO2", color="m")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["FCO2"][:index], label="FCO2", color="c")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["QT"][:index], label="QT", color="k")
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()






    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["cCO2_diff"][:index], label="cCO2_diff")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["cO2_diff"][:index], label="cO2_diff")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_vc"][:index], label="Q_vc")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sa"][:index], label="Q_sa")
    #
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_CO2"][:index], label="Pd_5_CO2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_O2"][:index], label="Pd_5_O2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt")
    # ax1.axhline(y=0.25/60, color="r", label="MRO2")
    # ax1.axhline(y=-0.2/60, color="k", label="MRCO2")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    # variables_to_plot = [
    #     # "f_sp_history", "f_sh_history", "f_v_history",
    #     # "xb_CO2", "P_sp", "P_bv", "Q_bp", "beta","U2", "T", "xb_O2", "Cvb_O2"
    #     # "PamCO2", "VE_integral"
    #     # "phi", "phi_atr"
    #     # "f_sp_history", "f_sh_history", "f_v_history", "phi_met_history", "f_sv_history",
    #     # "Vflow_ua", "P_ua", "P_musc", "dV_dt", "V",
    #     # "Pd_5_O2"
    #     "VAflow", "f_ac_history", "Q_pp", "PvtCO2", "V", "dV_dt"# , "VT", "VE_flow", "VAflow", "Q_pp", "V", "PA_O2_old", "PA_CO2_old","Cv_CO2", "Ca_CO2", "Cv_O2",
    #     # "Ca_O2", "dPA_CO2_dt", "dPA_O2_dt",
    #     # "dCvO2_dt", "dCvCO2_dt", "PA_CO2", "QT", "PA_O2",  # "V", "Cv_O2", "Ca_O2"
    #     # "Vu_ev", "Vu_amv", "Vu_rmv", "Vu_sv", "R_ep", "R_amp", "R_rmp", "R_sp",
    #     # "R_bp", "R_hp", "Emax_lv", "Emax_rv", "I", "phi_met", "Nt",
    #     # "Vu_sv_change", "prev_flat_bit", "Pa_O2", "HR"
    # ]
    #
    # for key in variables_to_plot:
    #     if key in Next_Conditions:  # Check if the key exists in updates
    #         plt.figure(figsize=(8, 4))  # Create a new figure for each variable
    #         plt.plot(Next_Conditions["time_history"][:index], Next_Conditions[key][:index], label=key, linewidth=2)
    #         plt.xlabel("Time (s)")
    #         plt.ylabel(key)
    #         plt.title(f"Plot of {key} over Time")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dPA_CO2_dt"][:index], label="dPA_CO2_dt")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dPA_O2_dt"][:index], label="dPA_O2_dt")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_CO2"][:index], label="Pd_5_CO2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_O2"][:index], label="Pd_5_O2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    # # Number of state variables
    # num_variables = state_variables.shape[0]
    # colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors
    #
    # # Plot all state variables
    # plt.figure(figsize=(14, 10))
    #
    # for i, label in enumerate(required_gas_keys):
    #     # if label == "Pd_2_O2":  # Skip "VT_sv"
    #     #     continue
    #     color = colors[
    #         i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
    #     plt.plot(time, state_variables[len(required_cardio_keys + required_cardio_control_keys) + i], label=label,
    #              color=color, linestyle='-', markersize=4)
    #
    # plt.xlabel("Time")
    # plt.ylabel("State Variables")
    # plt.title("Evolution of State Variables Over Time")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    # plt.grid()
    # plt.tight_layout()
    # plt.show()




    fig, ax1 = plt.subplots()
    print(len(Next_Conditions["time_history"][:index]))

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["time_since_beat"][:index], label="time_since_beat", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi"][:index], label="phi", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi_atr"][:index], label="phi_atr", color="k")


    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()

    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="VT_lv", color="g")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR"][:index], label="HR", color="g")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR averaged", color="r")
    # ax2.set_ylabel("Flow (mL/s)", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    # #
    plt.show()



    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv")

    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sp"][:index], label="P_sp")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_ev"][:index], label="P_ev")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_hv"][:index], label="P_hv")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_bv"][:index], label="P_bv")



    plt.xlabel("Time (s)")
    # plt.ylabel("f")
    plt.legend()
    plt.grid(True)
    plt.show()





    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["G_AW_guess"][:index], label="G_AW")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vflow_ua"][:index], label="Vflow_ua")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_ua"][:index], label="P_ua")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_musc"][:index], label="P_musc")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_pl"][:index], label="P_pl")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_la"][:index], label="Q_la")
    # # plt.plot(Next_Conditions["Q_la"][:index], Next_Conditions["Q_la"][:index], label="Q_la")
    #
    #
    #
    #
    # # plt.plot(Next_Conditions["U2"][:index], label="U2")
    # plt.xlabel("Time (s)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # plt.plot(Next_Conditions["time_history"])
    # plt.show()

    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_O2"][:index], label="Pa_O2")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Nt"][:index], label="Nt")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_CO2"][:index], label="PaCO2")

    # plt.xlabel("Time (s)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    plt.plot(Next_Conditions["VT_lv"][:index], Next_Conditions["P_lv"][:index], label="LV")  # 10 s all    #
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["VT_rv"][:index], Next_Conditions["P_rv"][:index], label="rV")  # 10 s all    #
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["VT_ra"][:index], Next_Conditions["P_ra"][:index], label="RA")  #
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()

    # get max's plot with Pmax_la instead of P_la
    plt.plot(Next_Conditions["VT_la"][:index], Next_Conditions["P_la"][:index], label="LA")
    # plt.plot(Next_Conditions["VT_ra"][:index], Next_Conditions["P_ra"][:index], label="LA")

    # # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()



    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="VT_lv", color="m")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_la"][:index], label="VT_la", color="y")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="b")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="c")

    ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_title("R_la and R_ra = 0.025 mmHg.s/ml")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # #
    # # Create second y-axis for volume
    ax2 = ax1.twinx()


    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_pa"][:index], label="V_pa", color="g")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_pv"][:index], label="V_pv", color="r")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_pp"][:index], label="V_pp", color="b")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_ra"][:index], label="V_ra", color="r")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_rv"][:index], label="V_rv", color="g")

    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="b")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="c")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa", color="k")


    # ax2.set_ylabel("Flow (mL/s)", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    # #
    plt.show()


    fig, ax1 = plt.subplots()
    #
    # # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pmax_la"][:index], label="Pmax_la", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_rv"][:index], label="VT_rv", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="VT_lv", color="m")



    ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_title("R_la and R_ra = 0.025 mmHg.s/ml")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="b")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="c")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()


    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_ev"][:index], label="Vu_ev")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_amv"][:index], label="Vu_amv")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_rmv"][:index], label="Vu_rmv")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_sv"][:index], label="Vu_sv")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_ep"][:index], label="R_ep")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_amp"][:index], label="R_amp")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_rmp"][:index], label="R_rmp")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_sp"][:index], label="R_sp")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_bp"][:index], label="R_bp")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["R_hp"][:index], label="R_hp")

    # Add labels and legend
    plt.ylabel("")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.grid(True)
    plt.show()



    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="VT_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_la"][:index], label="VT_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_ra"][:index], label="VT_ra (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_rv"][:index], label="VT_rv (Right Ventricle)")

    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_sv"][:index], label="V_sv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_rmv"][:index], label="V_rmv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_amv"][:index], label="V_amv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_vc"][:index], label="V_vc", color="c")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["V_hv"][:index], label="V_hv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_ev"][:index], label="VT_ev")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_sv"][:index], label="Vu_sv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_rmv"][:index], label="Vu_rmv")
    # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_amv"][:index], label="Vu_amv")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_vc"][:index], label="Vu_vc", linestyle="dashed", color="c")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_hv"][:index], label="Vu_hv")
    # # plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_ev"][:index], label="Vu_ev")
    #
    # # Add labels and legend
    # plt.xlabel("Time (s)")
    # plt.ylabel("Volume (mL)")
    # plt.title("Volume Traces")
    # plt.legend()
    # plt.grid(True)
    # plt.show()



    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_lv"][7500:index], label="Q_lv (leaving LV)")
    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_la"][7500:index], label="Q_la (into LA)")
    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_ra"][7500:index], label="Q_ra (into RA)")
    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_rv"][7500:index], label="Q_rv (leaving RV/into pul art)")
    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Qi_lv"][7500:index], label="Qi_lv")
    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Qi_rv"][7500:index], label="Qi_rv")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()


    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_la"][7500:index], label="Q_la (into LA)")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_rv"][7500:index],
             label="Q_rv (leaving RV/into pul art)")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Q_ra"][7500:index], label="Q_ra (into RA)")
    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Qi_rv"][7500:index], label="Qi_rv")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["time_history"][7500:index], Next_Conditions["Qi_lv"][7500:index], label="Qi_lv")
    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()






    # Number of state variables
    num_variables = state_variables.shape[0]

    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot cardio control variables
    plt.figure()
    for i, label in enumerate(required_cardio_control_keys):
        # if label != "beta":
        if label != "P_tilda":  # Skip "Wh"
            continue
        color = colors[i % len(colors)]
        plt.plot(time, state_variables[len(required_cardio_keys) + i], label=label, color=color, linestyle='-',
                 markersize=4)

    plt.xlabel("Time")
    plt.ylabel("Cardio Control Variables")
    plt.title("Evolution of Cardio Control Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()
