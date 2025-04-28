import numpy as np
import bisect
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from line_profiler import LineProfiler
from collections import deque

from Gas_Exchange_just_resp import gas_exchange
from Initial_Conditions_just_resp import Initial_Conditions
from Next_Conditions_just_resp import Next_Conditions
from Parameters_just_resp import Parameters
from Resp_Control_Ventilation_just_resp import resp_control_vent
from Respiratory_Mechanics_just_resp import respiratory_mechanics


t_span = (0, 150) # Simulate for 30 seconds for just the cardiovascular system for global sensitivity
target_values = np.arange(0, 10000, 10)

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, Initial_Conditions_dict, num_gas, num_resp_control, num_resp_mech):
    """

    """
    i = Next_Conditions["i"].item()
    if t != 0:
        latest_nonzero_value = Next_Conditions["all_time"][i - 1]
        if t < latest_nonzero_value:
            index = bisect.bisect_left(Next_Conditions["time_history"], t)
            num_removed = i - index
            Next_Conditions["time_history"][index:i + 1] = np.full((num_removed + 1,), 1e6)
        else:
            num_removed = 0
    else:
        num_removed = 0

    # Indices for slicing
    idx_gas = num_gas
    idx_resp_contr = idx_gas + num_resp_control
    idx_resp_mech = idx_resp_contr + num_resp_mech

    # Extract each subsystem's state variables
    gas_state = Initial_Conditions_numpy[:idx_gas]
    resp_contr_state = Initial_Conditions_numpy[idx_gas:idx_resp_contr]
    resp_mech_state = Initial_Conditions_numpy[idx_resp_contr:idx_resp_mech]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    d_gas = gas_exchange(t, gas_state, Parameters, Next_Conditions["time_history"], Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    d_resp_vent = resp_control_vent(t, resp_contr_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, num_removed, i)
    d_resp_mech = respiratory_mechanics(t, resp_mech_state, Parameters, Initial_Conditions_dict, num_removed, i)

    d_combined = np.concatenate((d_gas, d_resp_vent, d_resp_mech))
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
            last_nonzero_value1 = Next_Conditions["time_history"][i - 1]
            last_nonzero_value2 = Next_Conditions["time_history"][i - 2]
            if t > 0.00001:
                if last_nonzero_value1 < last_nonzero_value2:
                    print("ISSUE")
            diff = np.abs(last_nonzero_value1 - target_values)
            if np.any(diff < 0.001):
                print(last_nonzero_value1)

    return d_combined

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PCSFCO2", "MRTO2", "MRTCO2", "CTO2", "CvtCO2", "CBO2", "CvbCO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# resp control ventilation
required_resp_control_keys = ["VE_integral"]
IC_resp_contr = np.array([Initial_Conditions[key] for key in required_resp_control_keys], dtype=float)
num_resp_control = len(required_resp_control_keys)

# resp mechanics
required_resp_mech_keys = ["Vflow_ua"]
IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
num_resp_mech = len(required_resp_mech_keys)

# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))
IC_overall = np.concatenate((IC_gas, IC_resp_contr, IC_resp_mech))

t_eval = np.linspace(0, t_span[1], t_span[1]*1000)
def simulate():
    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_overall, t_eval=t_eval, max_step = 0.003, method="RK23", rtol=1e-3,
                             atol=1e-6, args=(Parameters, Next_Conditions, num_gas, num_resp_control, num_resp_mech))

    return ODE_solution


if __name__ == "__main__":

    # lp = LineProfiler()
    # lp.add_function(BreathOptimiser.objective)

    # lp.add_function(combined_system)
    # lp.add_function(cardiovascular_controller)
    # lp.add_function(cardiovascular_system)
    # lp.add_function(gas_exchange)
    # lp.add_function(respiratory_mechanics)
    # lp.enable()
    solution = simulate()
    # lp.disable()
    # lp.print_stats()

    time = solution.t
    state_variables = solution.y

    state_variable_names = (
            required_gas_keys +
            required_resp_mech_keys +
            required_resp_control_keys
    )

    index = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 1

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_O2"][:index], label="Pa_O2", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_O2"][:index], label="PA_O2", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pa_CO2"][:index], label="Pa_CO2", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["PA_CO2"][:index], label="PA_CO2", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pb_CO2"][:index], label="Pb_CO2", color="c")

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V", color="k")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # ax2 = ax1.twinx()

    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V", color="g")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_O2"][:index], label="Cv_O2", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_O2"][:index], label="Ca_O2", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Ca_CO2"][:index], label="Ca_CO2", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Cv_CO2"][:index], label="Cv_CO2", color="k")
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["cCO2_diff"][:index], label="cCO2_diff")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["cO2_diff"][:index], label="cO2_diff")
    ax1.axhline(y=0.33/60, color="r", label="MRO2")
    ax1.axhline(y=-0.3/60, color="k", label="MRCO2")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()

    # fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dP_musc_dt"][:index], label="dP_dt")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["V"][:index], label="V")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_CO2"][:index], label="Pd_5_CO2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Pd_5_O2"][:index], label="Pd_5_O2")
    # # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["dV_dt"][:index], label="dV_dt")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # plt.show()

    variables_to_plot = [
        # "Vflow_ua", "P_ua", "P_musc", "dV_dt", "V",
        # "Pd_5_O2"
        "Pd_5_O2", "V", "PA_O2", "P_musc", "dP_musc_dt", "VAflow", "dV_dt", "P_musc", "VT"# , "VT", "VE_flow", "VAflow", "Q_pp", "V", "PA_O2_old", "PA_CO2_old","Cv_CO2", "Ca_CO2", "Cv_O2",
        # "Ca_O2", "dPA_CO2_dt", "dPA_O2_dt",
        # "dCvO2_dt", "dCvCO2_dt", "PA_CO2", "QT", "PA_O2",  # "V", "Cv_O2", "Ca_O2"
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

    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot all state variables
    plt.figure(figsize=(14, 10))

    for i, label in enumerate(required_gas_keys):
        if label not in ["CTO2", "CvtCO2", "CBO2", "CvbCO2"]:  # Skip "VT_sv"
            continue
        color = colors[
            i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
        plt.plot(time, state_variables[i], label=label,
                 color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()


 #    required_resp_mech_keys = ["Vflow_ua"]
 #
 #    # Number of state variables
 #    num_variables = state_variables.shape[0]
 #
 #
 #    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors
 #
 #    # Plot all state variables
 #    plt.figure(figsize=(14, 10))
 #
 #    for i, label in enumerate(required_resp_mech_keys):
 #        if label == "V":  # Skip "VT_sv"
 #            continue
 #        color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables
 # # Cycle through markers
 #        plt.plot(time, state_variables[len(required_gas_keys) + i], label=label,
 #                 color=color, linestyle='-')
 #
 #    plt.xlabel("Time")
 #    plt.ylabel("State Variables")
 #    plt.title("Evolution of State Variables Over Time")
 #    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
 #    plt.grid()
 #    plt.tight_layout()
 #    plt.show()