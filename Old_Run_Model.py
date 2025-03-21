import numpy as np
import bisect
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.animation as animation
from PIL import Image, ImageDraw
from line_profiler import LineProfiler
import cProfile

from Old_Cardiovascular_controller import cardiovascular_controller
from Old_Cardiovascular_system import cardiovascular_system
from Old_Gas_Exchange import gas_exchange
from Initial_Conditions import Initial_Conditions
from Old_Next_Conditions import Next_Conditions
from Parameters import Parameters
from Resp_Control_Breath_Optimiser import breath_optimiser
from Resp_Control_Ventilation import resp_control_vent
from Old_Respiratory_Mechanics import respiratory_mechanics


# Resp control breath optimiser
t = 0.1
initial_Nd_guess = np.array([0, 0, 0, 0.4, 1, 2])  # Example initial values for a0, a1, a2, tau, t1, t2

# Next_Conditions["t_eval1"] = t_eval
# Next_Conditions["t_eval2"] = t_eval
# Next_Conditions["t_eval3"] = t_eval
# Next_Conditions["t_eval4"] = t_eval
# Next_Conditions["t_eval5"] = t_eval
# Next_Conditions["t_eval6"] = t_eval

# bounds = [(0, None), (0, None), (0, None), (0, None), (0.1, None), (0.1, None)]
#
# # Optimize
# result = minimize(breath_optimiser, initial_Nd_guess, args=(t, Next_Conditions["time_history"], Parameters, Next_Conditions, Next_Conditions, Next_Conditions, Next_Conditions["all_time"]), method='SLSQP', bounds=bounds)
# print(result.x)
target_values = np.arange(0, 500, 10)

# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, time_history, Initial_Conditions_dict, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech, all_time):
    """

    """
    if t != 0:
        if t < all_time[-1]:
            index = bisect.bisect_left(time_history, t)
            num_removed = len(time_history[index:])
            time_history[:] = time_history[:index]
        else:
            num_removed = 0
    else:
        num_removed = 0

    # just for checking progress of code
    if t != 0:
        if t > 0.00001:
            if time_history[-1] < time_history[-2]:
                print("ISSUE")
        if np.any(np.isclose(time_history[-1], target_values, atol=0.001)):
            print(time_history[-1])

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
    d_cardio = cardiovascular_system(t, cardio_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)
    d_cardio_contr = cardiovascular_controller(t, cardio_contr_state, Parameters, time_history, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)
    d_gas = gas_exchange(t, gas_state, Parameters, time_history, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)
    d_resp_mech = respiratory_mechanics(t, resp_mech_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)
    d_resp_vent = resp_control_vent(t, resp_contr_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, num_removed)

    # d_combined = np.concatenate((d_cardio, d_cardio_contr))
    d_combined = np.concatenate((d_cardio, d_cardio_contr, d_gas, d_resp_mech, d_resp_vent))

    time_history.append(t)
    all_time.append(t)

    return d_combined


t_span = (0,50) # Simulate for x seconds

# t_eval = np.arange(t_span[0], t_span[1], 0.01) # set as the number of times calculated in solution.t

# gas exchange
required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                     "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                     "PvbCO2", "PCSFCO2", "MRTO2", "MRTCO2", "Cv_O2", "Cv_CO2", "MRV"]
IC_gas = np.array([Initial_Conditions[key] for key in required_gas_keys], dtype=float)
num_gas = len(required_gas_keys)

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc", "theta_ao", "dtheta_ao_dt"]
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
required_resp_mech_keys = ["V", "alpha"]
IC_resp_mech = np.array([Initial_Conditions[key] for key in required_resp_mech_keys], dtype=float)
num_resp_mech = len(required_resp_mech_keys)

# IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_mech))
IC_overall = np.concatenate((IC_cardio, IC_cardio_contr, IC_gas, IC_resp_mech, IC_resp_contr))



def simulate():

    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_overall, max_step = 0.005, method="RK23", rtol=1e-3,
                             atol=1e-6, args=(Parameters, Next_Conditions["time_history"], Next_Conditions, num_gas, num_cardio, num_cardio_control, num_resp_control, num_resp_mech, Next_Conditions["all_time"]))

    return ODE_solution


if __name__ == "__main__":

    # lp = LineProfiler()
    # lp.add_function(combined_system)
    # lp.add_function(cardiovascular_controller)
    # lp.add_function(cardiovascular_system)
    # lp.add_function(gas_exchange)
    # lp.add_function(respiratory_mechanics)
    # lp.enable()
    solution = simulate()
    # # # # cProfile.run('simulate()', sort='time')
    # lp.disable()
    # lp.print_stats()

    time = solution.t
    state_variables = solution.y

    print(len(Next_Conditions["time_history"]))

    required_cardio_keys = ["VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                            "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc", "theta_ao", "dtheta_ao_dt"]
    required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv",
                                    "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", "R_ep_change",
                                    "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                                    "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                                    "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2",
                                    "xam_O2", "xM", "x_met"]
    required_gas_keys = ["Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2", "Pd_4_O2", "Pd_4_CO2",
                         "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2", "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2",
                         "PvbCO2", "PCSFCO2", "MRTO2", "MRTCO2", "Cv_O2", "Cv_CO2", "MRV"]
    required_resp_mech_keys = ["V", "alpha"]
    required_resp_control_keys = ["VE_integral"]

    state_variable_names = (
            required_cardio_keys +
            required_cardio_control_keys +
            required_gas_keys +
            required_resp_mech_keys +
            required_resp_control_keys
    )

    # Ts_change_index = state_variable_names.index("Ts_change")
    # Ts_change_values = solution.y[Ts_change_index, :]
    # Tv_change_index = state_variable_names.index("Tv_change")
    # Tv_change_values = solution.y[Tv_change_index, :]
    #
    # plt.plot(time, Ts_change_values, label="Ts_change")
    # plt.plot(time, Tv_change_values, label="Tv_change")
    # plt.legend()
    # plt.show()

    # variables_to_plot = [
    #     # "f_sp_history", "f_sh_history", "f_v_history",
    #     # "xb_CO2", "P_sp", "P_bv", "Q_bp", "beta","U2", "T", "xb_O2", "Cvb_O2"
    #     "Q_ev", "Q_ep", "dCvO2_dt", "dCvCO2_dt", "cCO2_diff", "cO2_diff", "PA_CO2", "QT", "PA_O2", "Cv_CO2", "Ca_CO2", "Cv_O2", "Ca_O2", "Q_bp" #"V", "Cv_O2", "Ca_O2"
    #     # "Nt", "VE_integral"
    #     # "AR_ao", "theta_ao", "HR", "d2theta_ao_dt2" # , "T", "xb_O2", "f_sp_history", "f_sh_history", "f_v_history", "phi_met_history", "f_sv_history",
    #     # "phi", "phi_atr" #"V", "Cv_O2", "Ca_O2"
    #     # "Vu_ev", "Vu_amv", "Vu_rmv", "Vu_sv", "R_ep", "R_amp", "R_rmp", "R_sp",
    #     # "R_bp", "R_hp", "Emax_lv", "Emax_rv", "I", "phi_met", "Nt",
    #     # "Vu_sv_change", "prev_flat_bit", "Pa_O2", "HR"
    # ]
    #
    # for key in variables_to_plot:
    #     if key in Next_Conditions:  # Check if the key exists in updates
    #         plt.figure(figsize=(8, 4))  # Create a new figure for each variable
    #         plt.plot(Next_Conditions[key], label=key, linewidth=2)
    #         plt.xlabel("Time Steps")
    #         plt.ylabel(key)
    #         plt.title(f"Plot of {key} over Time")
    #         plt.legend()
    #         plt.grid(True)
    #         plt.show()


    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Cv_O2"][1:], label="Cv_O2", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Ca_O2"][1:], label="Ca_O2", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Ca_CO2"][1:], label="Ca_CO2", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Cv_CO2"][1:], label="Cv_CO2", color="k")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["FO2"], label="FO2", color="m")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["FCO2"], label="FCO2", color="c")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["QT"], label="QT", color="k")
    #
    #
    #
    #
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["Q_ev"], label="Q_ev", color="g")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["Q_ep"], label="Q_ep", color="k")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["VT_ev"][1:], label="VT_ev", color="b")
    #
    #
    #
    # ax1.set_xlabel("Time (s)")
    # # ax1.set_ylabel("Pressure (mmHg)", color="k")
    # # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)

    # # Create second y-axis for volume
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"][1:], label="VT_lv", color="c")
    # ax2.plot(Next_Conditions["time_history"][92500:], Next_Conditions["Q_sa"][92501:], label="Q_sa",
    #          linestyle="dashed", color="c")
    # ax2.set_ylabel("Flow (mL/s)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    plt.show()


    plt.plot(Next_Conditions["time_history"], Next_Conditions["f_sp_history"], label="f_sp_history")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["f_sh_history"], label="f_sh_history")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["f_v_history"], label="f_v_history")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["f_sv_history"], label="f_sv_history")
    plt.xlabel("Time (s)")
    plt.ylabel("f")
    plt.legend()
    plt.grid(True)
    plt.show()


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["xb_O2"][1:], label="xb_O2")
    # # plt.plot(Next_Conditions["U2"], label="U2")
    # plt.xlabel("Time (s)")
    # plt.ylabel("HR")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.plot(Next_Conditions["time_history"])
    # plt.show()

    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pa_O2"][1:], label="Pa_O2")
    # plt.plot(Next_Conditions["Q_pp"])
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Nt"], label="Nt")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pa_CO2"][1:], label="PaCO2")


    # d_beta_dt = np.gradient(Next_Conditions["beta"][1:], Next_Conditions["time_history"])
    # plt.plot(a, Next_Conditions["HR"], label="HR")

    # plt.plot(Next_Conditions["time_history"], d_beta_dt, label="d(beta)/dt")
    # Add labels and legend

    # plt.xlabel("Time (s)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_ra"], label="Pmax_ra")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_lv"], label="Pmax_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"][1:], label="VT_lv (Left Ventricle)")
    # # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_la"], label="P_la (Left Atrium)")
    # # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["phi"], label="phi")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_sa"][1:], label="P_sa")


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_la"], label="Pmax_la", alpha = 0.2)
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv (Right Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ev"], label="P_ev")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_sa"][1:], label="P_sa")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_im"], label="P_im")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ev"], label="P_ev")

    # Add labels and legend
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.title("Pressure Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["VT_ra"][1:], Next_Conditions["P_ra"], label="RA")  # 10 s all
    # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["VT_la"][1:], Next_Conditions["P_la"], label="LA")  # 10 s all
    # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()



    plt.plot(Next_Conditions["VT_lv"][1:], Next_Conditions["P_lv"], label="P_lv (Left Ventricle)")  # 10 s all
    # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()
    # #
    fig, ax1 = plt.subplots()
    #
    # # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_rv"][99500:], label="P_rv", color="b")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_ra"][99500:], label="P_ra", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_la"], label="Pmax_la", color="g")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["VT_rv"][1:], label="VT_rv", color="r")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"][1:], label="VT_lv", color="m")

    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["VT_la"][1:], label="VT_la", color="c")
    #
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv", color="g")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Pmax_rv"][99500:], label="Pmax_rv", color="c")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa", color="g")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pv"], label="P_pv", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp", color="b")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc", color="c")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_pa"][99500:], label="P_pa", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", linestyle="dashed", color="c")
    #
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["phi_atr1"], label="phi_atr1", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["phi_atr2"], label="phi_atr2", color="k")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["phi"], label="phi", color="r")


    # ax1.plot(Next_Conditions["time_history"][0:11400], Next_Conditions["VT_ra"][0:11400], label="VT_ra", color="m")
    # ax1.plot(Next_Conditions["time_history"][0:11400], Next_Conditions["P_ra"][0:11400], label="P_ra", color="m")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["VT_la"][1:], label="VT_la", color="m")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_la"], label="P_la", color="b")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv", color="c")

    # ax1.plot(Next_Conditions["time_history"][0:11400], Next_Conditions["phi"][0:11400], label="phi", color="k")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["phi_cond"], label="phi_cond", color="m")

    ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_title("R_la and R_ra = 0.025 mmHg.s/ml")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # #
    # # Create second y-axis for volume
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_rv"][99500:], label="Q_rv", linestyle="dashed",
    #          color="b")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Qi_rv"][99500:], label="Qi_rv",
    #          linestyle="dashed", color="c")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_ra"][99500:], label="Q_ra", linestyle="dashed",
    #          color="g")

    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pa"], label="V_pa", color="g")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pv"], label="V_pv", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pp"], label="V_pp", color="b")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv", color="g")

    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="c")

    # ax2.set_ylabel("Flow (mL/s)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    # #
    plt.show()






    # fig, ax1 = plt.subplots()
    #
    # # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_lv"][99500:], label="P_lv", color="b")
    # # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_la"][99500:], label="P_la", color="g")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Pmax_la"][99500:], label="Pmax_la", color="k")
    # # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["dPmax_la_dt"][99500:], label="dPmax_la_dt", color="b")

    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Pmax_lv"][99500:], label="Pmax_lv", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pv"], label="P_pv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc", color="c")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_sa"][99501:], label="P_sa", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", linestyle="dashed", color="c")

    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["phi_atr"][99500:], label="phi_atr", color="m")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["phi"][99500:], label="phi", color="k")

    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    # ax1.tick_params(axis='y', labelcolor="k")
    # # ax1.set_title("R_la and R_ra = 0.025 mmHg.s/ml")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # #
    # # Create second y-axis for volume
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_lv"][99500:], label="Q_lv", linestyle="dashed", color="b")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Qi_lv"][99500:], label="Qi_lv", linestyle="dashed", color="c")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_la"][99501:], label="Q_la", linestyle="dashed", color="g")

    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pa"], label="V_pa", color="g")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pv"], label="V_pv", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pp"], label="V_pp", color="b")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv", color="g")

    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="c")

    # ax2.set_ylabel("Flow (mL/s)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    # #
    # plt.show()
    #
    # fig, ax1 = plt.subplots()

    # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_lv"][99500:], label="P_lv", color="b")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_la"][99500:], label="P_la", color="g")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Pmax_la"][99500:], label="Pmax_la", color="k")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["dPmax_la_dt"][99500:], label="dPmax_la_dt",
    #          color="k")

    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv", color="g")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Pmax_lv"][99500:], label="Pmax_lv", color="c")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa", color="g")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pv"], label="P_pv", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp", color="b")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc", color="c")
    # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["P_sa"][99501:], label="P_sa", color="r")
    # # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", linestyle="dashed", color="c")
    #
    # # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["phi_atr"][99500:], label="phi_atr", color="m")
    # # ax1.plot(Next_Conditions["time_history"][99500:], Next_Conditions["phi"][99500:], label="phi", color="k")
    #
    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    # ax1.tick_params(axis='y', labelcolor="k")
    # # ax1.set_title("R_la and R_ra = 0.025 mmHg.s/ml")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)
    # #
    # # Create second y-axis for volume
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_lv"][99500:], label="Q_lv", linestyle="dashed",
    #          color="b")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Qi_lv"][99500:], label="Qi_lv",
    #          linestyle="dashed", color="c")
    # ax2.plot(Next_Conditions["time_history"][99500:], Next_Conditions["Q_la"][99501:], label="Q_la", linestyle="dashed",
    #          color="g")
    #
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pa"], label="V_pa", color="g")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pv"], label="V_pv", color="r")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pp"], label="V_pp", color="b")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra", color="r")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv", color="g")
    #
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="c")
    #
    # ax2.set_ylabel("Volume (mL)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    # #
    # plt.show()

    plt.plot(Next_Conditions["time_history"], Next_Conditions["Vu_ev"][1:], label="Vu_ev")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Vu_amv"][1:], label="Vu_amv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Vu_rmv"][1:], label="Vu_rmv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Vu_sv"][1:], label="Vu_sv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_ep"][1:], label="R_ep")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_amp"][1:], label="R_amp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_rmp"][1:], label="R_rmp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_sp"][1:], label="R_sp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_bp"][1:], label="R_bp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["R_hp"][1:], label="R_hp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["HR"][1:], label="HR")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Emax_lv"][1:], label="Emax_lv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Emax_rv"][1:], label="Emax_rv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["I"][1:], label="I")
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




    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_lv"], label="Pmax_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_la"], label="P_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv (Right Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ev"], label="P_ev")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_sa"][1:], label="P_sa")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_im"], label="P_im")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ev"], label="P_ev")

    # # Add labels and legend
    # plt.ylabel("Pressure (mmHg)")
    # plt.xlabel("Time (s)")
    # plt.title("Pressure Traces")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.plot(Next_Conditions["time_history"], Next_Conditions["phi_atr"], label="phi_atr")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["phi"], label="phit")
    # # plt.plot(Next_Conditions["time_history"], Next_Conditions["Emax_lv"][1:], label="Emax_lv")
    # # plt.plot(Next_Conditions["index1"], label="index1")
    # # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa")
    # # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_abd"], label="P_abd")
    # # Add labels and legend
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"][1:], label="VT_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_la"][1:], label="VT_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_ra"][1:], label="VT_ra (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_rv"][1:], label="VT_rv (Right Ventricle)")

    # a = Next_Conditions["time_history"]
    # b = Next_Conditions["V_lv"]
    # # # for 20 s, go from [8750:]
    # plt.plot(Next_Conditions["time_history"][:900], Next_Conditions["VT_lv"][:900], label="VT_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv")
    # plt.plot(Next_Conditions["time_history"][:900], Next_Conditions["VT_la"][:900], label="VT_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["phi"], label="phi")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_rv"][1:], label="VT_rv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_ra"][1:], label="VT_ra (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra (Right Atrium)")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_sv"], label="V_sv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_rmv"], label="V_rmv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_amv"][1:], label="VT_amv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_amv"], label="V_amv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_vc"], label="V_vc", linestyle="dashed", color="c")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_hv"], label="V_hv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_ev"][1:], label="VT_ev")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Volume (mL)")
    plt.title("Volume Traces")
    plt.legend()
    plt.grid(True)
    plt.show()



    # plt.plot(Next_Conditions["V_lv"][80000:], Next_Conditions["Pmax_lv"][80000:], label="P_lv (Left Ventricle)")
    # get max's plot with Pmax_la instead of P_la
    plt.plot(Next_Conditions["VT_ra"][97501:], Next_Conditions["P_ra"][97500:], label="RA")
    # plt.plot(Next_Conditions["VT_ra"][1:], Next_Conditions["P_ra"], label="LA")

    # # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    # plt.title("Pressure-Volume Traces")
    plt.legend()
    # plt.grid(True)
    plt.show()

    # # a = Next_Conditions["Q_la"]
    #
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_pp"][1:], label="Q_pp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_ep"], label="Q_ep")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_sp"], label="Q_sp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_bp"][1:], label="Q_bp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_hp"][1:], label="Q_hp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_rmp"][1:], label="Q_rmp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_amp"][1:], label="Q_amp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_amv"], label="Q_amv")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Q_la"][7501:], label="Q_la (into LA)")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Q_ra"][7500:], label="Q_ra (into RA)")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Q_rv"][7500:], label="Q_rv (leaving RV/into pul art)")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Q_lv"][7500:], label="Q_lv (leaving LV)")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Qi_lv"][7500:], label="Qi_lv")
    plt.plot(Next_Conditions["time_history"][7500:], Next_Conditions["Qi_rv"][7500:], label="Qi_rv")


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_la"][1:], label="Q_la")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_vc"], label="Q_vc")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_lv"], label="Q_lv")


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_jp"], label="Q_jp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_vc"], label="Q_vc")
    #
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_sa"][1:], label="Q_sa")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces")
    plt.legend()
    plt.grid(True)
    plt.show()



    required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                               "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "P_sa", "Q_sa", "VT_vc", "theta_ao", "dtheta_ao_dt"]

    # Number of state variables
    num_variables = state_variables.shape[0]


    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot all state variables
    plt.figure()

    for i, label in enumerate(required_cardio_keys):
        if label == "VT_sv":  # Skip "VT_sv"
            continue
        color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables
        plt.plot(time, state_variables[i], label=label, color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()




    required_cardio_control_keys = ["theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv",
                                    "theta_change_CO2_sv",
                                    "theta_change_O2_sh", "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap",
                                    "R_ep_change",
                                    "R_sp_change", "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change",
                                    "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change", "Emax_rv_change", "Ts_change",
                                    "Tv_change", "xb_O2", "xb_CO2", "xh_O2", "xh_CO2", "Wh", "xrm_O2", "xrm_CO2",
                                    "xam_O2",
                                    "xM", "x_met"]

    # Number of state variables
    num_variables = state_variables.shape[0]

    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot cardio control variables
    plt.figure()
    for i, label in enumerate(required_cardio_control_keys):
        # if label != "beta":
        if label == "Wh" or label == "P_tilda":  # Skip "Wh"
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





    required_gas_keys = [
        "Pd_1_O2", "Pd_1_CO2", "Pd_2_O2", "Pd_2_CO2", "Pd_3_O2", "Pd_3_CO2",
        "Pd_4_O2", "Pd_4_CO2", "Pd_5_O2", "Pd_5_CO2", "Pa_O2", "Pa_CO2",
        "dPa_O2_dt", "dPa_CO2_dt", "PA_O2", "PA_CO2", "PvbCO2", "PCSFCO2",
        "MRTO2", "MRTCO2", "CvO2", "CvCO2", "MRV"
    ]

    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors

    # Plot all state variables
    plt.figure(figsize=(14, 10))

    for i, label in enumerate(required_gas_keys):
        # if label == "Pd_2_O2":  # Skip "VT_sv"
        #     continue
        color = colors[i % len(colors)]  # Cycle through colors if there are more than 20 variables # Cycle through markers
        plt.plot(time, state_variables[len(required_cardio_keys + required_cardio_control_keys) + i], label=label, color=color, linestyle='-', markersize=4)

    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.title("Evolution of State Variables Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
    plt.grid()
    plt.tight_layout()
    plt.show()




 #    required_resp_mech_keys = ["V", "alpha"]
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
 #        plt.plot(time, state_variables[len(required_cardio_keys + required_cardio_control_keys + required_gas_keys) + i], label=label,
 #                 color=color, linestyle='-')
 #
 #    plt.xlabel("Time")
 #    plt.ylabel("State Variables")
 #    plt.title("Evolution of State Variables Over Time")
 #    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place the legend outside the plot
 #    plt.grid()
 #    plt.tight_layout()
 #    plt.show()





    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Cv_O2"][1:], label="CvO2")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Cv_CO2"][1:], label="CvCO2")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Ca_O2"][1:], label="CaO2")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Ca_CO2"][1:], label="CaCO2")
    #
    # # Add labels and legend
    # plt.xlabel("Time (s)")
    # plt.ylabel("Gas Concentration (mmol/L)")
    # plt.title("Gas Concentrations")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    #




    fig, ax1 = plt.subplots()

    # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_lv"], label="Pmax_lv", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", color="g")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_im"], label="P_im", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc", color="g")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_la"], label="P_la", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Qi_rv"], label="Qi_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Qi_lv"], label="Qi_lv", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_sa"][1:], label="P_sa", color="g")


    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)", color="k")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.set_title("Pressure Trace")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # Create second y-axis for volume
    ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"], label="VT_lv", linestyle="dashed", color="c")
    ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_la"][1:], label="VT_la", linestyle="dashed", color="g")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_rv"], label="VT_rv", linestyle="dashed", color="r")
    ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_ra"][1:], label="VT_ra", linestyle="dashed", color="b")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_amv"], label="V_amv", linestyle="dashed", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["VT_amv"][1:], label="VT_amv", linestyle="dashed", color="g")
    #
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="b")

    ax2.set_ylabel("Volume (mL)", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()

    fig, ax1 = plt.subplots()

    # Plot pressures on primary y-axis
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_amv"], label="P_amv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_im"], label="P_im", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Qi_rv"], label="Qi_rv", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Qi_lv"], label="Qi_lv", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", color="c")

    # ax1.set_xlabel("Time (s)")
    # ax1.set_ylabel("Pressure (mmHg)", color="k")
    # ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_title("Pressure-Volume Trace")
    # ax1.legend(loc="upper left")
    # ax1.grid(True)

    # # Create second y-axis for volume
    # ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_lv"], label="V_lv", linestyle="dashed", color="c")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pa"], label="V_pa", linestyle="dashed", color="g")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pv"], label="V_pv", linestyle="dashed", color="r")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pp"], label="V_pp", linestyle="dashed", color="b")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra", linestyle="dashed", color="r")
    # # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv", linestyle="dashed", color="g")
    #
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="b")
    #
    # ax2.set_ylabel("Volume (mL)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    #
    # plt.show()








    # # Load schematic image
    # schematic_path = "C:/Users/vanes/Documents/Screenshot 2025-02-09 164337.png"
    # schematic = Image.open(schematic_path)
    #
    # # Sample volume data (replace with your real data)
    # time_steps = len(Next_Conditions["time_history"][5000:])  # Number of time steps
    # V_lv = Next_Conditions["V_lv"][5000:]
    # V_rv = Next_Conditions["V_rv"][5000:]  # Simulated RV volume
    #
    # # Define locations for volume filling (manually set based on schematic image)
    # lv_box = (200, 300, 250, 400)  # Example coordinates (left, top, right, bottom)
    # rv_box = (300, 300, 350, 400)  # Example coordinates
    #
    # frames = []
    #
    # for t in range(time_steps):
    #     frame = schematic.copy()
    #     draw = ImageDraw.Draw(frame)
    #
    #     # Scale volumes to pixel heights
    #     max_height = 100  # Maximum filling height in pixels
    #     lv_fill = int((V_lv[t] - min(V_lv)) / (max(V_lv) - min(V_lv)) * max_height)
    #     rv_fill = int((V_rv[t] - min(V_rv)) / (max(V_rv) - min(V_rv)) * max_height)
    #
    #     # Draw LV fill
    #     draw.rectangle([lv_box[0], lv_box[3] - lv_fill, lv_box[2], lv_box[3]], fill="red")
    #
    #     # Draw RV fill
    #     draw.rectangle([rv_box[0], rv_box[3] - rv_fill, rv_box[2], rv_box[3]], fill="blue")
    #
    #     frames.append(frame)
    #
    # # Save as GIF
    # gif_path = "C:/Users/vanes/Documents/heart_cycle.gif"
    # frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0)
    # print(f"GIF saved at: {gif_path}")