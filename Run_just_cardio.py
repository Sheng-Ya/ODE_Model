import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Cardiovascular_system import cardiovascular_system
from Initial_Conditions import Initial_Conditions
from Next_Conditions import Next_Conditions
from Parameters import Parameters


# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, time_history, Initial_Conditions_dict, all_time):
    """

    """
    # remove values that were in the rejected steps
    if t != 0:
        if t < all_time[-1]:
            num_removed = sum(1 for x in time_history if x > t)
            for _ in range(num_removed):
                time_history.pop()
        else:
            num_removed = 0
    else:
        num_removed = 0

    # just for checking progress of code
    if t != 0:
        target_values = np.arange(0, 200, 1)
        if np.any(np.isclose(time_history[-1], target_values, atol=0.001)):
            print(time_history[-1])


    # Extract each subsystem's state variables
    cardio_state = Initial_Conditions_numpy

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    d_cardio = cardiovascular_system(t, cardio_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)


    time_history.append(t)
    all_time.append(t)

    return d_cardio


def simulate():


    t_span = (0, 30) # Simulate for x seconds

    # cardiovascular system
    required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                               "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "V_sa", "P_sa", "Q_sa", "VT_vc", "beta"]
    IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)

    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_cardio, max_step = 0.01, method='RK23', rtol=1e-3,
                             atol=1e-6, args=(Parameters, Next_Conditions["time_history"], Next_Conditions, Next_Conditions["all_time"]))

    return ODE_solution


if __name__ == "__main__":


    solution = simulate()

    time = solution.t
    state_variables = solution.y

    fig, ax1 = plt.subplots()

    # Plot pressures on primary y-axis
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv", color="b")
    ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_rv"], label="Pmax_rv", linestyle="dashed", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa", linestyle="dashed", color="g")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pv"], label="P_pv", linestyle="dashed", color="r")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp", linestyle="dashed", color="b")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_sa"][1:], label="P_sa", color="c")
    # ax1.plot(Next_Conditions["time_history"], Next_Conditions["P_thor"], label="P_thor", linestyle="dashed", color="c")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)", color="k")
    ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_title("Pressure and Volume Traces")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    #
    # Create second y-axis for volume
    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_lv"], label="V_lv", linestyle="dashed", color="b")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pa"], label="V_pa", color="g")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pv"], label="V_pv", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_pp"], label="V_pp", color="b")
    ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra", color="r")
    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv", color="g")

    # ax2.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la", linestyle="dashed", color="c")

    # ax2.set_ylabel("Volume (mL)", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")
    # ax2.legend(loc="upper right")
    #
    plt.show()


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Pmax_lv"], label="Pmax_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_lv"], label="P_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_la"], label="P_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_ra"], label="P_ra (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["P_rv"], label="P_rv (Right Ventricle)")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pa"], label="P_pa")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pp"], label="P_pp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["P_pv"], label="P_pv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["P_vc"], label="P_vc")

    # Add labels and legend
    plt.ylabel("Pressure (mmHg)")
    plt.xlabel("Time (s)")
    plt.title("Pressure Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.plot(Next_Conditions["time_history"], Next_Conditions["U"][1:])
    # plt.grid(True)
    # plt.show()


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_lv"][1:], label="VT_lv (Left Ventricle)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_la"][1:], label="VT_la (Left Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_ra"][1:], label="VT_ra (Right Atrium)")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["VT_rv"][1:], label="VT_rv (Right Ventricle)")

    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_lv"], label="V_lv (Left Ventricle)")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_la"], label="V_la (Left Atrium)")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_ra"], label="V_ra (Right Atrium)")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["V_rv"], label="V_rv (Right Ventricle)")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Volume (mL)")
    plt.title("Volume Traces of Cardiac Chambers")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.plot(Next_Conditions["V_lv"][9120:], Next_Conditions["Pmax_lv"][9120:], label="V_lv (Left Ventricle)") # 10 s all
    # plt.plot(Next_Conditions["V_lv"][39750:], Next_Conditions["Pmax_lv"][39750:], label="V_lv (Left Ventricle)")
    plt.plot(Next_Conditions["V_lv"], Next_Conditions["Pmax_lv"], label="V_lv (Left Ventricle)")
    # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.title("Pressure-Volume Traces")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(Next_Conditions["V_la"], Next_Conditions["P_la"], label="Left Atrium")
    # Add labels and legend
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.title("Pressure-Volume Traces")
    plt.legend()
    plt.grid(True)
    plt.show()


    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_pp"][1:], label="Q_pp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_bp"][1:], label="Q_bp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_hp"][1:], label="Q_hp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_rmp"][1:], label="Q_rmp")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_amp"][1:], label="Q_amp")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_la"][1:], label="Q_la")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_ra"], label="Q_ra")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_rv"], label="Q_rv")
    plt.plot(Next_Conditions["time_history"], Next_Conditions["Q_lv"], label="Q_lv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Qi_lv"], label="Qi_lv")
    # plt.plot(Next_Conditions["time_history"], Next_Conditions["Qi_rv"], label="Qi_rv")

    # Add labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (mL/s)")
    plt.title("Flow Traces of Right Cardiac Chambers")
    plt.legend()
    plt.grid(True)
    plt.show()








