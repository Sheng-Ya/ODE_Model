import numpy as np
from scipy.integrate import solve_ivp
from Cardiovascular_system_new import cardiovascular_system
from Initial_Conditions import Initial_Conditions
from Next_Conditions import Next_Conditions
from Parameters import Parameters


# E_max_rv, Vu_rv, P0_rv, KE_rv
# First iteration
# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Parameters, time_history, Initial_Conditions_dict, all_time, E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv):
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
    # if t != 0:
    #     target_values = np.arange(0, 200, 10)
    #     if np.any(np.isclose(time_history[-1], target_values, atol=0.001)):
    #         print(time_history[-1])


    # Extract each subsystem's state variables
    cardio_state = Initial_Conditions_numpy

    Next_Conditions["Emax_lv"] = [E_max_lv]
    Parameters["Vu_lv"] = Vu_lv
    Parameters["P0_lv"] = P0_lv
    Parameters["KE_lv"] = KE_lv
    Next_Conditions["Emax_rv"] = [E_max_rv]
    Parameters["Vu_rv"] = Vu_rv
    Parameters["P0_rv"] = P0_rv
    Parameters["KE_rv"] = KE_rv


    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    d_cardio = cardiovascular_system(t, cardio_state, Parameters, Initial_Conditions_dict, Initial_Conditions_dict, Initial_Conditions_dict, all_time, num_removed)


    time_history.append(t)
    all_time.append(t)

    return d_cardio


def simulate(E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv):


    t_span = (0, 6) # Simulate for x seconds

    # cardiovascular system
    required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                               "VT_hv", "VT_rmv", "VT_amv", "VT_ev", "P_sp", "V_sa", "P_sa", "Q_sa", "VT_vc", "beta"]
    IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)

    # Solve ODE
    ODE_solution = solve_ivp(combined_system, t_span, IC_cardio, max_step = 0.005, method='RK23', rtol=1e-3,
                             atol=1e-6, args=(Parameters, Next_Conditions["time_history"], Next_Conditions, Next_Conditions["all_time"], E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv))

    return ODE_solution


if __name__ == "__main__":
    # Initial parameter values
    E_max_lv = 2.3  # Initial left ventricle contractility (mmHg/mL)
    Vu_lv = 15  # Initial unstressed volume of LV (mL)
    P0_lv = 1.55
    KE_lv = 0.014
    target_pressure_lv = 120  # Target peak LV pressure (mmHg)
    E_max_rv = 1.5
    Vu_rv = 38
    P0_rv = 1.5
    KE_rv = 0.014
    tolerance = 5  # Allowable deviation from target


    std_dev = {
        "E_max_lv": 0.2,
        "Vu_lv": 2,
        "P0_lv": 0.1,
        "KE_lv": 0.005,
        "E_max_rv": 0.2,
        "Vu_rv": 2,
        "P0_rv": 0.1,
        "KE_rv": 0.005,
    }



    max_iterations = 1000
    prev_error = float('inf')
    best_error = float('inf')
    best_params = (E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv)


    def truncated_normal(mean, std_dev, lower_bound=0):
        """Sample from a normal distribution but ensure values stay above lower_bound."""
        value = np.random.normal(mean, std_dev)
        while value < lower_bound:  # Resample if below bound
            value = np.random.normal(mean, std_dev)
        return value

    for i in range(max_iterations):
        E_max_lv = truncated_normal(best_params[0], std_dev["E_max_lv"], lower_bound=0)
        Vu_lv = truncated_normal(best_params[1], std_dev["Vu_lv"], lower_bound=0)
        P0_lv = truncated_normal(best_params[2], std_dev["P0_lv"], lower_bound=0)
        KE_lv = truncated_normal(best_params[3], std_dev["KE_lv"], lower_bound=0)
        E_max_rv = truncated_normal(best_params[4], std_dev["KE_rv"], lower_bound=0)
        Vu_rv = truncated_normal(best_params[5], std_dev["Vu_rv"], lower_bound=0)
        P0_rv = truncated_normal(best_params[6], std_dev["P0_rv"], lower_bound=0)
        KE_rv = truncated_normal(best_params[7], std_dev["KE_rv"], lower_bound=0)


        solution = simulate(E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv)

        index = Next_Conditions["VT_lv"].index(min(Next_Conditions["VT_lv"]))
        A = Next_Conditions["VT_lv"]
        B = Next_Conditions["P_lv"][index]
        error = abs(target_pressure_lv - Next_Conditions["P_lv"][index])
        # condition2 = np.min(Next_Conditions["P_ra"][3000:])

        if error <= tolerance:
            print("\nPass")
            print(E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv)
            # if condition2 > -2:
            break

        print(i)

        if error < best_error:
            best_error = error
            best_params = (E_max_lv, Vu_lv, P0_lv, KE_lv, E_max_rv, Vu_rv, P0_rv, KE_rv)

    print("\nFail")





















