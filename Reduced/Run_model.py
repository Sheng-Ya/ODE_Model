import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from Derivatives import model_derivatives
from Parameters import Parameters 
from Initial_Conditions import Initial_Conditions
from Next_Conditions import Next_Conditions


target_values = np.arange(0, 10000, 10)

time_saved = 0.005
BUFFER_LIMIT = 20000

min_time = 10 # Minimum time in seconds before checking
max_time = 80 # Maximum time limit to avoid infinite loops
time_step = 200  # Chunk size per solve


# get the first derivative and outputs from all the separated systems
def combined_system(t, Initial_Conditions_numpy, Initial_Conditions_dict, num_cardio, num_cardio_control, Input_Parameters):

    i = Initial_Conditions_dict["i"].item()
    actual_index = i % BUFFER_LIMIT

    all_time = Initial_Conditions_dict["all_time"]

    # this loop is just for removing stored values from RK23 rejected steps
    if i > 1: 
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
                
        else:
            num_removed = 0
    else:
        num_removed = 0


    # Indices for slicing
    idx_cardio_contr = num_cardio + num_cardio_control

    # Extract each subsystem's state variables
    cardio_contr_state = Initial_Conditions_numpy[:idx_cardio_contr]

    # Cardiovascular dynamics (look at separate systems by just commenting out other states, and changing IC_overall, d_combined)
    derivatives_all = model_derivatives(t, cardio_contr_state, Initial_Conditions_dict, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters)


    all_time[(i - num_removed) % BUFFER_LIMIT] = t
    Initial_Conditions_dict["i"][0] = i - num_removed + 1
    Initial_Conditions_dict["j"][0] = Initial_Conditions_dict["j"].item() - num_removed + 1

    # Loop for debugging check for progress
    if t != 0:
        diff = np.abs(t - target_values)
        if np.any(diff < 0.0001):
            print(t)

    return derivatives_all

# cardiovascular system
required_cardio_keys = [ "VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
                         "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt"]
IC_cardio = np.array([Initial_Conditions[key] for key in required_cardio_keys], dtype=float)
num_cardio = len(required_cardio_keys)

# cardiovascular controller
required_cardio_control_keys = ["P_tilda", "R_ep_change", "R_sp_change", "R_rmp_n_change", "R_amp_n_change",
                                "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change",
                                "Emax_rv_change", "Ts_change", "Tv_change", "P_n_current"]
IC_cardio_contr = np.array([Initial_Conditions[key] for key in required_cardio_control_keys], dtype=float)
num_cardio_control = len(required_cardio_control_keys)

IC_overall = np.concatenate((IC_cardio, IC_cardio_contr))


def simulate():
    # Initial setup
    IC_current = IC_overall.copy()

    (C_pa, C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la, Emax_ra, P0_ra, KE_ra,
     C_sa, L_sa, R_sa, D1, K1_vc, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n, R_amv_n, C_ev,
     C_sv, C_bv, C_hv, C_rmv, C_amv, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Wb_sh, Wb_sp,
     Wb_sv, Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev,
     GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, f_ab_max, f_ab_min, k_ab, P_n, DT_v, GT_s, GT_v, T0, R_bpn, R_hpn,
     # added params
     Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po, Kf_po, Kb_po, Kv_po,
     theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, R_po, R_mi, R_tr, R_ao, Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv,
     Vu_vc, Vvc_max, Vu_pa, Vu_pp, Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep,
     tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_p, tau_z, tau_Ts,
     tau_Tv, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv, DV_sv, DT_s, DT_v, scale_param2,
     shift_param1, shift_param2, shift_param3, shift_param4, rise_time_atr, fall_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P) = (
    Parameters[k] for k in
    ["C_pa", "C_pp", "C_pv", "L_pa", "R_pa", "R_pp", "R_pv", "KE_lv", "KE_rv", "P0_lv", "P0_rv", "Emax_la", "P0_la",
     "KE_la", "Emax_ra", "P0_ra", "KE_ra", "C_sa", "L_sa", "R_sa", "D1", "K1_vc", "Kr_vc", "Rvc_n", "C_jp", "R_ev_n",
     "R_sv_n", "R_bv_n", "R_hv_n", "R_rmv_n", "R_amv_n", "C_ev", "C_sv", "C_bv", "C_hv", "C_rmv", "C_amv", "fab_o",
     "fes_o", "fes_inf", "fes_max", "fev_o", "fev_inf", "kes", "kev", "Wb_sh", "Wb_sp", "Wb_sv", "Emax_lv0", "Emax_rv0",
     "fes_min", "GEmax_lv", "GEmax_rv", "GR_amp", "GR_ep", "GR_rmp", "GR_sp", "GV_amv", "GV_ev", "GV_rmv", "GV_sv",
     "R_amp0", "R_ep0", "R_rmp0", "R_sp0", "f_ab_max", "f_ab_min", "k_ab", "P_n", "DT_v", "GT_s", "GT_v", "T0", "R_bpn", "R_hpn",
     # added params
     "Kp_ao", "Kf_ao", "Kb_ao", "Kv_ao", "theta_ao_max", "Kp_mi", "Kf_mi", "Kb_mi", "Kv_mi", "theta_mi_max", "Kp_po",
     "Kf_po", "Kb_po", "Kv_po", "theta_po_max", "Kp_tr", "Kf_tr", "Kb_tr", "Kv_tr", "theta_tr_max", "R_po", "R_mi",
     "R_tr", "R_ao", "Vu_sa", "V_tot", "Vu_jp", "Vu_bv", "Vu_hv", "Vu_vc", "Vvc_max", "Vu_pa", "Vu_pp", "Vu_pv",
     "Vu_la", "Vu_lv", "Vu_ra", "Vu_rv", "tau_Emax_lv", "tau_Emax_rv", "tau_Ramp", "tau_Rep", "tau_Rrmp", "tau_Rsp",
     "tau_Vamv", "tau_Vev", "tau_Vrmv", "tau_Vsv", "Vu_amv0", "Vu_ev0", "Vu_rmv0", "Vu_sv0", "tau_p", "tau_z", "tau_Ts",
     "tau_Tv", "DEmax_lv", "DEmax_rv", "DR_amp", "DR_ep", "DR_rmp", "DR_sp", "DV_amv", "DV_ev", "DV_rmv", "DV_sv",
     "DT_s", "DT_v", "scale_param2", "shift_param1", "shift_param2", "shift_param3", "shift_param4", "rise_time_atr",
     "fall_time_atr", "rise_time_ven", "fall_time_ven", "ahead1", "theta_min", "delta_P"])

    Input_Parameters = [C_pa, C_pp, C_pv, L_pa, R_pa, R_pp, R_pv, KE_lv, KE_rv, P0_lv, P0_rv, Emax_la, P0_la, KE_la,
     Emax_ra, P0_ra, KE_ra, C_sa, L_sa, R_sa, D1, K1_vc, Kr_vc, Rvc_n, C_jp, R_ev_n, R_sv_n, R_bv_n, R_hv_n, R_rmv_n,
     R_amv_n, C_ev, C_sv, C_bv, C_hv, C_rmv, C_amv, fab_o, fes_o, fes_inf, fes_max, fev_o, fev_inf, kes, kev, Wb_sh,
     Wb_sp, Wb_sv, Emax_lv0, Emax_rv0, fes_min, GEmax_lv, GEmax_rv, GR_amp, GR_ep, GR_rmp, GR_sp, GV_amv, GV_ev,
     GV_rmv, GV_sv, R_amp0, R_ep0, R_rmp0, R_sp0, f_ab_max, f_ab_min, k_ab, P_n, DT_v, GT_s, GT_v, T0, R_bpn, R_hpn,
     # added params
     Kp_ao, Kf_ao, Kb_ao, Kv_ao, theta_ao_max, Kp_mi, Kf_mi, Kb_mi, Kv_mi, theta_mi_max, Kp_po, Kf_po, Kb_po, Kv_po,
     theta_po_max, Kp_tr, Kf_tr, Kb_tr, Kv_tr, theta_tr_max, R_po, R_mi, R_tr, R_ao, Vu_sa, V_tot, Vu_jp, Vu_bv, Vu_hv,
     Vu_vc, Vvc_max, Vu_pa, Vu_pp, Vu_pv, Vu_la, Vu_lv, Vu_ra, Vu_rv, tau_Emax_lv, tau_Emax_rv, tau_Ramp, tau_Rep,
     tau_Rrmp, tau_Rsp, tau_Vamv, tau_Vev, tau_Vrmv, tau_Vsv, Vu_amv0, Vu_ev0, Vu_rmv0, Vu_sv0, tau_p, tau_z, tau_Ts,
     tau_Tv, DEmax_lv, DEmax_rv, DR_amp, DR_ep, DR_rmp, DR_sp, DV_amv, DV_ev, DV_rmv, DV_sv, DT_s, DT_v, scale_param2,
     shift_param1, shift_param2, shift_param3, shift_param4, rise_time_atr, fall_time_atr, rise_time_ven,
     fall_time_ven, ahead1, theta_min, delta_P]

    # Solve ODE in one go
    ODE_solution = solve_ivp(
        combined_system,
        (0, max_time),
        IC_current,
        max_step=0.001,
        method="RK23",
        rtol=1e-3,
        atol=1e-6,
        args=(Next_Conditions, num_cardio, num_cardio_control, Input_Parameters)
    )

    if ODE_solution.status == -1:
        return 0.0, 0.0, 0.0

    # Post-processing: use buffer to get recent data
    i_buffer = Next_Conditions["i"].item() % BUFFER_LIMIT

    P_sa = np.concatenate((Next_Conditions["P_sa_store"][i_buffer:], Next_Conditions["P_sa_store"][:i_buffer]))
    peaks, _ = find_peaks(P_sa, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-P_sa, distance=int(500), prominence=1)

    last_10_troughs_P_sa = troughs[-10:-1]
    last_10_min_P_sa = P_sa[last_10_troughs_P_sa]

    last_10_peaks_P_sa = peaks[-10:-1]
    last_10_max_P_sa = P_sa[last_10_peaks_P_sa]

    V_lv = np.concatenate((Next_Conditions["V_lv_store"][i_buffer:], Next_Conditions["V_lv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_lv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_lv, distance=int(500), prominence=1)

    last_10_troughs_V_lv = troughs[-10:-1]
    last_10_min_V_lv = V_lv[last_10_troughs_V_lv]

    last_10_peaks_V_lv = peaks[-10:-1]
    last_10_max_V_lv = V_lv[last_10_peaks_V_lv]

    V_rv = np.concatenate((Next_Conditions["V_rv_store"][i_buffer:], Next_Conditions["V_rv_store"][:i_buffer]))
    peaks, _ = find_peaks(V_rv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-V_rv, distance=int(500), prominence=1)

    last_10_troughs_V_rv = troughs[-10:-1]
    last_10_min_V_rv = V_rv[last_10_troughs_V_rv]

    last_10_peaks_V_rv = peaks[-10:-1]
    last_10_max_V_rv = V_rv[last_10_peaks_V_rv]

    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i_buffer:], Next_Conditions["P_rv_store"][:i_buffer]))
    peaks, _ = find_peaks(P_rv, distance=int(500), prominence=1)
    troughs, _ = find_peaks(-P_rv, distance=int(500), prominence=1)

    last_10_troughs_P_rv = troughs[-10:-1]
    last_10_min_P_rv = P_rv[last_10_troughs_P_rv]

    last_10_peaks_P_rv = peaks[-10:-1]
    last_10_max_P_rv = P_rv[last_10_peaks_P_rv]

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



    # left atria
    V_la = np.concatenate((Next_Conditions["V_la_store"][i_buffer:], Next_Conditions["V_la_store"][:i_buffer]))
    peaks, _ = find_peaks(V_la, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-V_la, distance=int(1000), prominence=1)

    last_10_troughs_V_la = troughs[-10:-1]
    last_10_min_V_la = V_la[last_10_troughs_V_la]

    last_10_peaks_V_la = peaks[-10:-1]
    last_10_max_V_la = V_la[last_10_peaks_V_la]

    P_la = np.concatenate((Next_Conditions["P_la_store"][i_buffer:], Next_Conditions["P_la_store"][:i_buffer]))
    peaks, _ = find_peaks(P_la, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-P_la, distance=int(1000), prominence=1)

    last_10_troughs_P_la = troughs[-10:-1]
    last_10_min_P_la = P_la[last_10_troughs_P_la]

    last_10_peaks_P_la = peaks[-10:-1]
    last_10_max_P_la = P_la[last_10_peaks_P_la]


    # right atria
    V_ra = np.concatenate((Next_Conditions["V_ra_store"][i_buffer:], Next_Conditions["V_ra_store"][:i_buffer]))
    peaks, _ = find_peaks(V_ra, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-V_ra, distance=int(1000), prominence=1)

    last_10_troughs_V_ra = troughs[-10:-1]
    last_10_min_V_ra = V_ra[last_10_troughs_V_ra]

    last_10_peaks_V_ra = peaks[-10:-1]
    last_10_max_V_ra = V_ra[last_10_peaks_V_ra]

    P_ra = np.concatenate((Next_Conditions["P_ra_store"][i_buffer:], Next_Conditions["P_ra_store"][:i_buffer]))
    peaks, _ = find_peaks(P_ra, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-P_ra, distance=int(1000), prominence=1)

    last_10_troughs_P_ra = troughs[-10:-1]
    last_10_min_P_ra = P_ra[last_10_troughs_P_ra]

    last_10_peaks_P_ra = peaks[-10:-1]
    last_10_max_P_ra = P_ra[last_10_peaks_P_ra]

    # get volume before atrial contraction
    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i_buffer:], Next_Conditions["phi_atr_store"][:i_buffer]))
    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1
    local_mins = starts[-10:]
    last_10_b4_LA_atrial_contract = V_la[local_mins]
    last_10_b4_RA_atrial_contract = V_ra[local_mins]

    # maximum ventricular pressure derivative
    P_lv = np.concatenate((Next_Conditions["P_lv_store"][i_buffer:], Next_Conditions["P_lv_store"][:i_buffer]))
    all_time = np.concatenate((Next_Conditions["all_time"][i_buffer:], Next_Conditions["all_time"][:i_buffer]))
    dPmax_lv_dt1 = np.gradient(P_lv, all_time)
    dPmax_lv_dt = savgol_filter(dPmax_lv_dt1, window_length=11, polyorder=3)
    peaks, _ = find_peaks(dPmax_lv_dt, distance=int(1000), prominence=1)
    last_10 = peaks[-10:-1]
    last_10_max_P_lv_deriv = dPmax_lv_dt[last_10]

    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i_buffer:], Next_Conditions["P_rv_store"][:i_buffer]))

    dPmax_rv_dt1 = np.gradient(P_rv, all_time)
    dPmax_rv_dt = savgol_filter(dPmax_rv_dt1, window_length=11, polyorder=3)
    peaks, _ = find_peaks(dPmax_rv_dt, distance=int(1000), prominence=1)
    last_10 = peaks[-10:-1]
    last_10_max_P_rv_deriv = dPmax_rv_dt[last_10]


    cardiac_output = np.mean(Next_Conditions["Q_pp_store"])


    # np.savez(f'HR_vs_time.npz', HR=Next_Conditions["HR_check"], time=Next_Conditions["time_history"], HR_average = Next_Conditions["HR"])
    print(np.mean(past_10_flat_segments), np.mean(last_10_max_P_sa), np.mean(last_10_min_P_sa),
            np.mean(last_10_max_V_lv), np.mean(last_10_min_V_lv), np.mean(last_10_max_V_rv), np.mean(last_10_min_V_rv),
            np.mean(last_10_max_P_rv), np.mean(last_10_min_P_rv),
            np.mean(last_10_min_V_ra), np.mean(last_10_max_V_ra), np.mean(last_10_min_P_ra), np.mean(last_10_max_P_ra),
            np.mean(last_10_min_V_la), np.mean(last_10_max_V_la), np.mean(last_10_min_P_la), np.mean(last_10_max_P_la),
            np.mean(last_10_b4_LA_atrial_contract), np.mean(last_10_b4_RA_atrial_contract),
            np.mean(last_10_max_P_lv_deriv), np.mean(last_10_max_P_rv_deriv), cardiac_output)


    return ODE_solution


if __name__ == "__main__":


    solution = simulate()
    print("ODE Status:", solution.status)
    print("ODE Message:", solution.message)

    time = solution.t
    state_variables = solution.y

    state_variable_names = (
            required_cardio_keys +
            required_cardio_control_keys)

    index = np.where(Next_Conditions["time_history"] == 1e6)[0][0] - 1
    print(len(Next_Conditions["time_history"][:index]))

    # Set global style
    plt.rcParams.update({
        "font.size": 14,  # Larger font
        # "font.weight": "bold",  # Bold text
        # "axes.labelweight": "bold",
        "axes.titlesize": 16,
        # "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,  # Thicker lines
    })


    i = Next_Conditions["i"].item() % BUFFER_LIMIT

    # RA
    plt.plot(Next_Conditions["VT_ra"][index - 10000:index], Next_Conditions["P_ra"][index - 10000:index],
             label="RA")
    # plt.plot(Next_Conditions["VT_la"][index - 5000:index], Next_Conditions["P_la"][index - 5000:index],
    #          label="LA")
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.show()

    # Flows
    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Q_lv"][index - 7500:index],
             label="Q$_{LV}$ (out of LV)")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Q_la"][index - 7500:index],
             label="Q$_{LA}$ (into LA)")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Q_ra"][index - 7500:index],
             label="Q$_{RA}$ (into RA)")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Q_rv"][index - 7500:index],
             label="Q$_{RV}$ (out of RV)")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Qi_lv"][index - 7500:index],
             label="Q$_{Mitral}$")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Qi_rv"][index - 7500:index],
             label="Q$_{Tricuspid}$")
    ax1.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["Q_vc"][index - 7500:index],
             label="Q$_{Vena Cava}$")

    ax1.set_xlabel("Time (s)")
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax1.set_ylabel("Flow (mL/s)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["Q_bp"][index - 75000:index], label="Q$_{bp}$", color="y")
    # ax2.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["Q_vc"][index - 75000:index], label="Q_vc", color="b")
    ax2.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["P_ra"][index - 7500:index], label="P_ra", linestyle="--", color="grey")
    ax2.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["P_vc"][index - 7500:index], label="P_vc", linestyle="--", color="c")
    ax2.plot(Next_Conditions["time_history"][index - 7500:index], Next_Conditions["P_rv"][index - 7500:index], label="P_rv", linestyle="--", color="m")

    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

    plt.show()
    plt.show()



    fig, ax1 = plt.subplots()
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Qi_rv"][:index], label="Qi_rv", color="g")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_pp"][:index], label="Q_pp", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_vc"][:index], label="P_vc", color="c")
    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_ra"][:index], label="VT_ra", color='c')

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Qi_rv"][:index], label="Q$_{Tricuspid}$", color="y")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_vc"][:index], label="Q_vc", color="b")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_ra"][:index], label="Q$_{RA}$ (into RA)")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_ev"][:index], label="Q_ev", color="tomato")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_hv"][:index], label="Q_hv", color='aquamarine')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sv"][:index], label="Q_sv", color='m')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_bv"][:index], label="Q_bv", color='k')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_rmv"][:index], label="Q_rmv", color="g")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_amv"][:index], label="Q_amv", color='plum')
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_amp"][:index], label="Q_amp", color='saddlebrown')


    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")

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

    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="b")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="c")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa", color="k")


    # ax2.set_ylabel("Flow (mL/s)", color="k")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    # #
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_ab"][:index], label="Baroreceptor firing",
             color="r")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    plt.show()


    fig, ax1 = plt.subplots()
    print(len(Next_Conditions["time_history"][:index]))

    # ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["time_since_beat"][:index], label="time_since_beat", color="g")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi"][:index], label="phi", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["phi_atr"][:index], label="phi_atr", color="k")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)


    plt.show()


    sorted_times = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))


    # left atria

    V_la = np.concatenate((Next_Conditions["V_la_store"][i:], Next_Conditions["V_la_store"][:i]))
    peaks, _ = find_peaks(V_la, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-V_la, distance=int(1000), prominence=1)

    last_10_troughs_V_la = troughs[-10:-1]
    last_10_min_V_la = V_la[last_10_troughs_V_la]

    last_10_peaks_V_la = peaks[-10:-1]
    last_10_max_V_la = V_la[last_10_peaks_V_la]

    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1

    local_mins = starts[-10:]



    fig, ax1 = plt.subplots()
    ax1.plot(sorted_times, V_la, label="V_la")

    ax1.scatter(sorted_times[troughs], V_la[troughs], color='r', marker='o',label="Atrial max volume during V-wave")
    ax1.scatter(sorted_times[peaks], V_la[peaks], color='g', marker='x', label="Atrial ESV")
    ax1.scatter(sorted_times[local_mins], V_la[local_mins], color='k', marker='o', label="Atrial EDV")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()

    P_la = np.concatenate((Next_Conditions["P_la_store"][i:], Next_Conditions["P_la_store"][:i]))
    peaks, _ = find_peaks(P_la, distance=int(2000), prominence=1)
    troughs, _ = find_peaks(-P_la, distance=int(2000), prominence=1)

    last_10_troughs_P_la = troughs[-10:-1]
    last_10_min_P_la = P_la[last_10_troughs_P_la]

    last_10_peaks_P_la = peaks[-10:-1]
    last_10_max_P_la = P_la[last_10_peaks_P_la]

    # print(np.mean(last_10_min_P_la), np.mean(last_10_max_P_la))

    fig, ax1 = plt.subplots()
    ax1.plot(sorted_times, P_la, label="P_la")

    ax1.scatter(sorted_times[troughs], P_la[troughs], color='r', marker='o', label="Detected Minima")
    ax1.scatter(sorted_times[peaks], P_la[peaks], color='g', marker='x', label="Detected Maxima")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()






    # right atria
    V_ra = np.concatenate((Next_Conditions["V_ra_store"][i:], Next_Conditions["V_ra_store"][:i]))
    peaks, _ = find_peaks(V_ra, distance=int(1000), prominence=1)
    troughs, _ = find_peaks(-V_ra, distance=int(1000), prominence=1)

    last_10_troughs_V_ra = troughs[-10:-1]
    last_10_min_V_ra = V_ra[last_10_troughs_V_ra]

    last_10_peaks_V_ra = peaks[-10:-1]
    last_10_max_V_ra = V_ra[last_10_peaks_V_ra]

    phi_atr = np.concatenate((Next_Conditions["phi_atr_store"][i:], Next_Conditions["phi_atr_store"][:i]))
    # Find transitions: where phi_atr goes from 0 to >0
    starts = np.where((phi_atr[:-1] == 0) & (phi_atr[1:] > 0))[0] + 1

    local_mins = starts[-10:]


    fig, ax1 = plt.subplots()
    ax1.plot(sorted_times, V_ra, label="V_ra")

    ax1.scatter(sorted_times[troughs], V_ra[troughs], color='r', marker='o', label="Atrial max volume during V-wave")
    ax1.scatter(sorted_times[peaks], V_ra[peaks], color='g', marker='x', label="Atrial ESV")
    ax1.scatter(sorted_times[local_mins], V_ra[local_mins], color='k', marker='o', label="Atrial EDV")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()

    P_ra = np.concatenate((Next_Conditions["P_ra_store"][i:], Next_Conditions["P_ra_store"][:i]))
    peaks, _ = find_peaks(P_ra, distance=int(2000), prominence=1)
    troughs, _ = find_peaks(-P_ra, distance=int(2000), prominence=1)

    last_10_troughs_P_ra = troughs[-10:-1]
    last_10_min_P_ra = P_ra[last_10_troughs_P_ra]

    last_10_peaks_P_ra = peaks[-10:-1]
    last_10_max_P_ra = P_ra[last_10_peaks_P_ra]


    fig, ax1 = plt.subplots()
    ax1.plot(sorted_times, P_ra, label="P_ra")

    ax1.scatter(sorted_times[troughs], P_ra[troughs], color='r', marker='o', label="Detected Minima")
    ax1.scatter(sorted_times[peaks], P_ra[peaks], color='g', marker='x', label="Detected Maxima")

    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.show()


    P_lv = np.concatenate((Next_Conditions["P_lv_store"][i:], Next_Conditions["P_lv_store"][:i]))
    all_time = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))

    dPmax_lv_dt1 = np.gradient(P_lv, all_time)
    dPmax_lv_dt = savgol_filter(dPmax_lv_dt1, window_length=11, polyorder=3)
    peaks1, _ = find_peaks(dPmax_lv_dt, distance=int(1000), prominence=10)


    P_rv = np.concatenate((Next_Conditions["P_rv_store"][i:], Next_Conditions["P_rv_store"][:i]))
    all_time = np.concatenate((Next_Conditions["all_time"][i:], Next_Conditions["all_time"][:i]))

    time_for_deriv = all_time
    dPmax_rv_dt1 = np.gradient(P_rv, all_time)
    dPmax_rv_dt = savgol_filter(dPmax_rv_dt1, window_length=11, polyorder=3)
    peaks, _ = find_peaks(dPmax_rv_dt, distance=int(1000), prominence=10)

    fig, ax1 = plt.subplots()
    ax1.plot(all_time, P_rv, label="P_rv", color="r")
    ax1.plot(all_time, P_lv, label="P_lv", color="k")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pressure (mmHg)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # # plt.show()
    ax2 = ax1.twinx()

    ax2.plot(time_for_deriv, dPmax_rv_dt1, label="dP_rv_dt", color="m")
    ax2.plot(time_for_deriv, dPmax_lv_dt1, label="dP_lv_dt", color="c")
    ax2.plot(all_time, dPmax_rv_dt, label="dP_rv_dt_smooth", color="b")
    ax2.plot(all_time, dPmax_lv_dt, label="dP_lv_dt_smooth", color="g")
    ax2.scatter(sorted_times[peaks], dPmax_rv_dt[peaks], color='r', marker='o', label="Max dP_rv_dt")
    ax2.scatter(sorted_times[peaks1], dPmax_lv_dt[peaks1], color='k', marker='o', label="Max dP_lv_dt")

    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], 57.2958 * Next_Conditions["theta_tr"][:index],
             label="Tricuspid valve flow", color="r")  #
    ax1.plot(Next_Conditions["time_history"][:index], 57.2958 * Next_Conditions["theta_mi"][:index],
             label="Mitral valve flow", color="b")  #

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Valve_angle")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    # # plt.show()
    ax2 = ax1.twinx()
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_sa"][:index], label="Q_sa", color="dimgrey")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["Q_lv"][:index], label="Q_lv", color="m")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="m")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color="y")
    # ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa", color="darkorange")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_rv"][:index], label="P_rv", color="dimgrey")
    ax2.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_ra"][:index], label="P_ra", color="g")

    ax2.set_ylabel("Flow (mL/s)")
    # plt.title("Pressure-Volume Traces")
    ax2.tick_params(axis='y', labelcolor="k")
    ax2.legend(loc="upper right")
    plt.show()



    # LV
    plt.plot(Next_Conditions["VT_lv"][index - 100000:index], Next_Conditions["P_lv"][index - 100000:index],
             label="LV", linewidth=2.5)
    plt.plot(Next_Conditions["VT_rv"][index - 20000:index], Next_Conditions["P_rv"][index - 20000:index],
             label="RV", linewidth=2.5)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][index - 75000:index], 60 * Next_Conditions["HR"][index - 75000:index], label="Heart Rate", color="r")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Heart Rate (bpm)")
    ax1.tick_params(axis='y', labelcolor="k")
    # ax1.set_ylim(top=135)  # y-axis of ax1 goes up to 135

    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["Emax_rv"][index - 75000:index], label="RV Max Elastance", color="g")
    ax2.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["Emax_lv"][index - 75000:index], label="LV Max Elastance", color='b')

    ax2.tick_params(axis='y', labelcolor="k")
    ax2.set_ylabel("Elastance (mmHg/ml)")

    # ax2.set_ylim(top=4.5)  # y-axis of ax2 goes up to 4.5
    ax2.legend()


    plt.show()

    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_ev"][:index], label="Extrasplanchnic V$_{Unstressed}$")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_amv"][:index], label="Active Muscle V$_{Unstressed}$")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_rmv"][:index], label="Resting Muscle V$_{Unstressed}$")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["Vu_sv"][:index], label="Splanchnic V$_{Unstressed}$")


    # Add labels and legend
    plt.ylabel("Volume (mL)")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.show()

    plt.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["R_ep"][index - 75000:index], label="Extrasplanchnic R$_{Peripheral}$")
    plt.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["R_amp"][index - 75000:index], label="Active Muscle R$_{Peripheral}$")
    plt.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["R_rmp"][index - 75000:index], label="Resting Muscle R$_{Peripheral}$")
    plt.plot(Next_Conditions["time_history"][index - 75000:index], Next_Conditions["R_sp"][index - 75000:index], label="Splanchnic R$_{Peripheral}$")

    # Add labels and legend
    plt.ylabel("Resistance (mmHg·s/ml)")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.show()




    plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_tr"][index - 8000:index],
             label="Tricuspid")  #
    plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_mi"][index - 8000:index], label="Mitral")  #
    plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_ao"][index - 8000:index], label="Aortic")  #
    plt.plot(Next_Conditions["time_history"][index - 8000:index], 57.2958 * Next_Conditions["theta_po"][index - 8000:index],
             label="Pulmonary")  #

    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    plt.ylabel("Valve Angle (degrees)")
    # plt.title("Pressure-Volume Traces")
    # Rotate tick labels
    # plt.xticks(rotation=45)
    plt.xlabel("Time (s)")

    # Legend in upper left
    plt.legend(loc="upper left")
    # plt.grid(True)
    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_sa"][:index], label="P_sa", color='g')

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_la"][:index], label="P_la", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["P_lv"][:index], label="P_lv", color='b')

    ax1.set_xlabel("Time (s)")
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend()



    plt.show()


    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_ra"][:index], label="V_ra")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_rv"][:index], label="V_rv")

    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_la"][:index], label="V_la")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["VT_lv"][:index], label="V_lv")

    ax1.set_xlabel("Time (s)")
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend()

    plt.show()




    fig, ax1 = plt.subplots()
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sp"][:index],
             label="Peripheral Resistance Sympathetic Activity", color="r")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sv"][:index], label="Venous Volume Sympathetic Activity", color="b")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_sh"][:index], label="HR & Contractility Sympathetic Activity", color="k")
    ax1.plot(Next_Conditions["time_history"][:index], Next_Conditions["f_v"][:index], label="Vagal Activity", color="g")

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Firing rate (spikes/s)")
    ax1.tick_params(axis='y', labelcolor="k")
    ax1.legend(loc="center left")
    ax1.grid(True)
    plt.show()

    # Number of state variables
    num_variables = state_variables.shape[0]
    colors = plt.cm.tab20.colors  # Use the Tab20 colormap for up to 20 unique colors



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





    # Plot all state variables
    plt.figure(figsize=(14, 10))
    for i, label in enumerate(state_variable_names):
        # if label != "beta":
        if label in ["VT_pa", "VT_pp", "VT_pv", "Q_pa", "VT_la", "VT_lv", "VT_ra", "VT_rv", "VT_sv", "VT_bv",
                           "VT_hv", "VT_rmv", "VT_amv", "P_sp", "P_sa", "Q_sa", "VT_vc",
                         "theta_ao", "dtheta_ao_dt", "theta_po", "dtheta_po_dt", "theta_mi", "dtheta_mi_dt", "theta_tr", "dtheta_tr_dt",

     # Cardio controller state variables
        "theta_change_O2_sp", "theta_change_CO2_sp", "theta_change_O2_sv", "theta_change_CO2_sv", "theta_change_O2_sh",
        "theta_change_CO2_sh", "P_tilda", "f_ac", "f_ap", 'R_ep_change', "R_sp_change",
        "R_rmp_n_change", "R_amp_n_change", "Vu_ev_change", "Vu_sv_change", "Vu_rmv_change", "Vu_amv_change", "Emax_lv_change",
        "Emax_rv_change", "Ts_change", "Tv_change", 'xb_O2', "xb_CO2", "xh_O2", 'xh_CO2', "Wh", 'xrm_O2', 'xrm_CO2', 'xam_O2', "xM", "x_met", "P_n_current"]:  # Skip "Wh"
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







    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR_check"][:index], label="HR Averaged")
    plt.plot(Next_Conditions["time_history"][:index], Next_Conditions["HR"][:index], label="HR")


    plt.ylabel("")
    plt.xlabel("Time (s)")
    plt.title("Traces")
    plt.legend()
    plt.grid(True)
    plt.show()



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