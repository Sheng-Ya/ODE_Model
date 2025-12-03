import numpy as np
from numba import njit

@njit
def activation_H(ti, atr, T, rise_time_atr, fall_time_atr, rise_time_ven, fall_time_ven, ahead1):
    tr_atr = rise_time_atr * T
    td_atr = fall_time_atr * T
    tr_ven = rise_time_ven * T
    td_ven = fall_time_ven * T

    if ti <= ahead1 * T:
        t_la = ti + (1-ahead1) * T
    else:
        t_la = ti - ahead1 * T

    if atr == 1:
        if t_la <= tr_atr:
            return 0.5 * (1.0 - np.cos(np.pi * (t_la / tr_atr)**1))
        elif t_la <= td_atr:
            return 0.5 * (1.0 + np.cos(np.pi * (t_la - tr_atr) / (td_atr - tr_atr)))
        else:
            return 0.0
    else:
        if ti <= tr_ven:
            return 0.5 * (1.0 - np.cos(np.pi * ti / tr_ven))
        elif ti <= td_ven:
            return 0.5 * (1.0 + np.cos(np.pi * (ti - tr_ven) / (td_ven - tr_ven)))
        else:
            return 0.0


@njit
def accept_index_evals(finish_time, all_time, last_index, buffer_limit, i):
    if finish_time >= all_time[0]:  # No wrap-around
        idx_in_2 = np.searchsorted(all_time[:last_index + 1], finish_time, side='right')
    else:  # Wrap-around
        idx_in_sorted2 = np.searchsorted(all_time[last_index + 1:], finish_time, side='right')
        idx_in_2 = (idx_in_sorted2 + last_index + 1) % buffer_limit

    if idx_in_2 <= last_index:
        indices = np.arange(idx_in_2, last_index + 1)
    else:
        indices = np.concatenate((np.arange(idx_in_2, buffer_limit), np.arange(0, last_index + 1)))

    # Take every 3rd step, rk23 - optimized loop
    indices_len = len(indices)
    mask = np.array([(i - 2 - j) % 3 == 0 and (i - 2 - j) > 0 for j in range(indices_len)])
    accepted_index = [indices[indices_len - 1 - j] for j in range(indices_len) if mask[j]]

    return accepted_index


@njit
def get_delayed_value(t, delay, all_time, heart_index, buffer_limit, history_array, default_value):
    delay_time = t - delay

    if delay_time < 0:
        return default_value

    if delay_time >= all_time[0]:
        # No wrap-around
        delay_index1 = np.searchsorted(all_time[:heart_index + 1], delay_time, side='right')
    else:
        # Wrap-around
        idx_in_sorted = np.searchsorted(all_time[heart_index + 1:], delay_time, side='right')
        delay_index1 = (idx_in_sorted + heart_index + 1) % buffer_limit

    delay_index0 = (delay_index1 - 1) % buffer_limit
    t1 = all_time[delay_index1]
    t0 = all_time[delay_index0]
    v1 = history_array[delay_index1]
    v0 = history_array[delay_index0]

    return float(v0 + (v1 - v0) * (delay_time - t0) / (t1 - t0))


@njit
def compute_mean_selected(HR_store, indices):
    total = 0.0
    for idx in indices:
        total += HR_store[idx]

    return total / len(indices)


@njit
def njit_compatible(t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, HR_store, time_since_beat_store,
                    HR_every_store, Vu_ev_every_store, Vu_sv_every_store, Vu_rmv_every_store, Vu_amv_every_store,
                    Emax_lv_every_store,
                    Emax_rv_every_store, Vu_ev_store, Vu_sv_store, Vu_rmv_store, Vu_amv_store, Emax_lv_store,
                    Emax_rv_store,
                    f_sp_history, f_sh_history, f_v_history, f_sv_history):
    """
    Main derivative computation function with improved organization
    Computes all system derivatives in a single optimized function
    """

    # # State variables
    (  # Cardio state variables
        VT_pa, VT_pp, VT_pv, Q_pa,
        VT_la, VT_lv, VT_ra, VT_rv,
        VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, P_sp, P_sa, Q_sa, VT_vc,
        theta_ao, dtheta_ao_dt, theta_po, dtheta_po_dt, theta_mi, dtheta_mi_dt, theta_tr, dtheta_tr_dt,

        # Cardio controller state variables
        P_tilda, R_ep_change, R_sp_change,
        R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
        Emax_rv_change, Ts_change, Tv_change, P_n_current

    ) = state

    # ============================================================================
    # PARAMETER EXTRACTION
    # ============================================================================
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
     fall_time_ven, ahead1, theta_min, delta_P
     ) = Input_Parameters

    # Determine the correct index based on t
    if t == 0:
        last_index = i % BUFFER_LIMIT
    else:
        last_index = (i - num_removed - 1) % BUFFER_LIMIT

    
    # ============================================================================
    # CARDIOVASCULAR CONTROLLER
    # ============================================================================
    T = 1 / HR_store[last_index]  # Heart period

    # Resistance calculations with improved organization
    R_ep = R_ep_change + R_ep0
    R_sp = R_sp_change + R_sp0

    # Active muscle resistance with metabolic feedback
    R_amp_n = R_amp_n_change + R_amp0

    # Resting muscle resistance with CO2/O2 feedback
    R_rmp_n = R_rmp_n_change + R_rmp0


    time_since_beat = time_since_beat_store[last_index]
    # Update after every heartbeat
    if t - time_since_beat > T:
        accepted_indices = accept_index_evals(time_since_beat, all_time, last_index, BUFFER_LIMIT, (i - num_removed))

        time_since_beat = time_since_beat + T

        HR = compute_mean_selected(HR_every_store, accepted_indices)
        T = 1 / HR
        Vu_ev = compute_mean_selected(Vu_ev_every_store, accepted_indices)
        Vu_sv = compute_mean_selected(Vu_sv_every_store, accepted_indices)
        Vu_rmv = compute_mean_selected(Vu_rmv_every_store, accepted_indices)
        Vu_amv = compute_mean_selected(Vu_amv_every_store, accepted_indices)
        Emax_lv = compute_mean_selected(Emax_lv_every_store, accepted_indices)
        Emax_rv = compute_mean_selected(Emax_rv_every_store, accepted_indices)

    else:
        HR = HR_store[last_index]
        Vu_ev = Vu_ev_store[last_index]  # previous mean value
        Vu_sv = Vu_sv_store[last_index]  # previous mean value
        Vu_rmv = Vu_rmv_store[last_index]  # previous mean value
        Vu_amv = Vu_amv_store[last_index]  # previous mean value
        Emax_lv = Emax_lv_store[last_index]  # previous mean value
        Emax_rv = Emax_rv_store[last_index]  # previous mean value

    Vu_amv_check = Vu_amv

    # ============================================================================
    # CARDIOVASCULAR SYSTEM
    # ============================================================================
    
    if VT_pa > Vu_pa:
        V_pa = VT_pa - Vu_pa
    else:
        V_pa = 0
        Vu_pa = VT_pa

    P_pa = V_pa / C_pa

    if VT_pp > Vu_pp:
        V_pp = VT_pp - Vu_pp
    else:
        V_pp = 0
        Vu_pp = VT_pp

    P_pp = V_pp / C_pp

    if VT_pv > Vu_pv:
        V_pv = VT_pv - Vu_pv
    else:
        V_pv = 0
        Vu_pv = VT_pv

    P_pv = V_pv / C_pv

    ## The Heart
    if VT_la > Vu_la:  # LA stressed volume is the total minus unstressed
        V_la = VT_la - Vu_la
    else:
        V_la = 0
        Vu_la = VT_la

    if VT_ra > Vu_ra:  # RA stressed volume is the total minus unstressed
        V_ra = VT_ra - Vu_ra
    else:
        V_ra = 0
        Vu_ra = VT_ra

    if VT_rv > Vu_rv:  # RV stressed volume is the total minus unstressed
        V_rv = VT_rv - Vu_rv
    else:
        V_rv = 0
        Vu_rv = VT_rv

    # V_lv can be growing but there should not be any flow (Q) into the ventricles?
    if VT_lv > Vu_lv:  # LV stressed volume is the total minus unstressed
        V_lv = VT_lv - Vu_lv
    else:
        V_lv = 0
        Vu_lv = VT_lv

    # activation function for contraction of the ventricle and atria
    phi = activation_H(t - time_since_beat, 0, T, rise_time_atr, fall_time_atr, rise_time_ven, fall_time_ven, ahead1)
    phi_atr = activation_H(t - time_since_beat, 1, T, rise_time_atr, fall_time_atr, rise_time_ven, fall_time_ven,
                           ahead1)

    # changing from 25 to 10 will move up the PV curve for phi_atr
    V_shift1 = shift_param1 / (shift_param2 * (phi * Emax_rv + (1 - phi) * P0_rv * KE_rv * (np.exp(KE_rv * VT_rv))) + (
                phi_atr * Emax_ra + (1 - phi_atr) * P0_ra * KE_ra * (np.exp(KE_ra * VT_ra))))

    V_shift2 = shift_param3 / (shift_param4 * (phi * Emax_lv + (1 - phi) * P0_lv * KE_lv * (np.exp(KE_lv * VT_lv))) + (
                phi_atr * Emax_la + (1 - phi_atr) * P0_la * KE_la * (np.exp(KE_la * VT_la))))


    P_lv = phi * Emax_lv * (VT_lv - Vu_lv) + (1 - phi) * P0_lv * (np.exp(KE_lv * VT_lv) - 1) 
    P_ra = phi_atr * Emax_ra * (VT_ra - Vu_ra - V_shift1) + (1 - phi_atr) * P0_ra * (
                np.exp(KE_ra * (VT_ra - V_shift1)) - 1) 
    P_rv = phi * Emax_rv * (VT_rv - Vu_rv) + (1 - phi) * P0_rv * (np.exp(KE_rv * VT_rv) - 1) 
    P_la = phi_atr * Emax_la * (VT_la - Vu_la - V_shift2) + (1 - phi_atr) * P0_la * (
                np.exp(KE_la * (VT_la - V_shift2)) - 1) 


    # Smooth valve state transition
    valve_signal = 0.5 * (1 + np.tanh((P_lv - P_sa) / delta_P))
    if abs(valve_signal) < 1e-8:
        theta_ao = theta_min

    if theta_ao > theta_ao_max:
        theta_ao = theta_ao_max
    elif theta_ao < theta_min:
        theta_ao = theta_min

    # Compute area ratio with smooth transition
    AR_ao = valve_signal * ((1 - np.cos(theta_ao)) ** 2) / ((1 - np.cos(theta_ao_max)) ** 2)

    # Flow with smooth transition
    Q_lv = valve_signal * (np.sqrt(np.maximum(P_lv - P_sa, 0)) * AR_ao * R_ao)

    # Dynamics with smooth transition
    d2theta_ao_dt2 = valve_signal * ((P_lv - P_sa) * Kp_ao * np.cos(theta_ao) - Kf_ao * dtheta_ao_dt +
                                     Kb_ao * Q_lv * np.cos(theta_ao) - Kv_ao * Q_lv * np.sin(2 * theta_ao))


    valve_signal = 0.5 * (1 + np.tanh((P_la - P_lv) / delta_P))
    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_mi = theta_min  # minimum angle (closed)

    if theta_mi > theta_mi_max:
        theta_mi = theta_mi_max
    elif theta_mi < theta_min:
        theta_mi = theta_min

    # Compute area ratio with smooth transition
    AR_mi = valve_signal * ((1 - np.cos(theta_mi)) ** 2) / ((1 - np.cos(theta_mi_max)) ** 2)
    AR_mi = 1

    # Flow with smooth transition
    Qi_lv = valve_signal * (np.sqrt(np.maximum(P_la - P_lv, 0)) * AR_mi * R_mi)

    # Dynamics with smooth transition
    d2theta_mi_dt2 = valve_signal * ((P_la - P_lv) * Kp_mi * np.cos(theta_mi) - Kf_mi * dtheta_mi_dt +
                                     Kb_mi * Qi_lv * np.cos(theta_mi) - Kv_mi * Qi_lv * np.sin(2 * theta_mi))



    # Smooth valve state transition (pulmonary valve opens when RV pressure > PA pressure)
    valve_signal = 0.5 * (1 + np.tanh((P_rv - P_pa) / delta_P))

    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_po = theta_min  # minimum angle (closed)

    if theta_po > theta_po_max:
        theta_po = theta_po_max
        # AR_po = valve_signal * ((1 - np.cos(theta_po_max)) ** 2) / ((1 - np.cos(theta_po_max)) ** 2)
    elif theta_po < theta_min:
        theta_po = theta_min

    # Compute area ratio with smooth transition
    AR_po = valve_signal * ((1 - np.cos(theta_po)) ** 2) / ((1 - np.cos(theta_po_max)) ** 2)

    # Flow with smooth transition
    Q_rv = valve_signal * (np.sqrt(np.maximum(P_rv - P_pa, 0)) * AR_po * R_po)

    # Dynamics with smooth transition
    d2theta_po_dt2 = valve_signal * ((P_rv - P_pa) * Kp_po * np.cos(theta_po) - Kf_po * dtheta_po_dt +
                                     Kb_po * Q_rv * np.cos(theta_po) - Kv_po * Q_rv * np.sin(2 * theta_po))


    valve_signal = 0.5 * (1 + np.tanh((P_ra - P_rv) / delta_P))

    # Enforce theta bounds when nearly closed
    if abs(valve_signal) < 1e-8:
        theta_tr = theta_min  # minimum angle (closed)

    if theta_tr > theta_tr_max:
        theta_tr = theta_tr_max
        # AR_tr = valve_signal * ((1 - np.cos(theta_tr_max)) ** 2) / ((1 - np.cos(theta_tr_max)) ** 2)
    elif theta_tr < theta_min:
        theta_tr = theta_min

    # # Compute area ratio with smooth transition
    AR_tr = valve_signal * ((1 - np.cos(theta_tr)) ** 2) / ((1 - np.cos(theta_tr_max)) ** 2)
    AR_tr = 1

    P_vc = D1 + K1_vc * (VT_vc - Vu_vc)

    # Flow with smooth transition
    Qi_rv = valve_signal * (np.sqrt(np.maximum(P_ra - P_rv, 0)) * AR_tr * R_tr)
    P_ra = P_ra

    if VT_vc > Vu_vc:
        V_vc = VT_vc - Vu_vc
    else:
        V_vc = 0
        Vu_vc = VT_vc

    if V_vc > 0:
        R_vc = Kr_vc * (Vvc_max / V_vc) ** 2 + Rvc_n
    else:
        R_vc = Rvc_n

    Q_ra = (P_vc - P_ra) / R_vc

    ####################################

    Q_la = (P_pv - P_la) / R_pv
    Q_pp = (P_pp - P_pv) / R_pp

    dVT_pa_dt = Q_rv - Q_pa
    dVT_pp_dt = Q_pa - Q_pp
    dVT_pv_dt = Q_pp - Q_la
    dQ_pa_dt = (P_pa - R_pa * Q_pa - P_pp) / L_pa

    dVT_lv_dt = Qi_lv - Q_lv
    dVT_la_dt = Q_la - Qi_lv

    dVT_ra_dt = Q_ra - Qi_rv
    dVT_rv_dt = Qi_rv - Q_rv


    # Dynamics with smooth transition
    d2theta_tr_dt2 = valve_signal * ((P_ra - P_rv) * Kp_tr * np.cos(theta_tr) -
                                     Kf_tr * dtheta_tr_dt + Kb_tr * Qi_rv * np.cos(
                theta_tr) - Kv_tr * Qi_rv * np.sin(2 * theta_tr))




    ## systemic peripheral and venous circulation
    # splanchnic

    if VT_sv >= Vu_sv:
        V_sv = VT_sv - Vu_sv
        P_sv = V_sv / C_sv
    else:
        V_sv = 0
        P_sv = VT_sv / C_sv

    Q_sp = (P_sp - P_sv) / R_sp


    if P_sv >= P_vc:
        Q_sv = (P_sv - P_vc) / R_sv_n
    else:
        Q_sv = 0

    dVT_sv_dt = Q_sp - Q_sv

    # brain

    if VT_bv >= Vu_bv:
        V_bv = VT_bv - Vu_bv
        P_bv = V_bv / C_bv
    else:
        V_bv = 0
        P_bv = VT_bv / C_bv

    Q_bp = (P_sp - P_bv) / R_bpn

    P_b = 0

    if P_vc < P_b:
        R_bv = R_bv_n * ((P_bv - P_vc) / (P_bv - P_b))
    else:
        R_bv = R_bv_n

    if P_bv >= P_vc:
        Q_bv = (P_bv - P_vc) / R_bv
    else:
        Q_bv = 0

    dVT_bv_dt = Q_bp - Q_bv

    # coronary circulation

    if VT_hv >= Vu_hv:
        V_hv = VT_hv - Vu_hv
        P_hv = V_hv / C_hv
    else:
        V_hv = 0
        P_hv = VT_hv / C_hv
        # Vu_hv = VT_hv

    Q_hp = max(((P_sp - P_hv) / R_hpn), 0.0001)

    P_h = 0

    if P_vc < P_h:
        R_hv = R_hv_n * ((P_hv - P_vc) / (P_hv - P_h))
    else:
        R_hv = R_hv_n

    if P_hv >= P_vc:
        Q_hv = (P_hv - P_vc) / R_hv
    else:
        Q_hv = 0

    dVT_hv_dt = Q_hp - Q_hv

    # resting muscle
    # V_rmp = C_rmp * P_sp

    if VT_rmv >= Vu_rmv:
        V_rmv = VT_rmv - Vu_rmv
        P_rmv = V_rmv / C_rmv
    else:
        V_rmv = 0
        P_rmv = VT_rmv / C_rmv
        # Vu_rmv = VT_rmv

    Q_rmp = max((P_sp - P_rmv) / R_rmp_n, 0.0001)

    P_rm = 0

    if P_vc < P_rm:
        R_rmv = R_rmv_n * ((P_rmv - P_vc) / (P_rmv - P_rm))
    else:
        R_rmv = R_rmv_n

    if P_rmv >= P_vc:
        Q_rmv = (P_rmv - P_vc) / R_rmv
    else:
        Q_rmv = 0

    dVT_rmv_dt = Q_rmp - Q_rmv

    # active muscle
    P_0 = Vu_amv / (C_amv * 10)

    # P_im (intramuscular pressure) is 0 at rest so removed here
    if VT_amv >= Vu_amv:
        V_amv = VT_amv - Vu_amv
        P_amv = V_amv / C_amv
    else:
        V_amv = 0
        if VT_amv > 0:
            P_amv = P_0 * (1 - (VT_amv / Vu_amv) ** -scale_param2)
            Vu_amv = VT_amv
        else:
            P_amv = P_0
            VT_amv = 0
            Vu_amv = VT_amv

    Q_amp = max(((P_sp - P_amv) / R_amp_n), 0.0001)

    P_am = 0

    if P_vc < P_am:
        R_amv = R_amv_n * ((P_amv - P_vc) / (P_amv - P_am))
    else:
        R_amv = R_amv_n

    if P_amv >= P_vc:
        Q_amv = (P_amv - P_vc) / R_amv
    else:
        Q_amv = 0

    dVT_amv_dt = Q_amp - Q_amv

    ## systemic peripheral and venous circulation
    # extrasplanchnic
    # V_ep = C_ep * P_sp

    # C_jp = C_ep + C_sp + C_bp + C_hp + C_rmp + C_amp
    # Vu_jp = Vu_ep + Vu_sp + Vu_bp + Vu_hp + Vu_rmp + Vu_amp
    Vu_jv = Vu_ev + Vu_sv + Vu_bv + Vu_hv + Vu_rmv + Vu_amv

    V_u = Vu_sa + Vu_pa + Vu_pp + Vu_pv + Vu_ra + Vu_la + Vu_jp + Vu_jv + Vu_rv + Vu_lv + Vu_vc

    V_sa = P_sa * C_sa
    V_s_peripheral = P_sp * C_jp

    # left over volume
    V_ev = (V_tot - V_sa - V_ra - V_rv - V_la - V_lv - V_pa - V_pp - V_pv - V_sv - V_rmv - V_amv - V_bv
            - V_hv - V_vc - V_u - V_s_peripheral)

    P_ev = V_ev / C_ev  # + source_values

    Q_ep = (P_sp - P_ev) / R_ep

    P_e = 0

    if P_vc < P_e:
        R_ev = R_ev_n * ((P_ev - P_vc) / (P_ev - P_e))
    else:
        R_ev = R_ev_n

    if P_ev >= P_vc:
        Q_ev = (P_ev - P_vc) / R_ev
    else:
        Q_ev = 0

    Q_vc = Q_ev + Q_sv + Q_bv + Q_hv + Q_rmv + Q_amv
    Q_jp = Q_ep + Q_sp + Q_bp + Q_hp + Q_rmp + Q_amp

    # ignore VT_ev - doesn't add to the equations
    dP_sa_dt = (Q_lv - Q_sa) / C_sa
    # dVT_ev_dt = Q_ep - Q_ev
    dVT_vc_dt = Q_vc - Q_ra
    dP_sp_dt = (Q_sa - Q_jp) / C_jp
    dQ_sa_dt = (P_sa - R_sa * Q_sa - P_sp) / L_sa

    # ============================================================================
    # CARDIOVASCULAR CONTROLLER
    # ============================================================================

    ## Afferent Pathways
    dP_n_current_dt = (-P_n_current + P_n) / tau_p

    exp_arg = (P_tilda - P_n_current) / k_ab
    f_ab = (f_ab_min + f_ab_max * np.exp(exp_arg)) / (1 + np.exp(exp_arg))
    dP_tilda_dt = (P_sa + tau_z * dP_sa_dt - P_tilda) / tau_p

    ## Efferent Pathways constant parameters
    f_ash = Wb_sh * f_ab
    f_sh = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_ash)))

    f_asp = Wb_sp * f_ab
    f_sp = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asp)))

    f_asv = Wb_sv * f_ab
    f_sv = min(fes_max, (fes_inf + (fes_o - fes_inf) * np.exp(kes * f_asv)))

    f_v = (fev_o + fev_inf * np.exp((f_ab - fab_o) / kev)) / (1 + np.exp((f_ab - fab_o) / kev))

    # Fetch delayed values
    f_sp_delay2_Ramp = get_delayed_value(t, DR_amp, all_time, last_index, BUFFER_LIMIT, f_sp_history, 5.725338528121857)
    f_sp_delay2_Rep = get_delayed_value(t, DR_ep, all_time, last_index, BUFFER_LIMIT, f_sp_history, 5.725338528121857)
    f_sp_delay2_Rrmp = get_delayed_value(t, DR_rmp, all_time, last_index, BUFFER_LIMIT, f_sp_history, 5.725338528121857)
    f_sp_delay2_Rsp = get_delayed_value(t, DR_sp, all_time, last_index, BUFFER_LIMIT, f_sp_history, 5.725338528121857)

    f_sv_delay5_Vu_ev = get_delayed_value(t, DV_ev, all_time, last_index, BUFFER_LIMIT, f_sv_history, 7.261875634917504)
    f_sv_delay5_Vu_sv = get_delayed_value(t, DV_sv, all_time, last_index, BUFFER_LIMIT, f_sv_history, 7.261875634917504)
    f_sv_delay5_Vu_rmv = get_delayed_value(t, DV_rmv, all_time, last_index, BUFFER_LIMIT, f_sv_history,7.261875634917504)
    f_sv_delay5_Vu_amv = get_delayed_value(t, DV_amv, all_time, last_index, BUFFER_LIMIT, f_sv_history,7.261875634917504)

    f_sh_delay2_Emax_lv = get_delayed_value(t, DEmax_lv, all_time, last_index, BUFFER_LIMIT, f_sh_history,7.811885872859872)
    f_sh_delay2_Emax_rv = get_delayed_value(t, DEmax_rv, all_time, last_index, BUFFER_LIMIT, f_sh_history,7.811885872859872)

    f_sh_delay2_s = get_delayed_value(t, DT_s, all_time, last_index, BUFFER_LIMIT, f_sh_history, 7.811885872859872)
    f_v_delay0_2 = get_delayed_value(t, DT_v, all_time, last_index, BUFFER_LIMIT, f_v_history, 2.7719269200056793)

    # heart period
    sigma_Ts = GT_s * np.log(max(f_sh_delay2_s, fes_min) - fes_min + 1)
    d_Ts_change_dt = (- Ts_change + sigma_Ts) / tau_Ts

    sigma_Tv = GT_v * f_v_delay0_2
    d_Tv_change_dt = (- Tv_change + sigma_Tv) / tau_Tv

    T = Ts_change + Tv_change + T0
    HR_every = 1 / T

    # continue with equations
    sigma_Rep = GR_ep * np.log(max(f_sp_delay2_Rep, fes_min) - fes_min + 1)
    sigma_Rsp = GR_sp * np.log(max(f_sp_delay2_Rsp, fes_min) - fes_min + 1)
    sigma_Rrmp_n = GR_rmp * np.log(max(f_sp_delay2_Rrmp, fes_min) - fes_min + 1)
    sigma_Ramp_n = GR_amp * np.log(max(f_sp_delay2_Ramp, fes_min) - fes_min + 1)

    sigma_Vu_ev = GV_ev * np.log(max(f_sv_delay5_Vu_ev, fes_min) - fes_min + 1)
    sigma_Vu_sv = GV_sv * np.log(max(f_sv_delay5_Vu_sv, fes_min) - fes_min + 1)
    sigma_Vu_rmv = GV_rmv * np.log(max(f_sv_delay5_Vu_rmv, fes_min) - fes_min + 1)
    sigma_Vu_amv = GV_amv * np.log(max(f_sv_delay5_Vu_amv, fes_min) - fes_min + 1)

    sigma_Emax_lv = GEmax_lv * np.log(max(f_sh_delay2_Emax_lv, fes_min) - fes_min + 1)
    sigma_Emax_rv = GEmax_rv * np.log(max(f_sh_delay2_Emax_rv, fes_min) - fes_min + 1)

    dR_ep_change_dt = (- R_ep_change + sigma_Rep) / tau_Rep
    dR_sp_change_dt = (- R_sp_change + sigma_Rsp) / tau_Rsp
    dR_rmp_n_change_dt = (- R_rmp_n_change + sigma_Rrmp_n) / tau_Rrmp
    dR_amp_n_change_dt = (- R_amp_n_change + sigma_Ramp_n) / tau_Ramp

    dVu_ev_change_dt = (- Vu_ev_change + sigma_Vu_ev) / tau_Vev
    dVu_sv_change_dt = (- Vu_sv_change + sigma_Vu_sv) / tau_Vsv
    dVu_rmv_change_dt = (- Vu_rmv_change + sigma_Vu_rmv) / tau_Vrmv
    dVu_amv_change_dt = (- Vu_amv_change + sigma_Vu_amv) / tau_Vamv
    Vu_ev_every = max(Vu_ev_change + Vu_ev0, 0)
    Vu_sv_every = max(Vu_sv_change + Vu_sv0, 0)
    Vu_rmv_every = max(Vu_rmv_change + Vu_rmv0, 0)
    Vu_amv_every = max(Vu_amv_change + Vu_amv0, 0)

    dEmax_lv_change_dt = (- Emax_lv_change + sigma_Emax_lv) / tau_Emax_lv
    dEmax_rv_change_dt = (- Emax_rv_change + sigma_Emax_rv) / tau_Emax_rv
    Emax_lv_every = Emax_lv_change + Emax_lv0
    Emax_rv_every = Emax_rv_change + Emax_rv0



    Vu_amv = Vu_amv_check

    # ============================================================================
    # RETURN ALL COMPUTED VALUES
    # ============================================================================
    return (time_since_beat,
            HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
            Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, HR_every, Vu_ev_every, Vu_sv_every,
            Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,

            dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
            dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
            dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
            d2theta_tr_dt2,

            # cardio controller derivatives
            dP_tilda_dt, dR_ep_change_dt,
            dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
            dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
            d_Tv_change_dt, dP_n_current_dt,

            # just for plotting purposes
            Q_sp, Q_ep, Q_bp, Q_hp, Q_rmp, Q_amp, Q_pp, Q_la, Q_lv, Q_ra, Q_rv, P_ra, P_la, P_lv, P_rv, P_lv,
            P_rv, P_la, P_ra, P_pa, P_pp, P_pv, P_vc, Qi_lv, Qi_rv, phi, phi_atr, P_amv, P_ev, V_u, Q_vc, Q_amv, V_sa,
            P_bv, R_bv, Q_ev, R_ep, R_amp_n, R_rmp_n, R_sp, f_ab, f_sh_delay2_Emax_rv, f_v_delay0_2,
            sigma_Ts, sigma_Tv, P_n_current, V_shift1, theta_ao, theta_tr, theta_mi, theta_po,
            Q_bv, Q_hv, Q_rmv, Q_sv, AR_mi, AR_tr, V_ev, V_sv, V_rmv, V_amv)


def model_derivatives(t, state, updates, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters):
    """
    Main model derivatives function with improved organization
    Coordinates all system computations and updates
    """
    # ============================================================================
    # STATE VARIABLE EXTRACTION
    # ============================================================================
    (  # Cardio state variables
        VT_pa, VT_pp, VT_pv, Q_pa,
        VT_la, VT_lv, VT_ra, VT_rv,
        VT_sv, VT_bv, VT_hv, VT_rmv, VT_amv, P_sp, P_sa, Q_sa, VT_vc,
        theta_ao, dtheta_ao_dt, theta_po, dtheta_po_dt, theta_mi, dtheta_mi_dt, theta_tr, dtheta_tr_dt,

        # Cardio controller state variables
        P_tilda, R_ep_change, R_sp_change,
        R_rmp_n_change, R_amp_n_change, Vu_ev_change, Vu_sv_change, Vu_rmv_change, Vu_amv_change, Emax_lv_change,
        Emax_rv_change, Ts_change, Tv_change, P_n_current
    ) = state

    Input_Parameters = np.array(Input_Parameters)

    (HR_store, time_since_beat_store, HR_every_store, Vu_ev_every_store, Vu_sv_every_store, Vu_rmv_every_store,
     Vu_amv_every_store, Emax_lv_every_store, Emax_rv_every_store, Vu_ev_store, Vu_sv_store, Vu_rmv_store, Vu_amv_store,
     Emax_lv_store, Emax_rv_store, f_sp_history, f_sh_history, f_v_history, f_sv_history) = [
        updates[key] for key in
        ["HR_store", "time_since_beat_store", "HR_every_store", "Vu_ev_every_store", "Vu_sv_every_store",
         "Vu_rmv_every_store",
         "Vu_amv_every_store", "Emax_lv_every_store", "Emax_rv_every_store", "Vu_ev_store", "Vu_sv_store",
         "Vu_rmv_store", "Vu_amv_store",
         "Emax_lv_store", "Emax_rv_store", "f_sp_store", "f_sh_store", "f_v_store", "f_sv_store"]]

    (time_since_beat,
    HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
    Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, HR_every, Vu_ev_every, Vu_sv_every,
    Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every,

    dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
    dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
    dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
    d2theta_tr_dt2,

    # cardio controller derivatives
    dP_tilda_dt, dR_ep_change_dt,
    dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
    dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
    d_Tv_change_dt, dP_n_current_dt,

    # just for plotting purposes
    Q_sp, Q_ep, Q_bp, Q_hp, Q_rmp, Q_amp, Q_pp, Q_la, Q_lv, Q_ra, Q_rv, P_ra, P_la, P_lv, P_rv, P_lv,
    P_rv, P_la, P_ra, P_pa, P_pp, P_pv, P_vc, Qi_lv, Qi_rv, phi, phi_atr, P_amv, P_ev, V_u, Q_vc, Q_amv, V_sa,
    P_bv, R_bv, Q_ev, R_ep, R_amp_n, R_rmp_n, R_sp, f_ab, f_sh_delay2_Emax_rv, f_v_delay0_2,
    sigma_Ts, sigma_Tv, P_n_current, V_shift1, theta_ao, theta_tr, theta_mi, theta_po,
    Q_bv, Q_hv, Q_rmv, Q_sv, AR_mi, AR_tr, V_ev, V_sv, V_rmv, V_amv

     ) = njit_compatible(t, state, num_removed, i, BUFFER_LIMIT, all_time, Input_Parameters, HR_store,
                         time_since_beat_store, HR_every_store, Vu_ev_every_store,
                         Vu_sv_every_store, Vu_rmv_every_store, Vu_amv_every_store, Emax_lv_every_store,
                         Emax_rv_every_store,
                         Vu_ev_store, Vu_sv_store, Vu_rmv_store, Vu_amv_store, Emax_lv_store, Emax_rv_store,
                         f_sp_history, f_sh_history, f_v_history, f_sv_history)

    # Cardiovascular Controller
    # update values needed in other systems
    for key, new_value in zip(
            [  # Cardio inputs
                "time_since_beat_store",

                "HR_store", "Vu_ev_store", "Vu_sv_store", "Vu_rmv_store", "Vu_amv_store",
                "Emax_lv_store", "Emax_rv_store", "f_sp_store", "f_sh_store",
                "f_v_store", "f_sv_store", "HR_every_store", "Vu_ev_every_store",
                "Vu_sv_every_store", "Vu_rmv_every_store", "Vu_amv_every_store", "Emax_lv_every_store",
                "Emax_rv_every_store", "P_sa_store", "V_lv_store", "V_rv_store", "P_rv_store",
                "P_la_store", "V_la_store", "V_ra_store", "P_ra_store",

                # Needed in cardio controller
                "P_lv_store", "phi_atr_store", "Q_pp_store"],

            [time_since_beat,
             HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv,
             Emax_lv, Emax_rv, f_sp, f_sh, f_v, f_sv, HR_every, Vu_ev_every, Vu_sv_every,
             Vu_rmv_every, Vu_amv_every, Emax_lv_every, Emax_rv_every, P_sa, VT_lv, VT_rv, P_rv,
             P_la, VT_la, VT_ra, P_ra, P_lv, phi_atr, Q_pp]
    ):
        updates[key][((i - num_removed) % BUFFER_LIMIT)] = new_value



        # just for plotting purposes
    keys_and_values = zip(
        [
            # Cardio control inputs
            "P_sa", "Q_bp", "Q_hp", "Q_rmp", "Q_amp", "Q_sp", "Q_ep",

            # Gas exchange inputs
            "Q_pp", "Q_la",

            # For plot
            "Q_lv", "Q_ra", "Q_rv", "P_ra", "P_la", "P_lv", "P_rv",
            "P_la", "P_ra", "VT_rv", "VT_ra",
            "VT_lv", "VT_la", "P_pa", "P_pp", "P_pv", "P_vc", "Qi_lv",
            "Qi_rv", "phi", "phi_atr", "P_amv", "P_ev", "V_u",
            "P_sp", "Q_sa", "Q_vc", "VT_amv",
            "Q_amv", "Q_pa", "V_sa", "P_bv", "R_bv",
            "Q_ev", "Q_bv", "Q_hv", "Q_rmv", "Q_sv", "VT_pa", "VT_pp", "VT_pv", "VT_sv", "VT_bv", "VT_hv", "VT_rmv",
            "VT_vc", "time_history", "theta_ao", "theta_po", "theta_mi", "theta_tr", "V_shift1", "AR_mi", "AR_tr"],

        [  # Corresponding values
            P_sa, Q_bp, Q_hp, Q_rmp, Q_amp, Q_sp, Q_ep,
            Q_pp, Q_la, Q_lv, Q_ra, Q_rv, P_ra, P_la, P_lv, P_rv,
            P_la, P_ra, VT_rv, VT_ra,
            VT_lv, VT_la, P_pa, P_pp, P_pv, P_vc, Qi_lv,
            Qi_rv, phi, phi_atr, P_amv, P_ev, V_u,
            P_sp, Q_sa, Q_vc, VT_amv,
            Q_amv, Q_pa, V_sa, P_bv, R_bv,
            Q_ev, Q_bv, Q_hv, Q_rmv, Q_sv, VT_pa, VT_pp, VT_pv, VT_sv, VT_bv, VT_hv, VT_rmv,
            VT_vc, t, theta_ao, theta_po, theta_mi, theta_tr, V_shift1, AR_mi, AR_tr])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value


    keys_and_values = zip(
        [  # Cardio inputs
            "HR", "Vu_ev", "Vu_sv", "Vu_rmv", "Vu_amv", "Emax_lv", "Emax_rv",
            "R_ep", "R_amp", "R_rmp", "R_sp", "f_sp", "f_sh", "f_v", "f_sv", "f_ab",
            "Tv_change", "Ts_change", "HR_check", "f_sh_delay2", "f_v_delay02", "sigma_Ts",
            "sigma_Tv", "P_n_current"
        ],

        [  # Corresponding values
            HR, Vu_ev, Vu_sv, Vu_rmv, Vu_amv, Emax_lv, Emax_rv,
            R_ep, R_amp_n, R_rmp_n, R_sp, f_sp, f_sh, f_v, f_sv, f_ab, Tv_change,
            Ts_change, HR_every,
            f_sh_delay2_Emax_rv, f_v_delay0_2, sigma_Ts, sigma_Tv, P_n_current])

    for key, value in keys_and_values:
        updates[key][updates["j"].item() - num_removed] = value


    return [  # cardio derivatives
        dVT_pa_dt, dVT_pp_dt, dVT_pv_dt, dQ_pa_dt, dVT_la_dt, dVT_lv_dt, dVT_ra_dt, dVT_rv_dt, dVT_sv_dt,
        dVT_bv_dt, dVT_hv_dt, dVT_rmv_dt, dVT_amv_dt, dP_sp_dt, dP_sa_dt, dQ_sa_dt, dVT_vc_dt,
        dtheta_ao_dt, d2theta_ao_dt2, dtheta_po_dt, d2theta_po_dt2, dtheta_mi_dt, d2theta_mi_dt2, dtheta_tr_dt,
        d2theta_tr_dt2,

        # cardio controller derivatives
        dP_tilda_dt, dR_ep_change_dt,
        dR_sp_change_dt, dR_rmp_n_change_dt, dR_amp_n_change_dt, dVu_ev_change_dt, dVu_sv_change_dt,
        dVu_rmv_change_dt, dVu_amv_change_dt, dEmax_lv_change_dt, dEmax_rv_change_dt, d_Ts_change_dt,
        d_Tv_change_dt, dP_n_current_dt]