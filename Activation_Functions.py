import math
import numpy as np
from numba import njit


def frac(x):
    return x - math.floor(x)

def activation_U(beta, atr, T, Tsys):
    if atr == 1:
        U_t0 = 0.1
    else:
        U_t0 = 0

    U1 = frac(beta + U_t0)

    if 0 <= U1 <= (Tsys / T):
        phi = (np.sin(((np.pi * T) / Tsys) * U1)) ** 2
    else:
        phi = 0

    return phi



# def activation_H(ti, atr, T):
#     # rise and decrease
#     tr_atr = 0.05*T
#     td_atr = 0.1*T
#
#     tr_ven = 0.15 * T
#     td_ven = 0.3 * T
#
#     if ti <= 0.9 * T:
#         t_la = ti + 0.1 * T
#     else:
#         t_la = ti - 0.9 * T
#
#
#     if atr == 1:
#         phi = np.where(t_la <= tr_atr,
#                            0.5 * (1.0 - np.cos(np.pi * t_la / tr_atr)),
#                        np.where(t_la <= td_atr,
#                                 0.5 * (1.0 + np.cos(np.pi * (t_la - tr_atr) / (td_atr - tr_atr))),
#                                 0))
#
#     else:
#         phi = np.where(ti <= tr_ven,
#                        0.5 * (1.0 - np.cos(np.pi * ti / tr_ven)),
#                        np.where(ti <= td_ven,
#                                 0.5 * (1.0 + np.cos(np.pi * (ti - tr_ven) / (td_ven - tr_ven))),
#                                 0))
#
#     return phi

@njit
def activation_F(t, atr, T):
    T_total_ven = 0.3 * T
    T_rise_ven = 0.1 * T
    T_fall_ven = 0.01 * T
    T_period = T

    T_total_atr = 0.1 * T
    T_rise_atr = 0.03 * T
    T_fall_atr = 0.03 * T

    # T_total_ven = 0.05 * T
    # T_rise_ven = 0.00001 * T
    # T_fall_ven = 0.01 * T
    # T_period = T
    #
    # T_total_atr = 0.05 * T
    # T_rise_atr = 0.001 * T
    # T_fall_atr = 0.00001 * T

    ti = t % T_period

    # Ventricular contraction force
    if atr == 0:
        if ti <= T_rise_ven:
            force = (ti / T_rise_ven)
        elif T_rise_ven < ti <= T_total_ven - T_fall_ven:
            force = 1.0
        elif T_total_ven - T_fall_ven < ti <= T_total_ven:
            force = 1.0 - ((ti - (T_total_ven - T_fall_ven)) / T_fall_ven)
        else:
            force = 0.0
    # Atrial contraction force (negative force)
    else:
        # Shift the atrial contraction to the end of the cycle
        t_atr_shifted = ti - (T_period - T_total_atr)
        if t_atr_shifted > 0:
            if t_atr_shifted <= T_rise_atr:
                force = -(t_atr_shifted / T_rise_atr)
            elif T_rise_atr < t_atr_shifted <= T_total_atr - T_fall_atr:
                force = -1.0
            elif T_total_atr - T_fall_atr < t_atr_shifted <= T_total_atr:
                force = -1.0 + ((t_atr_shifted - (T_total_atr - T_fall_atr)) / T_fall_atr)
            else:
                force = 0.0
        else:
            force = 0.0

    return force


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

# @njit
# def activation_H(ti, atr, T, rise_time_atr, fall_time_atr, rise_time_ven, fall_time_ven, ahead1):
#     tr_atr = rise_time_atr * T
#     td_atr = fall_time_atr * T
#     tr_ven = rise_time_ven * T
#     td_ven = fall_time_ven * T
#
#     if ti <= ahead1 * T:
#         t_la = ti + (1-ahead1) * T
#     else:
#         t_la = ti - ahead1 * T
#
#     if atr == 1:
#         # --- first peak (existing atrial one) ---
#         if t_la <= tr_atr:
#             phi1 = 0.5 * (1.0 - np.cos(np.pi * (t_la / tr_atr)**1))
#         elif t_la <= td_atr:
#             phi1 = 0.5 * (1.0 + np.cos(np.pi * (t_la - tr_atr) / (td_atr - tr_atr)))
#         else:
#             phi1 = 0.0
#
#         # --- second peak (same shape as ventricular) ---
#         if ti <= tr_ven:
#             phi2 = 0.5 * (1.0 - np.cos(np.pi * ti / tr_ven))
#         elif ti <= td_ven:
#             phi2 = 0.5 * (1.0 + np.cos(np.pi * (ti - tr_ven) / (td_ven - tr_ven)))
#         else:
#             phi2 = 0.0
#
#         def smooth_max(a, b, eps=0.02):
#             # eps controls smoothness — smaller = sharper but still smooth
#             return a + eps * np.log1p(np.exp((b - a) / eps))
#
#         # return phi1 + 0.2 * phi2   # two peaks added
#         return smooth_max(phi1, 0.2 * phi2)
#     else:
#         if ti <= tr_ven:
#             return 0.5 * (1.0 - np.cos(np.pi * ti / tr_ven))
#         elif ti <= td_ven:
#             return 0.5 * (1.0 + np.cos(np.pi * (ti - tr_ven) / (td_ven - tr_ven)))
#         else:
#             return 0.0


def activation_S(t, atr, T):
    # rise and decrease
    tr_atr = 0.05*T
    td_atr = 0.1*T

    tr_ven = 0.15 * T
    td_ven = 0.3 * T

    ti = t % T

    if ti <= 0.9 * T:
        t_la = ti + 0.1 * T
    else:
        t_la = ti - 0.9 * T

    if atr == 1:
        phi1 = np.where(t_la <= tr_atr,
                        0.5 * (1.0 - np.cos(np.pi * t_la / tr_atr)),
                        0)

        phi2 = np.where(np.logical_and(t_la > tr_atr, t_la <= td_atr),
                        0.5 * (1.0 + np.cos(np.pi * (t_la - tr_atr) / (td_atr - tr_atr))),
                        0)

    return phi1, phi2




def activation_conduit(t, T):
    # rise and decrease

    td_ven = 0.3 * T
    t_end = 0.9 * T
    rise_end = 0.5 * T

    ti = t % T

    phi_cond = np.where(
        ti < td_ven,
        0,
        np.where(
            ti <= rise_end,
            0.5 * (1 - np.cos(np.pi * (ti - td_ven) / (rise_end - td_ven))),
            np.where(
                ti <= t_end,
                0.5 * (1 + np.cos(np.pi * (ti - rise_end) / (t_end - rise_end))),
                0
            )
        )
    )


    # phi_cond = np.where(np.logical_and(td_ven - 0.06 <= ti, ti <= t_end),
    #                     0.5 * (1.0 - np.cos(np.pi * (ti - (td_ven - 0.06)) / (t_end - (td_ven - 0.06)))),
    #                     0)

    # phi_cond = np.where(
    #     ti < td_ven,
    #     0,
    #     np.where(np.logical_and(td_ven <= ti, ti <= rise_end),
    #                     0.5 * (1.0 - np.cos(np.pi * (ti - (td_ven)) / (rise_end - (td_ven)))),
    #                     np.where(ti <= t_end,
    #                              1,
    #                              0)))

    # phi_cond = np.where(
    #     ti < td_ven,
    #     0,
    #     np.where(np.logical_and(td_ven <= ti, ti <= rise_end),
    #              1,
    #              np.where(ti <= t_end,
    #                       1,
    #                       0)))

    # phi_cond = np.where(
    #     np.logical_and(td_ven - 0.06 <= ti, ti <= t_end),
    #     0.5 * (1.0 - np.cos(np.pi * (ti - td_ven) / (rise_end - td_ven))),
    #         0
    #     )
    # phi_cond = 2*phi_cond

    # phi_cond = np.where(np.logical_and(td_ven - 0.06 <= ti, ti <= t_end), 1, 0)

    return phi_cond


def activation_H_derivative(t, atr, T):
    # rise and decrease
    tr_atr = 0.045*T
    td_atr = 0.09*T

    tr_ven = 0.15 * T
    td_ven = 0.3 * T

    ti = t % T

    if ti <= 0.9 * T:
        t_la = ti + 0.1 * T
    else:
        t_la = ti - 0.9 * T

    if atr == 1:
        dphi_dt = np.where(t_la <= tr_atr,
                           0.5 * (np.sin(np.pi * t_la / tr_atr)) * np.pi / tr_atr,
                       np.where(t_la <= td_atr,
                                -0.5 * (np.sin(np.pi * (t_la - tr_atr) / (td_atr - tr_atr))) * (np.pi/(td_atr - tr_atr)),
                                0))
    else:
        dphi_dt = np.where(ti <= tr_ven,
                       0.5 * (np.sin(np.pi * ti / tr_ven)) * np.pi / tr_ven,
                       np.where(ti <= td_ven,
                                -0.5 * (np.sin(np.pi * (ti - tr_ven) / (td_ven - tr_ven))) * (np.pi/(td_ven - tr_ven)),
                                0))

    return dphi_dt





def activation_Naghavi(t, atr, T, Tsys):
    tr = Tsys

    ti = t % T

    if ti <= 0.9 * T:
        t_la = ti + 0.1 * T
    else:
        t_la = ti - 0.9 * T

    if atr == 1:
        phi = np.where(t_la <= tr,
                       0.5 * (1.0 - np.cos(np.pi * t_la / T)),
                       np.where(t_la <= T,
                                0.5 * (np.exp(-(t_la - tr) * (1 / 0.025))),
                                0))
    else:
        phi = np.where(ti <= tr,
                   0.5 * (1.0 - np.cos(np.pi * ti / T)),
                   np.where(ti <= T,
                            0.5 * (np.exp(-(ti - tr) * (1 / 0.025))),
                            0))

    return phi


def g_function(t, atr, T):
    tmax = T
    ti = t % T

    if ti <= 0.9 * T:
        t_la = ti + 0.1 * T
    else:
        t_la = ti - 0.9 * T

    if atr == 1:
        if t_la < 0:
            phi = 0
        elif 0 <= t_la < tmax:
            phi = np.sin(np.pi * t_la / tmax) ** 2
        else:
            phi = 0

    else:
        if ti < 0:
            phi = 0
        elif 0 <= ti < tmax:
            phi = np.sin(np.pi * ti / tmax) ** 2
        else:
            phi = 0

    return phi