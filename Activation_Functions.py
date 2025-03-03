import math

import numpy as np

from Next_Conditions import Next_Conditions
from Parameters import Parameters

Tsys_0 = Parameters["Tsys_0"]
ksys = Parameters["ksys"]
T = 1/Next_Conditions["HR"][-1]
Tsys = Tsys_0 - ksys * (1/T)

def frac(x):
    return x - math.floor(x)

def activation_U(beta, atr):
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



def activation_H(t, atr):
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
        phi = np.where(t_la <= tr_atr,
                           0.5 * (1.0 - np.cos(np.pi * t_la / tr_atr)),
                       np.where(t_la <= td_atr,
                                0.5 * (1.0 + np.cos(np.pi * (t_la - tr_atr) / (td_atr - tr_atr))),
                                0))
    else:
        phi = np.where(ti <= tr_ven,
                       0.5 * (1.0 - np.cos(np.pi * ti / tr_ven)),
                       np.where(ti <= td_ven,
                                0.5 * (1.0 + np.cos(np.pi * (ti - tr_ven) / (td_ven - tr_ven))),
                                0))

    return phi




def activation_Naghavi(t, atr):
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


def g_function(t, atr):
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