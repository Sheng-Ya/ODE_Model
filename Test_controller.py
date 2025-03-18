import numpy as np

def source(t, t_on=100., t_off=110., LBNPswitch=1):
    LBNP = -20. * LBNPswitch
    grad = LBNP / (t_off - t_on)
    return (np.heaviside(t - t_on, 1) - np.heaviside(t - t_off, 1)) * grad



def source1(t, t_start=100., t_2=110., t_3=120., t_4=130., Pmax=-20):
    LBNPgrad = ((Pmax/(2*(t_2-t_start)) * np.pi*(np.sin(np.pi*((t-t_start)/(t_2-t_start)))))*(t > t_start)*(t <= t_2) +
                (-Pmax/(2*(t_4-t_3)) * np.pi*(np.sin(np.pi*((t-t_3)/(t_4-t_3)))))*(t > t_3)*(t <= t_4))
    return LBNPgrad

def source2(t, t_start=100., t_2=110., Pmax=-20):
    LBNPgrad = ((Pmax/(2*(t_2-t_start)) * np.pi*(np.sin(np.pi*((t-t_start)/(t_2-t_start)))))*(t > t_start)*(t <= t_2))
    return LBNPgrad


# def source3(t, t_start=100., t_2=110., t_3=120., t_4=130., Pmax=20):
#     if t_start <= t <= t_2:
#         # Smooth ramp-up using sine function
#         LBNPgrad = -Pmax * 0.5 * (1 - np.cos(np.pi * (t - t_start) / (t_2 - t_start)))
#     elif t_2 < t <= t_3:
#         # Hold steady at max negative pressure
#         LBNPgrad = -Pmax
#     elif t_3 < t <= t_4:
#         # Smooth ramp-down using sine function
#         LBNPgrad = -Pmax * 0.5 * (1 + np.cos(np.pi * (t - t_3) / (t_4 - t_3)))
#     else:
#         # No pressure change outside intervals
#         LBNPgrad = 0
#     return LBNPgrad