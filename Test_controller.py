import numpy as np

def source(t, t_on=100., t_off=110., LBNPswitch=1):
    LBNP = -60. * LBNPswitch
    grad = LBNP / (t_off - t_on)
    return (np.heaviside(t - t_on, 1) - np.heaviside(t - t_off, 1)) * grad