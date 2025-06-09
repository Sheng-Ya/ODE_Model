import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Possible variables to plot:

time_history	VT_pa	VT_pp	VT_pv	Q_pa	VT_la	VT_lv	VT_ra	VT_rv	VT_sv	VT_bv	VT_hv	VT_rmv	VT_amv
VT_ev	P_sp	V_sa	P_sa	Q_sa	VT_vc	Qi_lv	Qi_rv	Q_vc	Q_amv	P_pa	P_pp	P_pv	P_ra	P_la	
P_lv	P_rv	Pmax_lv	Q_lv	Q_ra	Q_rv	P_thor	P_vc	phi	phi_atr	P_bv	BF	Vu_ev	Vu_amv	Vu_rmv	Vu_sv	
R_ep	R_amp	R_rmp	R_sp	R_bp	R_hp	HR	Emax_lv	Emax_rv	I	P_ev	V_u	R_bv	P_amv	Ca_O2	Ca_CO2	Q_ev
Q_hp	Q_rmp	Q_amp	VT	TI	VE_flow	V	dV_dt	P_musc	Pa_O2	Pa_CO2	MRTCO2	Cv_O2	Cv_CO2	Q_pp	Q_bp	Q_la	
VD	Pmax_rv	Pmax_ra	Pmax_la	Pb_CO2	VAflow	PvtCO2

"""

filename = 'C:/Users/vanes/Documents/Next_Conditions_Output.parquet'

# Replace with the column names you want to plot
y_columns = ["P_rv"]

df = pd.read_parquet(filename)

# First column is time
time_col = df.columns[0]
time = df[time_col]

# Extract datapoints for any variable
rv_pressures = np.array(df["P_rv"])

# Plot variables
plt.figure(figsize=(10, 5))
for col in y_columns:
    plt.plot(time, df[col], label=col)


# Exercise simulation between t = 2000 s and t =2500 s
plt.axvline(x=2000, color='red', linestyle='--', linewidth=2, label='Start Exercise')
plt.axvline(x=2500, color='blue', linestyle='--', linewidth=2, label='End Exercise')


plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.title("Right Ventricle Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()