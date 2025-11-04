# Define all parameter lists
heart_rate = ['T0','GT_s','GT_v','fev_o','Fi_O2','AT','V_tot','Yv_max','Io_sh','R_rs',
              'E_rs','Wp_v','G_ap','P_n_max','Ysh_max']

rest = ['T0','V_tot','P_n','fev_o','GT_v','GT_s','C2','C_O2_param1','Fi_O2',
        'Vu_sv0','fes_o','fab_o','kes','Wb_sh','K2','k_ab','f_acCO2_n']

systolic_pressure_1 = ['V_tot','Vu_sv0','GV_sv','R_rs','G_ap','R_sa','fes_o','P_n','Fi_O2',
                       'E_rs','fab_o','C_pv','rise_time_ven','GT_v','Vu_ev0','f_acCO2_n',
                       'C_sv','Vu_jp','fall_time_ven','T0','Wc_v','C_O2_param1','Kv_mi',
                       'k_ab','V0_dead','C_pp','Kp_mi','GV_dead','Wb_sh','fev_inf','Kv_tr',
                       'fev_o','Wp_v','Ysh_max','PaO2_ac_n','kev','theta_v','AT','tauMR',
                       'VA_rest','P_n_max','GT_s','R_pv','f_ab_max','k_ac','GR_amp',
                       'f_ac_max','Yv_max','Io_met','theta_mi_max','KE_lv','Kp_tr',
                       'Io_sh','MO2_bp','KcCO2','Tc','Vu_amv0','theta_tr_max','phi_max',
                       'Vu_bv','kes','PaCO2_n','f_ac_min']
# C2, a2, K2, fes_min, Cvrm_O2_n
systolic_pressure_2 = ['V_tot','Vu_sv0','P_n','C2','PaCO2_n','kes','a2','V0_dead','fes_o',
                       'R_rs','E_rs','GV_dead','Vu_ev0','K2','Vu_jp','C_pv','fes_min',
                       'R_pv','R_sa','Fi_O2','Cvrm_O2_n','C_O2_param1','fab_o',
                       'rise_time_ven','fall_time_ven','GV_sv','C_pp','Kv_tr']
# KcMRV
minute_ventilation_1 = ['R_rs','E_rs','GV_dead','V0_dead','PaCO2_n','VA_rest','KcCO2',
                        'V_tot','C_O2_param1','C2','MO2_bp','KcMRV']

minute_ventilation_2 = ['R_rs','PaCO2_n','E_rs','C2','V0_dead','GV_dead','V_tot']

# Combine all lists and take the union
all_params = (heart_rate + rest + systolic_pressure_1 + systolic_pressure_2 +
              minute_ventilation_1 + minute_ventilation_2)

unique_params = sorted(set(all_params))  # remove duplicates and sort alphabetically

# Print results
print(f"Unique parameters ({len(unique_params)} total):")
print(unique_params)
