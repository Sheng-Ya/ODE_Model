import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint

from Next_Conditions import Next_Conditions
from Parameters import Parameters as params
from Resp_Control_Breath_Optimiser import objective

data = np.load("t1_t2_vs_VAflow.npz")
VAflow = data["VAflow"]
t1 = data["t1"]
t2 = data["t2"]



# Define VAflow range
VAflow_vals = np.linspace(0.06, 0.4, 1000)

# Coefficients for t1 and t2 best-fit polynomials (highest degree to constant)
coeffs_t1 = [6328, -9812, 6219, -2081, 397.1, -44.14, 3.544]
coeffs_t2 = [7645, -12330, 8109, -2804, 550.2, -62.43, 4.724]

# Define polynomial functions
t1_poly = np.poly1d(coeffs_t1)
t2_poly = np.poly1d(coeffs_t2)

# Evaluate the fitted curves
t1_fitted = t1_poly(VAflow_vals)
t2_fitted = t2_poly(VAflow_vals)

# Plot both best-fit curves
plt.figure(figsize=(12, 5))
plt.scatter(VAflow, t1, color='blue', s=10, label='t1 (Inspiration)')
plt.scatter(VAflow, t2, color='red', s=10, label='t2 (Expiration)')
plt.plot(VAflow_vals, t1_fitted, 'b-', label='Best Fit for t1 (Inspiration)')
plt.plot(VAflow_vals, t2_fitted, 'r--', label='Best Fit for t2 (Expiration)')
plt.xlabel('VAflow (L/s)')
plt.ylabel('Time (s)')
plt.title('Best Fit Curves for t1 and t2 vs VAflow')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# repeats = 5
# VAflow_unique = VAflow[::repeats]  # Grab every 5th (first of each group)
# t1_mean = np.mean(t1.reshape(-1, repeats), axis=1)
# t2_mean = np.mean(t2.reshape(-1, repeats), axis=1)
#
# # Fit a polynomial (or linear)
# t1_poly = np.poly1d(np.polyfit(VAflow_unique, t1_mean, deg=6))
# t2_poly = np.poly1d(np.polyfit(VAflow_unique, t2_mean, deg=6))
#
# VAflow_fit = np.linspace(min(VAflow_unique), max(VAflow_unique), 1000)
#
# print("Best fit equation for t1:", t1_poly)
# print("Best fit equation for t2:", t2_poly)
#
# plt.figure(figsize=(14, 6))
# plt.plot(VAflow_unique, t1_mean, 'bo', markersize=3, label='Mean t1')
# plt.plot(VAflow_fit, t1_poly(VAflow_fit), 'b-', linewidth=2, label='Fit t1')
# plt.plot(VAflow_unique, t2_mean, 'ro', markersize=3, label='Mean t2')
# plt.plot(VAflow_fit, t2_poly(VAflow_fit), 'r-', linewidth=2, label='Fit t2')
# plt.xlabel("VAflow (L/s)")
# plt.ylabel("Time (s)")
# plt.title("Average t1 and t2 vs VAflow with Best-Fit Curves")
# plt.legend()
# plt.grid(True)
# plt.show()



coeffs_t1 = np.polyfit(VAflow, t1, deg=6)
coeffs_t2 = np.polyfit(VAflow, t2, deg=6)

fit_t1 = np.poly1d(coeffs_t1)
fit_t2 = np.poly1d(coeffs_t2)

print("Best fit equation for t1:", fit_t1)
print("Best fit equation for t2:", fit_t2)

plt.figure(figsize=(16, 6))

# Scatter plot
plt.scatter(VAflow, t1, color='blue', s=10, label='t1 (Inspiration)')
plt.scatter(VAflow, t2, color='red', s=10, label='t2 (Expiration)')

# Fitted lines
# VAflow_sorted = np.linspace(min(VAflow), max(VAflow), 1000)
plt.plot(VAflow, fit_t1(VAflow), color='navy', linewidth=2, label='Best Fit t1')
plt.plot(VAflow, fit_t2(VAflow), color='darkred', linewidth=2, label='Best Fit t2')

plt.xlabel("VAflow (L/s)")
plt.ylabel("Time (s)")
plt.title("Optimal t1 and t2 with a Fitted 6 Degree Polynomial")
plt.legend()
plt.grid(True)
plt.show()




GV_dead = params["GV_dead"]
V0_dead = params["V0_dead"]
dt = 0.001

# VAflow = 0.086667
# VD = GV_dead * VAflow + V0_dead


bounds = [(0.4, 3), (0.4, 6)]  # [t1, t2]
tolerance = 0.001

VAflow_vals = np.linspace(0.06, 0.4, 1000)
VAflow_repeated = np.repeat(VAflow_vals, 5)

VD = GV_dead * VAflow_repeated + V0_dead

optimal_t1 = []
optimal_t2 = []
failed_indices = []

for idx, VAflow in enumerate(VAflow_repeated):
    VD_volume = VD[idx]
    required_params = [params["lambda1"], params["lambda2"], params["n"], params["Pmax"], params["Pmax_dot"], params["E_rs"], params["R_rs"], params["P_ao"]]

    try:
        res = minimize(objective, x0=Next_Conditions["Nd"][-2:], args=(required_params, VAflow, VD_volume, dt, tolerance), method='COBYLA', bounds=bounds)
        # res = minimize(objective, x0=Next_Conditions["Nd"][-2:], method='COBYLA', bounds=bounds)
        if res.success:
            t1_opt, t2_opt = res.x
            optimal_t1.append(t1_opt)
            optimal_t2.append(t2_opt)
            Next_Conditions["Nd"].extend(res.x)
            print(f"VAflow = {VAflow:.4f} → optimal t1 = {t1_opt:.4f}, t2 = {t2_opt:.4f}")
        else:
            print(f"VAflow = {VAflow:.4f} → optimization failed")
            optimal_t1.append(np.nan)
            optimal_t2.append(np.nan)
            failed_indices.append(idx)
    except Exception as e:
        print(f"VAflow = {VAflow:.4f} → exception: {e}")
        optimal_t1.append(np.nan)
        optimal_t2.append(np.nan)
        failed_indices.append(idx)


# Convert to arrays for indexing
VAflow_clean = np.array(VAflow_repeated)
VD_clean = np.array(VD)
t1_clean = np.array(optimal_t1)
t2_clean = np.array(optimal_t2)

# Mask out NaNs
valid_mask = ~np.isnan(t1_clean) & ~np.isnan(t2_clean)
VAflow_clean = VAflow_clean[valid_mask]
VD_clean = VD_clean[valid_mask]
t1_clean = t1_clean[valid_mask]
t2_clean = t2_clean[valid_mask]

VEflow = 60 * (VAflow_clean + VD_clean / (t1_clean + t2_clean))


# plt.figure(figsize=(10, 5))
# plt.scatter(VEflow, t1_clean, label='Optimal t1 (Inspiration Time)', color='blue', alpha=0.6)
# plt.scatter(VEflow, t2_clean, label='Optimal t2 (Expiration Time)', color='red', alpha=0.6)
# plt.xlabel('VE_Flow (L/s)')
# plt.ylabel('Time (s)')
# plt.title('Optimal t1 and t2 vs VEflow Using COBYLA')
# plt.legend()
# plt.grid(True)
# plt.show()
np.savez("t1_t2_vs_VAflow.npz", VAflow=VAflow_clean, t1=t1_clean, t2=t2_clean)

plt.figure(figsize=(10, 5))
plt.scatter(VAflow_clean, t1_clean, label='Optimal t1 (Inspiration Time)', color='blue', alpha=0.6)
plt.scatter(VAflow_clean, t2_clean, label='Optimal t2 (Expiration Time)', color='red', alpha=0.6)
plt.xlabel('VAflow (L/s)')
plt.ylabel('Time (s)')
plt.title('Optimal t1 and t2 vs VAflow Using COBYLA')
plt.legend()
plt.grid(True)
plt.show()




# dt = 0.001
# bounds = [(1, 3), (1.5, 6)]  # [t1, t2] bounds
# tolerance = 0.001
# opt = BreathOptimiser(params, VAflow, VD, dt, tolerance)
#
# t1_vals = np.linspace(1, 3, 30)
# t2_vals = np.linspace(1.5, 6, 30)
#
#
# T1, T2 = np.meshgrid(t1_vals, t2_vals)
# Z_inspire = np.zeros_like(T1)
# Z_expire = np.zeros_like(T1)
#
# for i in range(T1.shape[0]):
#     for j in range(T1.shape[1]):
#         t1 = T1[i, j]
#         t2 = T2[i, j]
#         try:
#             I_inspire, I_expire = opt.compute_integral_inspire(t1, t2)
#             Z_inspire[i, j] = I_inspire
#             Z_expire[i, j] = I_expire
#         except Exception:
#             Z_inspire[i, j] = np.nan
#             Z_expire[i, j] = np.nan
#
# # Plot side-by-side
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))
#
# cs1 = axs[0].contourf(T1, T2, Z_inspire, levels=50, cmap='viridis')
# fig.colorbar(cs1, ax=axs[0], label='Integral Inspire')
# axs[0].set_title('Inspiratory Work')
# axs[0].set_xlabel('t1 (Inspiration time)')
# axs[0].set_ylabel('t2 (Expiration time)')
#
# cs2 = axs[1].contourf(T1, T2, Z_expire, levels=50, cmap='plasma')
# fig.colorbar(cs2, ax=axs[1], label='Integral Expire')
# axs[1].set_title('Expiratory Work')
# axs[1].set_xlabel('t1 (Inspiration time)')
# axs[1].set_ylabel('t2 (Expiration time)')
#
# plt.tight_layout()
# plt.show()





# T1, T2 = np.meshgrid(t1_vals, t2_vals)
# Z = np.zeros_like(T1)
#
# for i in range(T1.shape[0]):
#     for j in range(T1.shape[1]):
#         t1 = T1[i, j]
#         t2 = T2[i, j]
#         try:
#             Z[i, j] = opt.objective([t1, t2])
#         except Exception as e:
#             Z[i, j] = np.nan  # mark failed evaluations
#
# # Plot
# plt.figure(figsize=(10, 6))
# cp = plt.contourf(T1, T2, Z, levels=50, cmap='viridis')
# plt.colorbar(cp, label='Work (J)')
# plt.xlabel('t1 (Inspiration time)')
# plt.ylabel('t2 (Expiration time)')
# plt.title('Work')
# plt.show()



# Set up lambda1 and lambda2 sweep ranges
lambda1_vals = np.linspace(0, 2, 30)
lambda2_vals = np.linspace(0, 2, 30)
L1, L2 = np.meshgrid(lambda1_vals, lambda2_vals)
Z = np.zeros_like(L1)

for i in range(L1.shape[0]):
    for j in range(L1.shape[1]):

        lambda1 = L1[i, j]
        lambda2 = L2[i, j]

        if lambda2 > 0.568:
            A = 2

        # Update parameters
        params["lambda1"] = lambda1
        params["lambda2"] = lambda2
        opt = BreathOptimiser(params, VAflow, VD, dt, tolerance)

        try:
            res = minimize(opt.objective, x0= [2, 5], method='COBYLA', bounds=bounds)
            if res.success:
                Z[i, j] = res.x[0]
            else:
                Z[i, j] = np.nan
        except Exception:
            Z[i, j] = np.nan

        print(f"Processing λ1={lambda1:.2f}, λ2={lambda2:.2f} → t2 = {Z[i, j]:.4f}")

# Plot heatmap of minimum work vs lambda1, lambda2
plt.figure(figsize=(10, 6))
cp = plt.contourf(L1, L2, Z, levels=50, cmap='viridis')
plt.colorbar(cp, label='Inspiration time t1')
plt.xlabel('lambda1 (smoothness penalty)')
plt.ylabel('lambda2 (expiration penalty weight)')
plt.title('Inspiration time t1 across λ1 and λ2 (optimized over t1, t2)')
plt.tight_layout()
plt.show()

