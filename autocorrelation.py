# # # Parameters
# # fs = 10  # sampling frequency (samples per second)
# # T = 300  # total duration in seconds
# # period = 30  # true period of the HR in seconds
#
# # # Time vector
# # t = np.arange(0, T, 1/fs)
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load data
data = np.load("HR_vs_time_800s.npz")
index = np.where(data["time"] == 1e6)[0][0] - 1
time_history = data["time"][4000000:index] - data["time"][4000000]
HR = data["HR"][4000000:index]
HR_averaged = data["HR_average"][4000000:index]



# Full signal
plt.plot(time_history, HR_averaged, label="Original HR", color='blue')

plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (bpm)")
plt.legend()
plt.grid(True)
plt.show()

# # no need to reverse
# HR_rev = HR[::-1]
# time_rev = time_history[-1] - time_history[::-1]
# # Full signal plot
# plt.plot(time_rev, HR_rev, label="Full HR", color='blue')

# wrong shift, comparing front of signal with signal after x seconds
# # Overlay: HR after 2.2 seconds
# mask = time_history > 326.3
# time_shifted = time_history[mask] - 326.3
# HR_shifted = HR[mask]
#
# # Plot shifted overlay
# plt.plot(time_shifted, HR_shifted, label="HR after 2.2 s (shifted)", color='red', linestyle='--')



# Step 1: Resample HR onto a uniform time grid
fs = 10  # sampling frequency in Hz (1 sample/sec)
t_uniform = np.arange(time_history[0], time_history[-1], 1/fs)

# Step-wise interpolation
interp_func = interp1d(time_history, HR, kind='previous', fill_value="extrapolate")
HR_uniform = interp_func(t_uniform)

# Step 2: Mean center
HR_centered = HR_uniform - np.mean(HR_uniform)

n = len(HR_centered)
lags = np.arange(n)

# Step 3: Autocorrelation
autocorr_unbiased = np.correlate(HR_centered, HR_centered, mode='full')
autocorr_unbiased = autocorr_unbiased[n - 1:]  # Keep non-negative lags
autocorr_unbiased /= (n - lags)  # Unbias: divide by # of overlapping samples
autocorr_unbiased /= autocorr_unbiased[0]  # Normalize to 1 at lag 0

# Step 4: Lag axis in seconds
lags_sec = np.arange(len(autocorr_unbiased)) / fs


# Step 5: Plot
plt.figure(figsize=(16, 6))
plt.plot(lags_sec, autocorr_unbiased)
plt.xlabel('Lag (seconds)')
plt.ylabel('Normalized Autocorrelation')
plt.title('Autocorrelation of Stepwise HR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Full signal
plt.plot(time_history, HR, label="Original HR", color='blue')
# Shifted signal (lagged version of original)
plt.plot(time_history + 165, HR, label="HR lagged by 165 s", color='red', linestyle='--', alpha=0.6)

plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (bpm)")
plt.legend()
plt.title("HR and Lagged HR (165 s)")
plt.grid(True)
plt.show()


# fig, ax1 = plt.subplots()
# plt.plot(heartbeat_times, heartbeat_HR, label="HR")
#
# # Add labels and legend
# plt.ylabel("")
# plt.xlabel("Time (s)")
# plt.title("Traces")
# plt.legend()
# plt.grid(True)
# plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# fs = 2 * np.pi  # sampling frequency (samples per second)
# T = 500  # total duration in seconds
# period = 2 * np.pi  # true period of the signal in seconds
#
# # Time vector
# t = np.arange(0, T, 1/fs)
#
# # Generate a periodic signal (sine wave)
# signal = np.sin(2 * np.pi * t / period)
#
# # Mean subtraction
# signal_centered = signal - np.mean(signal)
#
#
# # plt.figure(figsize=(10, 5))
# # plt.plot(t, signal_centered)
# # plt.show()
#
# # Max lag in seconds and samples
# max_lag_sec = 5*np.pi
# max_lag_samples = int(max_lag_sec * fs)
#
# # Compute autocorrelation (biased estimator)
# n = len(signal_centered)
# lags = np.arange(n)
# autocorr_unbiased = np.correlate(signal_centered, signal_centered, mode='full')
# autocorr_unbiased = autocorr_unbiased[n - 1:]  # Keep non-negative lags
# autocorr_unbiased /= (n - lags)  # Unbias: divide by # of overlapping samples
# autocorr_unbiased /= autocorr_unbiased[0]  # Normalize to 1 at lag 0
#
# # Prepare lag time axis
# lags = np.arange(len(autocorr_unbiased)) / fs
#
# # Plot autocorrelation up to max lag
# plt.figure(figsize=(10, 5))
# plt.plot(lags[:max_lag_samples], autocorr_unbiased[:max_lag_samples], label='Autocorrelation')
#
# # plt.axvline(period, color='r', linestyle='--', label='True period (30 s)')
# plt.xlabel('Lag (seconds)')
# plt.ylabel('Normalized Autocorrelation')
# plt.title('Autocorrelation of Periodic Signal')
# plt.legend()
# plt.grid(True)
# plt.show()