import matplotlib.pyplot as plt
import pandas as pd

# # Filepath to your data
# filename = 'C:/Users/vanes/Desktop/t1_t2.txt'
#
# t1 = []
# t2 = []
# times = []
#
# with open(filename, 'r') as f:
#     lines = [line.strip() for line in f if line.strip()]
#
# current_guess_block = []
#
# i = 0
# while i < len(lines):
#     line = lines[i]
#     if line.startswith('guess:'):
#         current_guess_block.append(line)
#     else:
#         # If we reach a non-guess line and have guesses buffered,
#         # take the last guess and the current line as the time
#         if current_guess_block:
#             last_guess = current_guess_block[-1]
#             values = eval(last_guess.split('guess:')[1])
#             t1.append(values[-2])
#             t2.append(values[-1])
#             try:
#                 time_val = float(line)
#                 times.append(time_val)
#             except ValueError:
#                 times.append(float('nan'))
#             current_guess_block = []
#     i += 1
#
# # Plotting
# plt.plot(times, t1, label='t1 (inspiration time)', marker='o')
# plt.plot(times, t2, label='t2 (expiration time)', marker='x')
# plt.xlabel('Simulation Time (s)')
# plt.ylabel('Inspiration/Expiration Time (s)')
# plt.title('t1 and t2 with time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# === Parameters to set ===
filename = 'C:/Users/vanes/Documents/Next_Conditions_Output.parquet'
y_columns = ["Q_rv"]     # Replace with the column names you want to plot

# === Load data ===
df = pd.read_parquet(filename)

# Assume first column is time
time_col = df.columns[0]
time = df[time_col]

# === Plot specified columns ===
plt.figure(figsize=(10, 5))
for col in y_columns:
    if col in df.columns:
        plt.plot(time, df[col], label=col)
    else:
        print(f"Warning: Column '{col}' not found in the CSV.")

# === Formatting ===
plt.xlabel(time_col)
plt.ylabel("Pressure (mmHg)")
plt.title("Right Ventricle Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()