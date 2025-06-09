import matplotlib.pyplot as plt
import pandas as pd


filename = 'C:/Users/vanes/Documents/Next_Conditions_Output.parquet'
y_columns = ["P_rv"]     # Replace with the column names you want to plot

df = pd.read_parquet(filename)

# First column is time
time_col = df.columns[0]
time = df[time_col]

# Plot variables
plt.figure(figsize=(10, 5))
for col in y_columns:
    if col in df.columns:
        plt.plot(time, df[col], label=col)
    else:
        print(f"Warning: Column '{col}' not found in the CSV.")

plt.xlabel(time_col)
plt.ylabel("Pressure (mmHg)")
plt.title("Right Ventricle Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()