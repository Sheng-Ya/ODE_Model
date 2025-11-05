import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

Variable = "Minute_vent"
ranked_exercise = pd.read_csv(f"{Variable}_sensitivities_exercise.csv", index_col=0)
ranked_rest = pd.read_csv(f"{Variable}_sensitivities.csv", index_col=0)

# --- Top N from exercise ---
top_n = 5
top_rest = ranked_rest.sort_values("ST", ascending=False).iloc[:top_n]
top_ex = ranked_exercise.sort_values("ST", ascending=False).iloc[:top_n]

# --- Calculate cumulative contribution of top 5 ---
cumusum = ranked_exercise["ST"].cumsum()
total_ST = cumusum.iloc[-1]
vars_top5 = ranked_exercise.iloc[:top_n]
top5_contribution = 100 * vars_top5["ST"].sum() / total_ST

print(f"Exercise: Top {top_n} variables contribute {top5_contribution:.1f}% of total sensitivity.")

cumusum = ranked_rest["ST"].cumsum()
total_ST = cumusum.iloc[-1]
vars_top5 = ranked_rest.iloc[:top_n]
top5_contribution = 100 * vars_top5["ST"].sum() / total_ST

print(f"Rest: Top {top_n} variables contribute {top5_contribution:.1f}% of total sensitivity.")


# --- Contribution percentage (optional info) ---
top_names_rest = top_rest.index.tolist()
top_names_ex = top_ex.index.tolist()

# --- Build combined x-axis order ---
all_top_names = list(top_names_rest)
for n in top_names_ex:
    if n not in all_top_names:
        all_top_names.append(n)

# --- Align both datasets to the combined parameter order ---
rest_vals = [ranked_rest.loc[n, "ST"] if n in ranked_rest.index else 0 for n in all_top_names]
rest_conf = [ranked_rest.loc[n, "ST_std"] if n in ranked_rest.index else 0 for n in all_top_names]
ex_vals = [ranked_exercise.loc[n, "ST"] if n in ranked_exercise.index else 0 for n in all_top_names]
ex_conf = [ranked_exercise.loc[n, "ST_std"] if n in ranked_exercise.index else 0 for n in all_top_names]

# --- Plot ---
x = np.arange(len(all_top_names))
width = 0.35

plt.rcParams.update({
    "font.size":30,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})



fig, ax = plt.subplots(figsize=(20, 6))

ax.bar(x - width/2, rest_vals, width, yerr=rest_conf, label='Rest', color="#3f69bf", capsize=3)
ax.bar(x + width/2, ex_vals, width, yerr=ex_conf, label='Exercise', color="#be6458", capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(all_top_names, rotation=45, ha="right")
ax.set_ylabel("Total-order index")
# ax.set_title(f"{Variable} Sobol Total-Order Sensitivities\nTop 5 Rest + Top 5 Exercise", fontsize=15)
# ax.legend(fontsize=50)

# --- Increase y-axis tick font size ---
ax.tick_params(axis='y', labelsize=25)  # change 16 to any desired font size
ax.tick_params(axis='x', labelsize=12)  # change 16 to any desired font size

for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')

plt.tight_layout()
plt.show()