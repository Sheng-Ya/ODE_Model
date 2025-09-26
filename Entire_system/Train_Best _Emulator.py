import os
import joblib
import torch
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # non-interactive backend
import numpy as np
from autoemulate import AutoEmulate
# A = np.load(r'C:\Users\vanes\Downloads\Result_DGSM_chunked1.npy')
# AA = np.load(r'C:\Users\vanes\Downloads\Result_DGSM_chunked.npy')

from sklearn.model_selection import KFold

# X1 = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[:8400,:]
# X2 = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[8400:(8400+32400),:]
# X3 = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[100000:(100000+50400),:]
# X4 = np.load('DGSM_filtered_LHCS_500000_X_sample_HR_Plv_Prv_Vlv_Vrv_rest.npy')[300000:(300000+51600),:]# size [N, 299]
# Result1 = np.load('Result_DGSM_chunked_0_8400.npy')
# Result2 = np.load('Result_DGSM_chunked_8400_32400.npy')
# Result3 = np.load('Result_DGSM_chunked_100000_50400.npy')
# Result4 = np.load('Result_DGSM_chunked_300000_51600.npy')
# X_all = np.vstack((X1, X2, X3, X4))
# Result_all = np.vstack((Result1, Result2, Result3, Result4))

X_all = np.load('X_sample.npy')
Result_all = np.load('Results.npy')

mask = Result_all[:,0] != 0

X = X_all[mask, :]
Result = Result_all[mask, :]

# HR
Result = Result[:, 0] # Heart rate here

idx = np.random.choice(len(Result), size=30000, replace=False)
X = X[idx,:]
Result = Result[idx]

## EMULATION

# compare emulators
ae = AutoEmulate(X, Result, log_level="error")
ae.summarise()
best = ae.best_result()
print("Model with id: ", best.id, " performed best: ", best.model_name)
print(best.params)

# ae.save(best, "rbf_new")
os.makedirs("best_HR", exist_ok=True)
joblib.dump(best, "best_HR/RBF.joblib")

fig = ae.plot(best, fname="best_HR.png")







