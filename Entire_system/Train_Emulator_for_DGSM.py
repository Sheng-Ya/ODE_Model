import os

import joblib
from SALib import ProblemSpec
from SALib.plotting.bar import plot as barplot
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
# from SALib.sample import saltelli
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
import matplotlib.pyplot as plt
import numpy as np
from autoemulate.compare import AutoEmulate
from sklearn.model_selection import KFold

X = np.load('LHCS_152000_X_samples_HR_P_sys_P_dia_rest.npy')
Result = np.load('LHCS_152000_Result_HR_P_sys_P_dia_rest.npy')

X = X[Result[:,0] != 0]
Result = Result[Result[:,0] != 0][:, 0]

idx = np.random.choice(len(Result), size=30000, replace=False)

# df = (Result - Result.mean()) / Result.std()
#
X = X[idx,:]
# Result = df[:1000]

Result = Result[idx]


# compare emulators
ae = AutoEmulate()
ae.setup(X, Result,  param_search=True, param_search_iters=30, n_jobs=-1, models=["rbf"], cross_validator=KFold(n_splits=3))

best_emulator = ae.compare()

ae.summarise_cv()

ae.plot_cv()
# ae.plot_cv(style="actual_vs_predicted")
ae.plot_cv(style="residual_vs_predicted")


rbf = ae.get_model("RadialBasisFunctions")
ae.evaluate(rbf)
ae.plot_eval(rbf)
rbf_final = ae.refit(rbf)

ae.save(rbf_final, "rbf_final_hyper_30000")


# gp = ae.get_model("GaussianProcess")
# ae.evaluate(gp)
# ae.plot_eval(gp)
# gp_final = ae.refit(gp)

# os.makedirs("gp_final", exist_ok=True)
# joblib.dump(gp_final, "gp_final/GaussianProcess.joblib")

# ae.save(gp_final, "gp_final")