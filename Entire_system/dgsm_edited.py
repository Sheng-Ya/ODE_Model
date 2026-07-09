from matplotlib import pyplot as plt
from scipy.stats import norm

import numpy as np

from SALib.analyze import common_args
from SALib.util import read_param_file, ResultDict


def analyze(
    problem, X, Y, num_resamples=100, conf_level=0.95, print_to_console=False, seed=None
):
    """Calculates Derivative-based Global Sensitivity Measure on model outputs.

    Returns a dictionary with keys 'vi', 'vi_std', 'dgsm', and 'dgsm_conf',
    where each entry is a list of size D (the number of parameters) containing
    the indices in the same order as the parameter file.


    Notes
    -----
    Compatible with:
        `finite_diff` : :func:`SALib.sample.finite_diff.sample`


    Examples
    --------
        >>> X = finite_diff.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = dgsm.analyze(problem, Y, print_to_console=False)


    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.matrix
        The NumPy matrix containing the model inputs
    Y : numpy.array
        The NumPy array containing the model outputs
    num_resamples : int
        The number of resamples used to compute the confidence
        intervals (default 1000)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed to generate a random number


    References
    ----------
    1. Sobol, I. M. and S. Kucherenko (2009). "Derivative based global
           sensitivity measures and their link with global sensitivity
           indices." Mathematics and Computers in Simulation, 79(10):3009-3017,
           doi:10.1016/j.matcom.2009.01.023.
    """
    if seed:
        np.random.seed(seed)

    D = problem["num_vars"]
    Y_size = Y.size

    if Y_size % (D + 1) == 0:
        N = int(Y_size / (D + 1))
    else:
        raise RuntimeError("Incorrect number of samples in model output file.")

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    dims = (N, D)
    perturbed = np.empty(dims)
    X_perturbed = np.empty(dims)
    step = D + 1

    base = Y[0:Y_size:step]
    X_base = X[0:Y_size:step, :]

    # First order (+conf.) and Total order (+conf.)
    keys = ("vi", "vi_std", "dgsm", "dgsm_conf")
    S = ResultDict((k, np.empty(D)) for k in keys)
    S["names"] = problem["names"]

    bounds = problem["bounds"]
    for j in range(D):
        # use all numbers of base points for one variable
        perturbed[:, j] = Y[(j + 1) : Y_size : step]
        X_perturbed[:, j] = X[(j + 1) : Y_size : step, j]
        variable = S["names"][j]

        # X_values = X_perturbed[:, j]
        # Y_values = perturbed[:, j]

        # if variable == "MRBCO2":
        #     A = list(X_base[4,:])
        #     AA = list(X_perturbed[4,:])
        #
        #     plt.figure()
        #     plt.scatter(X_values, Y_values, alpha=0.7)
        #     plt.xlabel(variable)
        #     plt.ylabel("Heart Rate")
        #     plt.title(f"Heart Rate vs {variable}")
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.show()

        diff = X_perturbed[:, j] - X_base[:, j]
        perturbed_j = perturbed[:, j]


        mask = (base != 0) & (perturbed_j != 0)

        # S["vi"][j], S["vi_std"][j] = calc_vi_stats(base[mask], perturbed_j[mask], diff[mask], variable)
        S["dgsm"][j], S["dgsm_conf"][j] = calc_dgsm(
            base[mask], perturbed_j[mask], diff[mask], bounds[j], num_resamples, conf_level, variable
        )

    if print_to_console:
        print(S.to_df())

    return S


def calc_vi_stats(base, perturbed, x_delta, variable):
    """Calculate v_i mean and std.

    v_i sensitivity measure following Sobol and Kucherenko (2009)
    For comparison, Morris mu* < sqrt(v_i)

    Same as calc_vi_mean but returns standard deviation as well.
    """
    dfdx = ((perturbed - base) / x_delta) ** 2

    # Calculate mean and std of all values
    mean_dfdx = np.mean(dfdx)
    std_dfdx = np.std(dfdx)
    # mean_dfdx1 = np.mean((perturbed - base)**2)
    # std_dfdx1 = np.std((perturbed - base)**2)
    mean_delta = np.mean(x_delta)
    std_delta = np.std(x_delta)

    # Keep values within 2 standard deviations from the mean
    mask = np.abs(x_delta - mean_delta) <= 3 * std_delta
    # mask1 = np.abs(((perturbed - base)**2) - mean_dfdx1) <= 2*std_dfdx1
    dfdx_filtered = dfdx[mask]


    mean_dfdx = np.mean(dfdx_filtered)
    std_dfdx = np.std(dfdx_filtered)
    mask = np.abs(dfdx_filtered - mean_dfdx) <= 3 * std_dfdx
    dfdx_filtered = dfdx_filtered[mask]


    return np.mean(dfdx_filtered), np.std(dfdx_filtered)
    # return np.mean(dfdx), np.std(dfdx)
    # return np.median(dfdx), np.std(dfdx)


def calc_vi_mean(base, perturbed, x_delta, variable):
    """Calculate v_i mean.

    Same as calc_vi_stats but only returns the mean.
    """
    dfdx = ((perturbed - base) / x_delta) ** 2

    # Calculate mean and std of all values
    mean_dfdx = np.mean(dfdx)
    std_dfdx = np.std(dfdx)
    # mean_dfdx1 = np.mean((perturbed - base) ** 2)
    # std_dfdx1 = np.std((perturbed - base) ** 2)
    mean_delta = np.mean(x_delta)
    std_delta = np.std(x_delta)

    # Keep values within 2 standard deviations from the mean
    mask = np.abs(x_delta - mean_delta) <= 3 * std_delta

    # Keep values within 2 standard deviations from the mean
    # mask = np.abs(dfdx - mean_dfdx) <= 3 * std_dfdx
    # mask1 = np.abs(((perturbed - base)**2) - mean_dfdx1) <= 2*std_dfdx1

    dfdx_filtered = dfdx[mask]


    mean_dfdx = np.mean(dfdx_filtered)
    std_dfdx = np.std(dfdx_filtered)
    mask = np.abs(dfdx_filtered - mean_dfdx) <= 3 * std_dfdx
    dfdx_filtered = dfdx_filtered[mask]

    return np.mean(dfdx_filtered)
    # return dfdx.mean()
    # return np.median(dfdx_filtered)


def calc_dgsm(base, perturbed, x_delta, bounds, num_resamples, conf_level, variable):
    """v_i sensitivity measure following Sobol and Kucherenko (2009).
    For comparison, total order S_tot <= dgsm
    """
    D = np.var(base)
    vi, _ = calc_vi_stats(base, perturbed, x_delta, variable)
    # A = calc_vi_mean(base, perturbed, x_delta, variable)
    dgsm = vi * (bounds[1] - bounds[0]) ** 2 / (D * np.pi**2)
    # AA = A * (bounds[1] - bounds[0]) ** 2 / (D * np.pi**2)

    # if don't need uncertainty bars, no need for bootstrap
    if num_resamples <= 0:
        return dgsm, np.nan

    len_base = len(base)
    s = np.empty(num_resamples)
    r = np.random.randint(len_base, size=(num_resamples, len_base))

    # w = np.empty(num_resamples)

    for i in range(num_resamples):
        r_i = r[i]
        # s[i] = calc_vi_mean(base[r_i], perturbed[r_i], x_delta[r_i], variable)
        mean_value = calc_vi_mean(base[r_i], perturbed[r_i], x_delta[r_i], variable)
        s[i] = mean_value * (bounds[1] - bounds[0]) ** 2 / (D * np.pi ** 2)

    #     mean_value2,_ = calc_vi_stats(base[r_i], perturbed[r_i], x_delta[r_i], variable)
    #     w[i] = mean_value2 * (bounds[1] - bounds[0]) ** 2 / (D * np.pi ** 2)
    #
    # AAA = norm.ppf(0.5 + conf_level / 2.0) * s.std(ddof=1)
    # AAAA = norm.ppf(0.5 + conf_level / 2.0) * w.std(ddof=1)

    return dgsm, norm.ppf(0.5 + conf_level / 2.0) * s.std(ddof=1)


def cli_parse(parser):
    parser.add_argument(
        "-X",
        "--model-input-file",
        type=str,
        required=True,
        default=None,
        help="Model input file",
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=1000,
        help="Number of bootstrap resamples for Sobol \
                           confidence intervals",
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(
        problem,
        X,
        Y,
        num_resamples=args.resamples,
        print_to_console=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
