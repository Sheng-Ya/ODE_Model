from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from types import MethodType
from multiprocessing import Pool, cpu_count

from functools import partial
from itertools import combinations, zip_longest

from SALib.analyze import common_args
from SALib.util import read_param_file, compute_groups_matrix, ResultDict

def analyze_NIMP(problem, Y,
                 calc_second_order=True,
                 num_resamples=100,
                 conf_level=0.95,
                 print_to_console=False,
                 parallel=False,
                 n_processors=None,
                 seed=None,
                 sample_weights=None):
    """Perform Sobol Analysis on model outputs.

    Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.  If calc_second_order is
    True, the dictionary also contains keys 'S2' and 'S2_conf'.

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
           mathematical models and their Monte Carlo estimates."  Mathematics
           and Computers in Simulation, 55(1-3):271-280,
           doi:10.1016/S0378-4754(00)00270-6.
    .. [2] Saltelli, A. (2002).  "Making best use of model evaluations to
           compute sensitivity indices."  Computer Physics Communications,
           145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.
    .. [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
           S. Tarantola (2010).  "Variance based sensitivity analysis of model
           output.  Design and estimator for the total sensitivity index."
           Computer Physics Communications, 181(2):259-270,
           doi:10.1016/j.cpc.2009.09.018.

    Examples
    --------
    >>> X = saltelli.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)

    WARNING: This is a modification of the original Saltelli output analysis
    to run a GSA, where binary weights are provided to exclude samples that
    are not plausible, according to a previously run history matching wave.

    The samples and weigths are generated from GSA_library.saltelli_pick_sampling.sample_NIMP.

    """

    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # print('                         USING MODIFIED VERSION OF SOBOL.ANALYZE                      ')
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    if seed:
        np.random.seed(seed)
    # determining if groups are defined and adjusting the number
    # of rows in the cross-sampled matrix accordingly
    if not problem.get('groups'):
        D = problem['num_vars']
    else:
        D = len(set(problem['groups']))

    if calc_second_order and Y.size % (2 * D + 2) == 0:
        N = int(Y.size / (2 * D + 2))
    elif not calc_second_order and Y.size % (D + 2) == 0:
        N = int(Y.size / (D + 2))
    else:
        raise RuntimeError("""
        Incorrect number of samples in model output file.
        Confirm that calc_second_order matches option used during sampling.""")

    if conf_level < 0 or conf_level > 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # if sample_weights is None:
    #     print('Weigths not set...Setting all of them to 1...')
    sample_weights = np.ones((Y.shape[0],),dtype=float)
    # else:
    #     print('Using sample weigths to exclude samples outside the non-implausible region...')

    # normalize the model output
    is_viable = np.where(sample_weights==1)[0]
    Y[is_viable] = (Y[is_viable] - np.mean(Y[is_viable])) / np.std(Y[is_viable])

    A, B, AB, BA = separate_output_values_NIMP(Y, D, N, calc_second_order)
    wA, wB, wAB, wBA = separate_samples_weigths(sample_weights, D, N, calc_second_order)

    r = np.random.randint(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)

    if not parallel:
        S = create_Si_dict_NIMP(D, calc_second_order)

        for j in range(D):
            S['S1'][j] = first_order_NIMP(A, AB[:, j], B, wA, wAB[:, j], wB)
            S['S1_conf'][j] = Z * first_order_NIMP(A[r], AB[r, j], B[r], wA[r], wAB[r,j], wB[r]).std(ddof=1)
            S['ST'][j] = total_order_NIMP(A, AB[:, j], B, wA, wAB[:, j], wB)
            S['ST_conf'][j] = Z * total_order_NIMP(A[r], AB[r, j], B[r], wA[r], wAB[r, j], wB[r]).std(ddof=1)

        # Second order (+conf.)
        if calc_second_order:
            for j in range(D):
                for k in range(j + 1, D):
                    S['S2'][j, k] = second_order_NIMP(
                        A, AB[:, j], AB[:, k], BA[:, j], B,
                        wA, wAB[:, j], wAB[:, k], wBA[:, j], wB)
                    S['S2_conf'][j, k] = Z * second_order_NIMP(A[r], AB[r, j],
                                                          AB[r, k], BA[r, j], B[r],
                                                          wA[r], wAB[r, j],
                                                          wAB[r, k], wBA[r, j], wB[r]).std(ddof=1)

    else:
        raise Exception('Cannot execute task in parallel')

    # Print results to console
    if print_to_console:
        print_indices_NIMP(S, problem, calc_second_order)

    # Add problem context and override conversion method for special case
    S.problem = problem
    S.to_df_NIMP = MethodType(to_df_NIMP, S)
    return S


def first_order_NIMP(A, AB, B, wA, wAB, wB):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance

    wA_bool = np.where(wA==1)[0]
    wB_bool = np.where(wB==1)[0]
    wAB_bool = np.where(wAB==1)[0]

    total_viable = np.intersect1d(wA_bool,wB_bool)
    total_viable = np.intersect1d(total_viable,wAB_bool)

    # if len(total_viable)==0:
    #     warnings.warn('No viable samples detected to compute Si.')
    # else:
    #     print('Found '+str(len(total_viable))+' samples to use for Si.')

    return np.mean(B[total_viable] * (AB[total_viable] - A[total_viable]), axis=0) / np.var(np.r_[A[total_viable], B[total_viable]], axis=0)

def total_order_NIMP(A, AB, B, wA, wAB, wB):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance

    wA_bool = np.where(wA==1)[0]
    wB_bool = np.where(wB==1)[0]
    wAB_bool = np.where(wAB==1)[0]

    total_viable = np.intersect1d(wA_bool,wB_bool)
    total_viable = np.intersect1d(total_viable,wAB_bool)

    # if len(total_viable)==0:
    #     warnings.warn('No viable samples detected to compute STi.')
    # else:
    #     print('Found '+str(len(total_viable))+' samples to use for STi.')

    return 0.5 * np.mean((A[total_viable] - AB[total_viable]) ** 2, axis=0) / np.var(np.r_[A[total_viable], B[total_viable]], axis=0)


def second_order_NIMP(A, ABj, ABk, BAj, B, wA, wABj, wABk, wBAj, wB):
    # Second order estimator following Saltelli 2002
    Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(np.r_[A, B], axis=0)
    Sj = first_order_NIMP(A, ABj, B, wA, wABj, wB)
    Sk = first_order_NIMP(A, ABk, B, wA, wABk, wB)

    return Vjk - Sj - Sk


def create_Si_dict_NIMP(D, calc_second_order):
    # initialize empty dict to store sensitivity indices
    S = ResultDict((k, np.zeros(D))
                   for k in ('S1', 'S1_conf', 'ST', 'ST_conf'))

    if calc_second_order:
        S['S2'] = np.zeros((D, D))
        S['S2'][:] = np.nan
        S['S2_conf'] = np.zeros((D, D))
        S['S2_conf'][:] = np.nan

    return S

def separate_output_values_NIMP(Y, D, N, calc_second_order):
    AB = np.zeros((N, D))
    BA = np.zeros((N, D)) if calc_second_order else None

    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0:Y.size:step]
    B = Y[(step - 1):Y.size:step]
    for j in range(D):
        AB[:, j] = Y[(j + 1):Y.size:step]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D):Y.size:step]

    return A, B, AB, BA

def separate_samples_weigths(sample_weights, D, N, calc_second_order):
    wAB = np.zeros((N, D))
    wBA = np.zeros((N, D)) if calc_second_order else None

    step = 2 * D + 2 if calc_second_order else D + 2

    wA = sample_weights[0:sample_weights.size:step]
    wB = sample_weights[(step - 1):sample_weights.size:step]
    for j in range(D):
        wAB[:, j] = sample_weights[(j + 1):sample_weights.size:step]
        if calc_second_order:
            wBA[:, j] = sample_weights[(j + 1 + D):sample_weights.size:step]

    return wA, wB, wAB, wBA

def create_task_list_NIMP(D, calc_second_order, n_processors):
    # Create list with one entry (key, parameter 1, parameter 2) per sobol
    # index (+conf.). This is used to supply parallel tasks to
    # multiprocessing.Pool
    tasks_first_order = [[d, j, None] for j in range(
        D) for d in ('S1', 'S1_conf', 'ST', 'ST_conf')]

    # Add second order (+conf.) to tasks
    tasks_second_order = []
    if calc_second_order:
        tasks_second_order = [[d, j, k] for j in range(D) for k in
                              range(j + 1, D) for d in ('S2', 'S2_conf')]

    if n_processors is None:
        n_processors = min(cpu_count(), len(
            tasks_first_order) + len(tasks_second_order))

    if not calc_second_order:
        tasks = np.array_split(tasks_first_order, n_processors)
    else:
        # merges both lists alternating its elements and splits the
        # resulting lists into n_processors sublists
        tasks = np.array_split([v for v in sum(
            zip_longest(tasks_first_order[::-1], tasks_second_order), ())
            if v is not None], n_processors)

    return tasks, n_processors


def Si_list_to_dict_NIMP(S_list, D, calc_second_order):
    # Convert the parallel output into the regular dict format for
    # printing/returning
    S = create_Si_dict_NIMP(D, calc_second_order)
    L = []
    for l in S_list:  # first reformat to flatten
        L += l

    for s in L:  # First order (+conf.)
        if s[2] is None:
            S[s[0]][s[1]] = s[3]
        else:
            S[s[0]][s[1], s[2]] = s[3]

    return S


def Si_to_pandas_dict_NIMP(S_dict):
    """Convert Si information into Pandas DataFrame compatible dict.

    Parameters
    ----------
    S_dict : ResultDict
        Sobol sensitivity indices

    See Also
    ----------
    Si_list_to_dict

    Returns
    ----------
    tuple : of total, first, and second order sensitivities.
            Total and first order are dicts.
            Second order sensitivities contain a tuple of parameter name
            combinations for use as the DataFrame index and second order
            sensitivities.
            If no second order indices found, then returns tuple of
            (None, None)

    Examples
    --------
    >>> X = saltelli.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)
    >>> T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si, problem)
    """
    problem = S_dict.problem
    total_order = {
        'ST': S_dict['ST'],
        'ST_conf': S_dict['ST_conf']
    }
    first_order = {
        'S1': S_dict['S1'],
        'S1_conf': S_dict['S1_conf']
    }

    idx = None
    second_order = None
    if 'S2' in S_dict:
        names = problem['names']
        idx = list(combinations(names, 2))
        second_order = {
            'S2': [S_dict['S2'][names.index(i[0]), names.index(i[1])]
                   for i in idx],
            'S2_conf': [S_dict['S2_conf'][names.index(i[0]), names.index(i[1])]
                        for i in idx]
        }
    return total_order, first_order, (idx, second_order)


def to_df_NIMP(self):
    '''Conversion method to Pandas DataFrame. To be attached to ResultDict.

    Returns
    ========
    List : of Pandas DataFrames in order of Total, First, Second
    '''
    total, first, (idx, second) = Si_to_pandas_dict_NIMP(self)
    names = self.problem['names']
    ret = [pd.DataFrame(total, index=names),
           pd.DataFrame(first, index=names)]

    if second:
        ret += [pd.DataFrame(second, index=idx)]

    return ret


def print_indices_NIMP(S, problem, calc_second_order):
    # Output to console
    if not problem.get('groups'):
        title = 'Parameter'
        names = problem['names']
        D = problem['num_vars']
    else:
        title = 'Group'
        _, names = compute_groups_matrix(problem['groups'])
        D = len(names)

    print('%s S1 S1_conf ST ST_conf' % title)

    for j in range(D):
        print('%s %f %f %f %f' % (names[j], S['S1'][
            j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j]))

    if calc_second_order:
        print('\n%s_1 %s_2 S2 S2_conf' % (title, title))

        for j in range(D):
            for k in range(j + 1, D):
                print("%s %s %f %f" % (names[j], names[k],
                                       S['S2'][j, k], S['S2_conf'][j, k]))


def cli_parse_NIMP(parser):
    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2],
                        help='Maximum order of sensitivity indices to '
                        'calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for Sobol '
                        'confidence intervals')
    parser.add_argument('--parallel', action='store_true', help='Makes '
                        'use of parallelization.',
                        dest='parallel')
    parser.add_argument('--processors', type=int, required=False,
                        default=None,
                        help='Number of processors to be used with the ' +
                        'parallel option.', dest='n_processors')
    return parser


def cli_action_NIMP(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
                   usecols=(args.column,))

    analyze_NIMP(problem, Y, (args.max_order == 2),
            num_resamples=args.resamples, print_to_console=True,
            parallel=args.parallel, n_processors=args.n_processors,
            seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse_NIMP, cli_action_NIMP)