import logging
import warnings
import numpy as np
from scipy.stats import gaussian_kde
import os
import joblib
from autoemulate.core.model_selection import evaluate, r2_metric
from autoemulate.core.model_selection import bootstrap
import torch
# from tqdm import tqdm
# import tqdm_joblib

from joblib import Parallel, delayed
import math
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from torch.distributions.multivariate_normal import MultivariateNormal

# silence_resource_tracker_childprocesserror.py
import sys

_old_hook = sys.unraisablehook

def _quiet_resource_tracker(unraisable):
    exc = unraisable.exc_value
    obj = unraisable.object

    # Filter only: multiprocessing.resource_tracker ResourceTracker + ChildProcessError
    if isinstance(exc, ChildProcessError):
        if obj is not None:
            mod = getattr(obj.__class__, "__module__", "")
            name = getattr(obj.__class__, "__name__", "")
            if mod == "multiprocessing.resource_tracker" and name == "ResourceTracker":
                return  # swallow it

    _old_hook(unraisable)

sys.unraisablehook = _quiet_resource_tracker


from autoemulate.core.device import TorchDeviceMixin
from autoemulate.core.logging_config import get_configured_logger
from autoemulate.core.plotting import display_figure
from autoemulate.core.results import Result
from autoemulate.core.types import DeviceLike, DistributionLike, TensorLike
from autoemulate.data.utils import set_random_seed
from autoemulate.emulators import TransformedEmulator, get_emulator_class
# from autoemulate.simulations.base import Simulator
from Simulator_new import Simulator

logger = logging.getLogger("autoemulate")


class HistoryMatching(TorchDeviceMixin):
    r"""
    History Matching class for model calibration.

    History matching is a model calibration method, which uses observed data to
    rule out ``implausible`` parameter values. The implausibility metric is:

    .. math::

        I_i(\bar{x_0}) = \frac{|z_i - \mathbb{E}(f_i(\bar{x_0}))|}
        {\sqrt{\text{Var}[z_i - \mathbb{E}(f_i(\bar{x_0}))]}}

    Queried parameters above a given implausibility threshold are ruled out (RO)
    whereas all other parameters are marked as not ruled out yet (NROY).
    """

    def __init__(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        device: DeviceLike | None = None,
    ):
        """
        Initialize the history matching object.

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default of ``1`` indicates that the largest implausibility will be used.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        """
        TorchDeviceMixin.__init__(self, device=device)

        self.threshold = threshold
        self.discrepancy = model_discrepancy
        self.out_dim = len(observations)

        if rank > self.out_dim or rank < 1:
            raise ValueError(
                f"Rank ({rank}) is outside valid range between 1 and output dimension "
                f"of simulator ({self.out_dim})",
            )
        self.rank = rank

        # Save mean and variance of observations, shape: [1, n_outputs]
        self.obs_means, self.obs_vars = self._process_observations(observations)

    def _process_observations(
        self,
        observations: dict[str, tuple[float, float]] | dict[str, float],
    ) -> tuple[TensorLike, TensorLike]:
        """
        Turn observations into tensors of shape [1, n_inputs].

        Parameters
        ----------
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of observations and the associated noise (which can be 0) specified
            as variances.
        """
        values = torch.tensor(list(observations.values()), device=self.device)

        # No variance
        if values.ndim == 1:
            means = values
            variances = torch.zeros_like(means, device=self.device)
        # Values are (mean, variance)
        elif values.ndim == 2:
            means = values[:, 0]
            variances = values[:, 1]
        else:
            msg = "Observations must be either float or tuple of two floats."
            raise ValueError(msg)

        # Reshape observation tensors for broadcasting
        return means.view(1, -1), variances.view(1, -1)

    def _create_nroy_mask(self, implausibility: TensorLike) -> TensorLike:
        """
        Create mask for NROY points based on rank.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested parameters.

        Returns
        -------
        TensorLike
            Tensor indicating whether each implausability score is NROY
            given self.rank and self.threshold values.
        """
        # Sort implausibilities for each sample (descending)
        I_sorted, index_for_sort = torch.sort(implausibility, dim=1, descending=True)
        values, row_idx = torch.sort(I_sorted[:, 0], descending=True)
        implausibility_sorted_by_col0 = I_sorted[row_idx]
        index_of_implausibility_sorted_by_col0 = index_for_sort[row_idx]

        # The rank-th highest output implausibility must be <= threshold
        return I_sorted[:, self.rank - 1] <= self.threshold

    def get_nroy(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of NROY points from implausibility scores.

        If `x` is provided, returns parameter values at NROY indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored input parameters.

        Returns
        -------
        TensorLike
            Indices of NROY points or `x` parameters at NROY indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def get_ro(
        self, implausibility: TensorLike, x: TensorLike | None = None
    ) -> TensorLike:
        """
        Get indices of RO points from implausibility scores.

        If `x` is provided, returns parameter values at RO indices.

        Parameters
        ----------
        implausibility: TensorLike
            Tensor of implausibility scores for tested input parameters.
        x: Tensorlike | None
            Optional tensor of scored iput parameters.

        Returns
        -------
        TensorLike
            Indices of RO points or `x` parameters at RO indices.
        """
        nroy_mask = self._create_nroy_mask(implausibility)
        idx = torch.where(~nroy_mask)[0]
        if x is None:
            return idx
        return x[idx]

    def calculate_implausibility(
        self,
        pred_means: TensorLike,  # [n_samples, n_outputs]
        pred_vars: TensorLike,  # [n_samples, n_outputs]
    ) -> TensorLike:
        """
        Calculate implausibility scores.

        Parameters
        ----------
        pred_means: TensorLike
            Tensor of prediction means [n_samples, n_outputs]
        pred_vars: TensorLike
            Tensor of prediction variances [n_samples, n_outputs].

        Returns
        -------
        TensorLike
            Tensor of implausibility scores.
        """
        # Additional variance due to model discrepancy (defaults to 0)
        discrepancy = torch.full_like(
            self.obs_vars, self.discrepancy, device=self.device
        )
        # obs_vars is the obs uncertainty, eg HR 1.1, std 0.1. Discrepancy is the uncertainty of the emulator prediction in that area
        # Calculate total variance
        Vs = pred_vars + discrepancy + self.obs_vars

        adjusted_pred_means = pred_means.clone()

        la_den = adjusted_pred_means[:, 14] - adjusted_pred_means[:, 13]
        ra_den = adjusted_pred_means[:, 10] - adjusted_pred_means[:, 9]

        adjusted_pred_means[:, 17] = (adjusted_pred_means[:, 17] - adjusted_pred_means[:, 13]) / la_den
        adjusted_pred_means[:, 18] = (adjusted_pred_means[:, 18] - adjusted_pred_means[:, 9]) / ra_den

        # Calculate implausibility
        return torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

    @staticmethod
    def generate_param_bounds(
        nroy_x: TensorLike,
        buffer_ratio: float = 0.05,
        param_names: list[str] | None = None,
        min_samples: int = 1,
    ) -> dict[str, tuple[float, float]] | None:
        """
        Generate lower/upper parameter bounds as min/max of NROY samples.

        Parameters
        ----------
        nroy_x: TensorLike
            A tensor of NROY parameter samples [n_samples, n_inputs].
        buffer_ratio: float
            A scaling factor used to expand the bounds of the (NROY) parameter space.
            It is applied as a ratio of the range (max_val - min_val) of each input
            parameter to create a buffer around the NROY minimum and maximum values.
        param_names: list[str] | None
            Optional list of parameter names. If None, uses default `["x1", ..., "xn"]`.
        min_samples: int
            Minimum number of samples needed to generate new bounds.

        Returns
        -------
        dict[str, [float, float]] | None
            The generated [lower, upper] parameter bounds. Returns None if there are
            not enough samples to generate bounds from.
        """
        if param_names is None:
            param_names = [f"x{i + 1}" for i in range(nroy_x.shape[1])]

        if nroy_x.shape[0] > min_samples:
            min_val = torch.min(nroy_x, dim=0).values
            max_val = torch.max(nroy_x, dim=0).values
            buffer = (max_val - min_val) * buffer_ratio
            lower_bound = min_val - buffer
            upper_bound = max_val + buffer

            return {
                param: (lower_bound[i].item(), upper_bound[i].item())
                for i, param in enumerate(param_names)
            }
        return None


class HistoryMatchingWorkflow(HistoryMatching):
    """
    History Matching Workflow class.

    Run history matching workflow:
    - sample parameter values to test from the current NROY parameter space
    - use emulator to rule out implausible parameter samples
    - run simulations for a subset of the NROY parameters
    - refit the emulator using the simulated data
    """

    def __init__(
        self,
        simulator: Simulator,
        result: Result,
        observations: dict[str, tuple[float, float]] | dict[str, float],
        threshold: float = 3.0,
        model_discrepancy: float = 0.0,
        rank: int = 1,
        train_x: TensorLike | None = None,
        train_y: TensorLike | None = None,
        calibration_params: list[str] | None = None,
        overlap_params: list[str] | None = None,
        exercise_only_params: list[str] | None = None,
        rest_overlap_source: str = "nroy",
        rest_overlap_path: str | None = None,
        rest_posterior_mass: float = 0.95,
        rest_posterior_region: str = "hpd",
        rest_overlap_sampling: str = "empirical",
        device: DeviceLike | None = None,
        random_seed: int | None = None,
        log_level: str = "debug",
    ):
        """
        Initialize the history matching workflow object.

        Parameters
        ----------
        simulator: Simulator
            A simulator.
        result: Result
            A Result object containing the pre-trained emulator and its hyperparameters.
        observations: dict[str, tuple[float, float] | dict[str, float]
            For each output variable, specifies observed [value, noise] (with noise
            specified as variances). In case of no uncertainty in observations, provides
            just the observed value.
        threshold: float
            Implausibility threshold (query points with implausibility scores that
            exceed this value are ruled out). Defaults to 3, which is considered
            a good value for simulations with a single output.
        model_discrepancy: float
            Additional variance to include in the implausibility calculation.
        rank: int
            Scoring method for multi-output problems. Must be 1 <= rank <= n_outputs.
            When the implausibility scores are ordered across outputs, it indicates
            which rank to use when determining whether the query point is NROY. The
            default val of ``1`` indicates that the largest implausibility will be used.
        train_x: TensorLike | None
            Optional tensor of input data the emulator was trained on.
        train_y: TensorLike | None
            Optional tensor of output data the emulator was trained on.
        calibration_params: list[str] | None
            Optional subset of parameters to calibrate. These have to correspond to the
            parameters that the emulator was trained on. If None, calibrate all
            simulator parameters.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        random_seed: int | None
            Optional random seed for reproducibility. If None, no seed is set.
        log_level: str
            The logging level to use. One of: "debug", "info", "warning", "error",
            "critical", "progress_bar" (default).
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.logger, self.progress_bar = get_configured_logger(log_level)

        self.result = result
        self.emulator = result.model
        self.emulator.device = self.device

        # New data is simulated in `run()` and appended here
        # It can be used to refit the emulator
        if train_x is not None and train_y is not None:
            self.train_x = train_x.float().to(self.device)
            self.train_y = train_y.float().to(self.device)
        else:
            self.train_x = torch.empty((0, self.simulator.in_dim), device=self.device)
            self.train_y = torch.empty((0, self.simulator.out_dim), device=self.device)

        # New NROY samples are generated in `run()` and used in `cloud_sample()`
        # We only ever use the most recent NROY samples
        # This means `self.nroy_samples` gets overwritten each time `run()` is called
        self.nroy_samples = None

        # If use `run_waves()`, results are stored here
        self.wave_results = []

        # Save names and indices of parameters to calibrate
        self.calibration_params = calibration_params or list(
            simulator.parameters_range.keys()
        )
        self.overlap_params = overlap_params or []
        self.exercise_only_params = exercise_only_params or []
        self.rest_overlap_source = rest_overlap_source
        self.rest_overlap_path = rest_overlap_path
        self.rest_posterior_mass = rest_posterior_mass
        self.rest_posterior_region = rest_posterior_region
        self.rest_overlap_sampling = rest_overlap_sampling
        self._rest_overlap_reference_cache = None

        self.parameter_idx = [
            self.simulator.get_parameter_idx(param) for param in self.calibration_params
        ]
        self.overlap_idx = [
            self.simulator.get_parameter_idx(param) for param in self.overlap_params
        ]
        self.exercise_only_idx = [
            self.simulator.get_parameter_idx(param) for param in self.exercise_only_params
        ]

    def _is_within_bounds(
        self, sample: TensorLike, bounds_dict: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Check if `sample` is within the bounds defined in `bounds_dict`.

        Parameters
        ----------
        sample: torch.Tensor
            A single sample of input parameters to check, shape [1, in_dim].
        bounds_dict: dict of {param_name: [lower, upper]}
            A dictionary of parameter bounds for each parameter.

        Returns
        -------
        bool
            True if the sample is within the bounds, False otherwise.
        """
        sample = sample.squeeze(0)  # shape: [in_dim]
        lowers = torch.tensor(
            [bounds[0] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        uppers = torch.tensor(
            [bounds[1] for bounds in bounds_dict.values()],
            dtype=sample.dtype,
            device=sample.device,
        )
        return bool(torch.all((sample >= lowers) & (sample <= uppers)).item())

    def _phi(self, x):
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def _phi_inv(self, u):
        return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

    def truncated_normal_1d(self, mean, std, low, high, n_samples):
        """
        mean/std/low/high: [d]
        returns: [n_samples, d] all within [low, high]
        """
        eps = 1e-7
        std = torch.clamp(std, min=1e-12)

        a = (low - mean) / std
        b = (high - mean) / std

        pa = torch.clamp(self._phi(a), eps, 1 - eps)
        pb = torch.clamp(self._phi(b), eps, 1 - eps)

        # sample uniformly between CDF(low) and CDF(high)
        u = torch.rand((n_samples, mean.numel()), device=mean.device, dtype=mean.dtype)
        u = pa + u * (pb - pa)
        u = torch.clamp(u, eps, 1 - eps)

        z = self._phi_inv(u)
        x = mean + std * z

        # numerical safety
        return torch.clamp(x, low, high)

    @staticmethod
    def _log_det_jac_np(z, prior_lo, prior_hi):
        width = np.clip(prior_hi - prior_lo, 1e-12, None)
        log_width = np.log(width)
        log_sig_pos = -np.logaddexp(0.0, -z)
        log_sig_neg = -np.logaddexp(0.0, z)
        return (log_sig_pos + log_sig_neg + log_width).sum(axis=-1)

    def _load_rest_overlap_reference(self):
        if self._rest_overlap_reference_cache is not None:
            return self._rest_overlap_reference_cache

        overlap_dim = len(self.overlap_idx)
        if overlap_dim == 0:
            empty = torch.empty((0, 0), dtype=torch.float32, device=self.device)
            self._rest_overlap_reference_cache = {
                "samples": empty,
                "low": torch.empty(0, dtype=torch.float32, device=self.device),
                "high": torch.empty(0, dtype=torch.float32, device=self.device),
                "source": "none",
                "n_reference": 0,
            }
            return self._rest_overlap_reference_cache

        source = (self.rest_overlap_source or "nroy").lower()

        if source == "nroy":
            path = self.rest_overlap_path or "nroy_samples_rest.pt"
            rest_samples = torch.load(path, map_location="cpu").float()
            overlap_idx_t = torch.tensor(self.overlap_idx, dtype=torch.long)
            overlap_samples = rest_samples[:, overlap_idx_t]

            cache = {
                "samples": overlap_samples.to(self.device),
                "low": overlap_samples.min(dim=0).values.to(self.device),
                "high": overlap_samples.max(dim=0).values.to(self.device),
                "source": "nroy",
                "n_reference": int(overlap_samples.shape[0]),
            }
            self._rest_overlap_reference_cache = cache
            print(
                f"Loaded {cache['n_reference']} overlap reference samples from rest NROY."
            )
            return cache

        if source != "posterior":
            raise ValueError(
                f"Unknown rest_overlap_source='{self.rest_overlap_source}'. "
                "Use 'nroy' or 'posterior'."
            )

        if self.rest_overlap_path is None:
            raise ValueError(
                "rest_overlap_path must point to a MCMC_Rest_* run directory "
                "when rest_overlap_source='posterior'."
            )

        run_dir = self.rest_overlap_path
        posterior = np.load(os.path.join(run_dir, "posterior_samples.npy"))
        subset_vars = np.load(
            os.path.join(run_dir, "subset_vars.npy"), allow_pickle=True
        ).tolist()

        overlap_col_idx = [subset_vars.index(name) for name in self.overlap_params]
        overlap_np = posterior[:, overlap_col_idx].astype(np.float32, copy=False)

        region = (self.rest_posterior_region or "hpd").lower()
        mass = float(self.rest_posterior_mass)

        if region == "hpd": # sample values from the posterior distribution, excluding the worst 5% log posterior
            posterior_z = np.load(os.path.join(run_dir, "posterior_z.npy"))
            log_post_z = np.load(os.path.join(run_dir, "log_posterior_trace.npy"))
            prior_lo = np.load(os.path.join(run_dir, "prior_lower.npy"))
            prior_hi = np.load(os.path.join(run_dir, "prior_upper.npy"))
            log_post_theta = log_post_z - self._log_det_jac_np(
                posterior_z, prior_lo, prior_hi
            )
            keep = max(1, int(np.ceil(mass * overlap_np.shape[0])))
            idx = np.argsort(log_post_theta)[::-1][:keep]
            overlap_np = overlap_np[idx]
            low_np = overlap_np.min(axis=0) # sets lower range of each parameter after removing 5%
            high_np = overlap_np.max(axis=0) # sets higher range of each parameter after removing 5%

        else:
            low_np = overlap_np.min(axis=0)
            high_np = overlap_np.max(axis=0)


        cache = {
            "samples": torch.from_numpy(overlap_np).to(self.device),
            "low": torch.from_numpy(low_np.astype(np.float32)).to(self.device),
            "high": torch.from_numpy(high_np.astype(np.float32)).to(self.device),
            "source": "posterior",
            "n_reference": int(overlap_np.shape[0]),
            "region": region,
            "mass": mass,
        }
        self._rest_overlap_reference_cache = cache
        print(
            "Loaded "
            f"{cache['n_reference']} overlap reference samples from rest posterior "
            f"({region}, mass={mass:.2f})."
        )
        return cache


    def cloud_sample(self, n: int, scaling_factor: float = 0.1) -> TensorLike:
        """
        Generate `n` additional parameter samples using cloud sampling.

        Handles fixed parameters (min == max) by not sampling those. The constant
        values are inserted at the correct indices in the sampled tensor.

        Parameters
        ----------
        n: int
            The number of samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        TensorLike
            A tensor of sampled (and potentially constant) parameters [n, in_dim].
        """
        assert isinstance(self.nroy_samples, TensorLike)

        bounds = self.generate_param_bounds(self.nroy_samples, buffer_ratio=0.0)
        assert bounds is not None

        # Identify constant parameters
        min_vals = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        max_vals = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        is_constant = min_vals == max_vals
        constant_params = {
            i: min_vals[i].item() for i, fixed in enumerate(is_constant) if fixed
        }
        sample_params_idx = [i for i, fixed in enumerate(is_constant) if not fixed]

        # If all parameters are constant just return the constant sample n times
        if len(sample_params_idx) == 0:
            msg = "All parameters are constant, cannot sample from them."
            raise ValueError(msg)

        # Only use non-constant parameters for mean and covariance to sample from
        nroy_params_to_sample = self.nroy_samples[:, sample_params_idx]
        stdev = (
                        nroy_params_to_sample.max(dim=0).values
                        - nroy_params_to_sample.min(dim=0).values
                ) * scaling_factor
        # covariance_matrix = torch.diag(stdev ** 2)

        # Shuffle the order of means to sample from
        num_means = nroy_params_to_sample.shape[0]
        perm = torch.randperm(num_means, device=nroy_params_to_sample.device)

        # Determine how many samples to draw for each mean, handle remainder
        min_samples_per_mean = n // num_means
        remainder_to_sample = n % num_means

        # Determine number of parallel jobs
        n_jobs = 64  # use all cores

        # Split permuted means into batches
        chunk_size = math.ceil(num_means / n_jobs)
        batches = [nroy_params_to_sample[perm][i:i + chunk_size] for i in range(0, num_means, chunk_size)]

        # precompute once outside the loop:
        low_all = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        high_all = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        low = low_all[sample_params_idx]
        high = high_all[sample_params_idx]
        std = stdev  # already [d_nonconst]

        # Precompute these once (outside sample_batch)
        sample_idx_t = torch.tensor(sample_params_idx, device=self.device, dtype=torch.long)

        if constant_params:
            const_idx_t = torch.tensor(list(constant_params.keys()), device=self.device, dtype=torch.long)
            const_vals_t = torch.tensor(list(constant_params.values()), device=self.device, dtype=low.dtype)
        else:
            const_idx_t, const_vals_t = None, None

        param_dim = len(bounds)

        def sample_batch(batch, batch_idx):
            outs = []
            for j, mean in enumerate(batch):
                i = batch_idx * chunk_size + j
                n_samples = min_samples_per_mean + (1 if i < remainder_to_sample else 0)

                x_nonconst = self.truncated_normal_1d(mean, std, low, high, n_samples)  # [n_samples, d_nonconst]

                full = torch.empty((n_samples, param_dim), device=self.device, dtype=x_nonconst.dtype)
                if const_idx_t is not None:
                    full[:, const_idx_t] = const_vals_t.to(x_nonconst.dtype)
                full[:, sample_idx_t] = x_nonconst

                outs.append(full)

            # print(f"==============Batch {batch_idx + 1} done")
            return torch.cat(outs, dim=0) if outs else torch.empty((0, param_dim), device=self.device)

        results = Parallel(n_jobs=n_jobs)(
            delayed(sample_batch)(batch, idx) for idx, batch in enumerate(batches)
        )
        print(f"==============Batch done")
        return torch.cat(results, dim=0)


    def cloud_sample_and_emulator(self, n: int, scaling_factor: float = 0.1) -> TensorLike:
        """
        Generate `n` additional parameter samples using cloud sampling.

        Handles fixed parameters (min == max) by not sampling those. The constant
        values are inserted at the correct indices in the sampled tensor.

        Parameters
        ----------
        n: int
            The number of samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        TensorLike
            A tensor of sampled (and potentially constant) parameters [n, in_dim].
        """
        rest_reference = self._load_rest_overlap_reference()
        overlap_idx_t = torch.tensor(self.overlap_idx, device=self.device, dtype=torch.long)
        overlap_reference_samples = rest_reference["samples"]
        low_overlap = rest_reference["low"]
        high_overlap = rest_reference["high"]

        if overlap_reference_samples.shape[0] == 0 and len(self.overlap_idx) > 0:
            raise ValueError("No overlap reference samples available for exercise initialisation.")

        overlap_mode = (self.rest_overlap_sampling or "empirical").lower()
        if overlap_mode == "empirical":
            num_reference = overlap_reference_samples.shape[0]
            if num_reference >= n:
                sampled_idx = torch.randperm(num_reference, device=self.device)[:n]
            else:
                sampled_idx = torch.randint(num_reference, (n,), device=self.device)
            overlap_samples = overlap_reference_samples[sampled_idx]
            print("==============Overlap empirical sampling done")
        elif overlap_mode == "cloud":
            stdev_overlap = (
                overlap_reference_samples.max(dim=0).values
                - overlap_reference_samples.min(dim=0).values
            ) * scaling_factor

            num_means = overlap_reference_samples.shape[0]
            perm = torch.randperm(num_means, device=self.device)
            min_samples_per_mean = n // num_means
            remainder_to_sample = n % num_means

            n_jobs = 64
            chunk_size = math.ceil(num_means / n_jobs)
            batches = [
                overlap_reference_samples[perm][i:i + chunk_size]
                for i in range(0, num_means, chunk_size)
            ]
            n_overlap = len(self.overlap_idx)

            def sample_overlap_batch(batch, batch_idx):
                outs = []
                for j, mean in enumerate(batch):
                    i = batch_idx * chunk_size + j
                    ns = min_samples_per_mean + (1 if i < remainder_to_sample else 0)
                    x_sampled = self.truncated_normal_1d(
                        mean, stdev_overlap, low_overlap, high_overlap, ns
                    )
                    outs.append(x_sampled)

                return (
                    torch.cat(outs, dim=0)
                    if outs
                    else torch.empty((0, n_overlap), device=self.device)
                )

            results_overlap = Parallel(n_jobs=n_jobs)(
                delayed(sample_overlap_batch)(batch, idx)
                for idx, batch in enumerate(batches)
            )
            overlap_samples = torch.cat(results_overlap, dim=0)
            print("==============Overlap cloud sampling done")
        else:
            raise ValueError(
                f"Unknown rest_overlap_sampling='{self.rest_overlap_sampling}'. "
                "Use 'empirical' or 'cloud'."
            )

        # --- Step 2: Uniform sample the EXERCISE-ONLY parameters ---
        exercise_only_idx_t = torch.tensor(self.exercise_only_idx, device=self.device, dtype=torch.long)
        uniform_all = self.simulator.sample_inputs(n).to(self.device)  # [n, in_dim]
        exercise_only_samples = uniform_all[:, exercise_only_idx_t]  # [n, n_exercise_only]
        print(f"==============Exercise-only uniform sampling done")

        # --- Step 3: Assemble full parameter tensor ---
        # Start with nominal/fixed values for ALL parameters
        # Use the midpoint of the simulator range for non-calibrated params
        all_param_names = list(self.simulator.parameters_range.keys())
        nominal = torch.tensor(
            [0.5 * (self.simulator.parameters_range[p][0] + self.simulator.parameters_range[p][1])
             for p in all_param_names],
            device=self.device, dtype=overlap_samples.dtype
        )
        full_samples = nominal.unsqueeze(0).expand(n, -1).clone()  # [n, in_dim]

        # Insert overlap columns
        full_samples[:, overlap_idx_t] = overlap_samples

        # Insert exercise-only columns
        full_samples[:, exercise_only_idx_t] = exercise_only_samples

        print(f"==============Full assembly done, shape: {full_samples.shape}")
        # --- Adhoc diagnostic plot: parameter ranges in full_samples ---
        # self._plot_parameter_ranges(full_samples, all_param_names)

        return full_samples

    def _plot_parameter_ranges(self, full_samples: TensorLike, all_param_names: list[str]) -> None:
        """
        Adhoc diagnostic plot showing the [min, max] range of each parameter
        across the sampled tensor, colour-coded by parameter type.

        Blue   = overlap (cloud-sampled from rest NROY)
        Orange = exercise-only (uniform prior)
        Grey   = fixed / nominal (not calibrated)

        Each parameter's range is normalised to [0, 1] relative to the simulator's
        prior bounds so that all parameters are visually comparable on the same axis.
        A bar spanning the full width means the samples cover the entire prior;
        a narrow bar means the parameter is tightly constrained.
        """
        samples_np = full_samples.detach().cpu().numpy()
        sample_mins = samples_np.min(axis=0)
        sample_maxs = samples_np.max(axis=0)

        # Get simulator prior bounds for normalisation
        prior_lo = np.array([self.simulator.parameters_range[p][0] for p in all_param_names])
        prior_hi = np.array([self.simulator.parameters_range[p][1] for p in all_param_names])
        prior_range = prior_hi - prior_lo
        prior_range[prior_range == 0] = 1.0  # avoid div-by-zero for truly fixed params

        # Normalise sample min/max into [0, 1] relative to prior
        norm_mins = (sample_mins - prior_lo) / prior_range
        norm_maxs = (sample_maxs - prior_lo) / prior_range

        # Build colour array
        overlap_set = set(self.overlap_idx)
        exercise_only_set = set(self.exercise_only_idx)
        n_params = len(all_param_names)

        colors = []
        for i in range(n_params):
            if i in overlap_set:
                colors.append("#1f77b4")  # blue
            elif i in exercise_only_set:
                colors.append("#ff7f0e")  # orange
            else:
                colors.append("#999999")  # grey

        # Split into pages of 40 params each for readability
        params_per_page = 40
        n_pages = math.ceil(n_params / params_per_page)

        for page in range(n_pages):
            start = page * params_per_page
            end = min(start + params_per_page, n_params)
            idx_slice = list(range(start, end))
            n_show = len(idx_slice)

            fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.35)))

            y_positions = np.arange(n_show)
            for k, i in enumerate(idx_slice):
                bar_left = norm_mins[i]
                bar_width = norm_maxs[i] - norm_mins[i]
                ax.barh(
                    y_positions[k],
                    width=bar_width,
                    left=bar_left,
                    height=0.7,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=0.8,
                )

            ax.set_yticks(y_positions)
            ax.set_yticklabels([all_param_names[i] for i in idx_slice], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(-0.05, 1.05)
            ax.set_xlabel("Fraction of prior range covered")
            ax.set_title(
                f"Sampled parameter ranges (params {start}\u2013{end - 1})\n"
                "Blue=overlap (rest NROY)  |  Orange=exercise-only (uniform)  |  Grey=fixed",
                fontsize=10,
            )
            ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
            ax.axvline(1, color="black", linewidth=0.5, linestyle=":")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)

            fig.tight_layout()
            fname = f"param_ranges_page_{page}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved diagnostic plot: {fname}")


    def pre_wave_train_emulators(self, n_simulations: int = 4096, refit_on_all_data: bool = False) -> None:
        """
        Pre-wave step: generate hybrid samples, run them through the simulator,
        train one emulator per output, and save them to Emulator_exercise/.

        This must be called BEFORE run_waves(). It populates train_x / train_y
        and creates the initial emulators that wave 0 will load.

        Parameters
        ----------
        n_simulations: int
            Number of samples to generate, simulate, and train emulators on.
        refit_on_all_data: bool
            Whether to refit on all accumulated data (True) or just this batch.
        """
        print("=" * 60)
        print("PRE-WAVE: Generating hybrid samples for initial emulator training")
        print("=" * 60)

        # Generate hybrid samples (overlap cloud-sampled, exercise-only uniform)
        samples = self.cloud_sample_and_emulator(n_simulations, scaling_factor=0.1)

        # Run through the simulator
        x, y = self.simulate(samples)
        print(f"PRE-WAVE: Simulator returned {x.shape[0]} valid samples out of {n_simulations}")

        # Train and save one emulator per output
        output_names_full = [
            "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
            "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
            "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
            "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
            "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
            "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff",
            "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
            "PaO2", "PaCO2"]

        for j, target_name in enumerate(output_names_full):
            print(f"\n  [{j + 1}/{len(output_names_full)}] Training emulator for {target_name}")

            if refit_on_all_data:
                X_fit = self.train_x
                Y_fit = self.train_y[:, j:j + 1]
            else:
                X_fit = x
                Y_fit = y[:, j:j + 1]

            self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)

            parent = os.path.join("Emulator_exercise", target_name)
            os.makedirs(parent, exist_ok=True)
            path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
            joblib.dump(self.emulator, path1)
            print(f"  Saved to {path1}")

        # torch.save(x, "X_train.pt")
        # torch.save(y, "Y_train.pt")

        print("=" * 60)
        print("PRE-WAVE: All emulators trained and saved to Emulator_exercise/")
        print("=" * 60)


    def _sample_within_bounds(
        self,
        dist: DistributionLike,
        bounds: dict[str, tuple[float, float]],
        n: int,
        constant_params: dict[int, float] | None = None,
        sample_params_idx: list[int] | None = None,
    ) -> list[TensorLike]:
        """
        Sample from distribution until `n` valid samples within the bounds are obtained.

        Handles constant parameters by inserting their values at the correct indices.

        Parameters
        ----------
        dist: DistributionLike
            A distribution to sample from, e.g., MultivariateNormal.
        bounds: dict[str, tuple[float, float]]
            A dictionary of [min, max] parameter bounds for each sampled parameter.
        n: int
            The number of samples to generate.
        constant_params: dict[int, float] | None
            A dictionary of constant parameter indices and their values.
        sample_params_idx: list[int]
            Indices of parameters that are not constant.

        Returns
        -------
        list[TensorLike]
            A list of valid samples that are within the bounds.
        """
        param_dim = len(bounds)
        if sample_params_idx is None:
            sample_params_idx = list(range(len(bounds)))

        valid_samples = []
        while len(valid_samples) < n:
            n_remaining = n - len(valid_samples)
            samples = dist.sample((n_remaining,))
            full = torch.empty(
                (n_remaining, param_dim),
                dtype=samples.dtype,
                device=samples.device,
            )
            if constant_params:
                const_idx = list(constant_params.keys())
                const_vals = torch.tensor(
                    list(constant_params.values()),
                    dtype=samples.dtype,
                    device=samples.device,
                )
                full[:, const_idx] = const_vals
            full[:, sample_params_idx] = samples
            valid_samples.extend([s for s in full if self._is_within_bounds(s, bounds)])
        return valid_samples


    def generate_samples(
        self, n: int, scaling_factor: float = 0.1
    ) -> tuple[TensorLike, TensorLike]:
        """
        Generate parameter samples and evaluate implausibility.

        Draw `n` samples either from the simulator min/max parameter bounds or
        using cloud sampling centered at NROY samples. Evaluate sample
        implausability using emulator predictions.

        Parameters
        ----------
        n: int
            The number of parameter samples to generate.
        scaling_factor: float
            The standard deviation of the Gaussian used in cloud sampling is
            set to: `parameter range * scaling_factor`.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausability scores.
        """
        # Generate `n` parameter samples (use simulator if have no NROY samples)
        if self.nroy_samples is None:
            test_x = self.cloud_sample_and_emulator(n, scaling_factor).to(self.device)
            parent = "Emulator_exercise"
        else:
            test_x = self.cloud_sample(n, scaling_factor).to(self.device)
            parent = "Emulator_exercise_wave"

        output_names = [
            "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
            "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
            "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
            "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
            "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
            "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff",
            "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
            "PaO2", "PaCO2"]
        models = {}
        for name in output_names:
            folder = name
            path1 = os.path.join(parent, folder, f"GaussianProcessMatern32_{name}_best.joblib")
            models[name] = joblib.load(path1)

        # means = {}
        # variances = {}
        #
        # for name in output_names:
        #     target_emulator = models[name]
        #     with torch.no_grad():
        #         means[name], variances[name] = target_emulator.predict_mean_and_variance(
        #             test_x[:, self.parameter_idx]
        #         )

        n_jobs = len(output_names)
        use_raw_model = self.nroy_samples is None

        def predict_one_output(name, X):
            if use_raw_model:
                target_emulator = models[name].model
            else:
                target_emulator = models[name]

            mean, var = target_emulator.predict_mean_and_variance(X)
            return name, mean, var

        results = Parallel(n_jobs=n_jobs)(
            delayed(predict_one_output)(name, test_x[:, self.parameter_idx]) for name in output_names)

        means = {name: mean for name, mean, var in results}
        variances = {name: var for name, mean, var in results}

        mean_tensor = torch.cat([means[name].reshape(-1, 1) for name in output_names], dim=1)
        var_tensor = torch.cat([variances[name].reshape(-1, 1) for name in output_names], dim=1)

        assert var_tensor is not None
        impl_scores = self.calculate_implausibility(mean_tensor, var_tensor)

        # Filter non-physiological emulator predictions before NROY selection:
        # col 13 = Min_LA_Volume > Vu_la (param 201), col 9 = Min_RA_Volume > Vu_ra (param 203)
        phys_mask = (
                (mean_tensor[:, 13] > test_x[:, 201])
                & (mean_tensor[:, 9] > test_x[:, 203])
        )
        test_x = test_x[phys_mask]
        mean_tensor = mean_tensor[phys_mask]
        impl_scores = impl_scores[phys_mask]

        mask = self._create_nroy_mask(impl_scores)

        min_col_13 = mean_tensor[mask, 13].min()
        min_col_17 = mean_tensor[mask, 17].min()
        min_col_18 = mean_tensor[mask, 18].min()

        print("min mean_tensor[:,13] where impl_score < 3:", min_col_13.item())
        print("min mean_tensor[:,17] where impl_score < 3:", min_col_17.item())
        print("min mean_tensor[:,18] where impl_score < 3:", min_col_18.item())

        # print("Done")

        return test_x, impl_scores

    def sample_tensor(self, n: int, x: TensorLike) -> TensorLike:
        """
        Randomly sample `n` rows from `x`.

        Parameters
        ----------
        n: int
            The number of samples to draw.
        x: TensorLike
            The tensor to sample from.

        Returns
        -------
        TensorLike
            A tensor of samples with `n` rows.
        """
        if x.shape[0] < n:
            warnings.warn(
                f"Number of tensor rows {x.shape[0]} is less than {n} samples.",
                stacklevel=2,
            )
        idx = torch.randperm(x.shape[0], device=self.device)[:n]
        return x[idx]

    def simulate(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Simulate `x` parameter inputs and filter out failed simulations.

        Parameters
        ----------
        x: TensorLike
            A tensor of parameters to simulate [n_samples, n_inputs].

        Returns
        -------
        tuple[TensorLike, TensorLike]
            Tensors of succesfully simulated input parameters and predictions.
        """
        # if simulation fails, returned y and x have fewer rows than input x
        y, x = self.simulator.forward_batch(x)

        y = y.to(self.device)
        x = x.to(self.device)

        # Drop output columns
        cols_to_drop = torch.tensor([11, 14, 17, 20, 27, 30], device=self.device)
        keep_mask = torch.ones(y.shape[1], dtype=torch.bool, device=self.device)
        keep_mask[cols_to_drop] = False
        y = y[:, keep_mask]

        # Remove non-finite rows first
        finite_mask = torch.isfinite(y).all(dim=1)
        x = x[finite_mask]
        y = y[finite_mask]

        # 3-sigma outlier filter (columnwise)
        col_mean = y.mean(axis=0)
        col_std = y.std(axis=0)
        within = (y >= (col_mean - 3 * col_std)) & (y <= (col_mean + 3 * col_std))
        row_mask = within.all(axis=1)
        x = x[row_mask, :]
        y = y[row_mask, :]

        within = (y >= (col_mean - 3 * col_std)) & (y <= (col_mean + 3 * col_std))
        row_mask = within.all(dim=1)

        x = x[row_mask]
        y = y[row_mask]

        # self.train_y = torch.cat([self.train_y, y], dim=0)
        # self.train_x = torch.cat([self.train_x, x], dim=0)
        self.train_y = y
        self.train_x = x

        return x, y

    def refit_emulator(self, x: TensorLike, y: TensorLike) -> None:
        """
        Refit the emulator on the provided data.

        Parameters
        ----------
        x: TensorLike
            Tensor of input data to refit the emulator on.
        y: TensorLike
            Tensor of output data to refit the emulator on.
        """

        # create test and train data
        n = x.shape[0]
        g = torch.Generator(device=x.device)
        g.manual_seed(42)
        perm = torch.randperm(n, generator=g, device=x.device)

        n_test = max(1, int(round(0.2 * n)))
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        # Create a fresh model with the same configuration
        self.emulator = TransformedEmulator(
            x_train.float(),
            y_train.float(),
            model=get_emulator_class(self.result.model_name),
            x_transforms=self.result.x_transforms,
            y_transforms=self.result.y_transforms,
            device=self.device,
            **self.result.params,
        )

        self.emulator.fit(x_train, y_train)
        # with torch.no_grad():
        #     y_pred = self.emulator.predict_mean(x.float())  # uses transforms internally
        # r2 = evaluate(y_pred, y.float(), r2_metric())
        # print("R² test:", float(r2))

        (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
            self.emulator,
            x_test.float(),
            y_test.float(),
            n_bootstraps=100,  # or None for single split behaviour (if supported)
            device=self.device,
        )

        print(f"R² test: {r2_mean:.4f} (±{r2_std:.4f}) | RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})")
        # y_pred, variance = self.emulator.predict_mean_and_variance(x)
        # y_np = y.detach().cpu().numpy().reshape(-1)
        # y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        # r2 = r2_score(y_np, y_pred_np)
        # print(y_pred[:5,:])
        # print(f"R² = {r2:.4f}")

    def run(
        self,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
        refit_emulator: bool = True,
        refit_on_all_data: bool = True,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Run a wave of the history matching workflow.

        Parameters
        ----------
        n_simulations: int
            Number of simulations to run.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausability of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.
        refit_emulator: bool
            Whether to refit the emulator at the end of the run. Defaults to True.
        refit_on_all_data: bool
            Whether to refit the emulator on all available data or just the data
            available from the most recent simulation run. Defaults to True.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores from
            which simulation samples were then selected.
        """

        msg = (
            f"Running history matching wave with {n_simulations} simulations and "
            f"{n_test_samples} test samples"
        )
        logger.debug(msg)

        test_parameters_list, impl_scores_list, nroy_parameters_list = (
            [],
            [],
            [torch.empty((0, self.simulator.in_dim), device=self.device)],
        )

        retries = 0
        while torch.cat(nroy_parameters_list, 0).shape[0] < n_simulations:
            if retries == max_retries:
                msg = (
                    f"Could not generate n_simulations ({n_simulations}) samples "
                    f"that are NROY after {max_retries} retries. "
                    f"Only {torch.cat(nroy_parameters_list, 0).shape[0]} "
                    "samples generated."
                )
                raise Warning(msg)
                break

            if retries > 10:
                scaling_factor = 0.2

            # Generate `n_test_samples` with implausability scores, identify NROY
            test_parameters, impl_scores = self.generate_samples(
                n_test_samples, scaling_factor
            )

            # print("done getting the impl_score")
            # test parameters is a concatenation of every parameter set from before
            nroy_parameters = self.get_nroy(impl_scores, test_parameters)

            # print("done getting the nroy from self.get_nroy")

            # Store results (test_parameters_list will have as many entries as 200000 * no. of retries)
            nroy_parameters_list.append(nroy_parameters)
            test_parameters_list.append(test_parameters)
            impl_scores_list.append(impl_scores)

            msg = (
                f"Generated {nroy_parameters.shape[0]} NROY samples on try "
                f"{retries + 1}, have {torch.cat(nroy_parameters_list, 0).shape[0]} "
                f"total NROY samples so far."
            )
            logger.debug(msg)

            retries += 1

        # # Next time that call run(), will sample using these NROY points
        # self.nroy_samples = torch.cat(nroy_parameters_list, 0)

        # Randomly pick at most `n_simulations` parameters from NROY to simulate
        # nroy_simulation_samples = self.sample_tensor(n_simulations, self.nroy_samples)
        # pick `n_simulations` parameters from NROY with lowest implausibility
        nroy_params = torch.cat(nroy_parameters_list, dim=0)

        implaus_tensor = torch.cat(impl_scores_list, 0)
        nroy_impl = self.get_nroy(implaus_tensor, implaus_tensor)

        # Rank by worst-output implausibility per sample
        max_impl_per_sample, _ = nroy_impl.max(dim=1)
        best_idx = torch.argsort(max_impl_per_sample)
        nroy_params_sorted = nroy_params[best_idx]

        # Take the best n_simulations to run through the simulator
        nroy_simulation_samples = nroy_params_sorted[:n_simulations]

        # Also update nroy_samples so cloud sampling next wave uses best seeds
        self.nroy_samples = nroy_params_sorted

        # np.save("check.npy", nroy_simulation_samples)
        # save on CPU so it's portable across machines/devices
        torch.save(self.nroy_samples.detach().cpu(), "nroy_samples.pt")
        # A = np.load("check.npy")[64:66]
        # print(A[:,-4:])
        # A = torch.from_numpy(A)

        # Make predictions using simulator (this updates self.x_train and self.y_train)
        x, y = self.simulate(nroy_simulation_samples)
        print(f"WAVE: Simulator returned {x.shape[0]} valid samples out of {n_simulations}")

        output_names_full = [
            "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
            "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
            "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
            "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
            "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
            "Max_LA_Pressure_Mitral_Opening", "LA_Contraction_Volume_diff", "RA_Contraction_Volume_diff",
            "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
            "PaO2", "PaCO2"]

        for j, target_name in enumerate(output_names_full):
            # print("\n" + "=" * 100)
            # print(f"[{j + 1}/{len(output_names_full)}] Target = {target_name}")
            # print("=" * 100)

            # Optionally refit the emulator using the most recent simulations or all data
            if refit_emulator:
                # data_msg = "all data" if refit_on_all_data else "most recent data"
                # msg = f"Refitting emulator on {data_msg}."
                # logger.info(msg)
                if refit_on_all_data:
                    X_fit = self.train_x
                    Y_fit = self.train_y[:, j:j+1]
                    self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)
                else:
                    X_fit = x
                    Y_fit = y[:, j:j+1]
                    self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)

            parent = os.path.join("Emulator_exercise_wave", target_name)
            os.makedirs(parent, exist_ok=True)

            path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
            joblib.dump(self.emulator, path1)

        # torch.save(x, f"X_train_wave_{(len(self.wave_results) - 1)}_exercise_.pt")
        # torch.save(y, f"Y_train_wave_{(len(self.wave_results) - 1)}_exercise_.pt")

        # Return test parameters and impl scores for this run/wave
        return torch.cat(test_parameters_list, 0), torch.cat(impl_scores_list, 0)

    def run_waves(
        self,
        n_waves: int = 5,
        frac_nroy_stop: float = 0.9,
        n_simulations: int = 100,
        n_test_samples: int = 10000,
        max_retries: int = 3,
        scaling_factor: float = 0.1,
        refit_emulator_on_last_wave: bool = True,
        refit_on_all_data: bool = True,
        resume_wave: bool = False,
    ) -> list[tuple[TensorLike, TensorLike]]:
        """
        Run multiple waves of the history matching workflow.

        Refits the emulator after each wave (except the last), using all available data.

        Parameters
        ----------
        n_waves: int
            The maximum number of waves to run.
        frac_nroy_stop: float
            Fraction of NROY samples to stop at. If less than this fraction of
            NROY samples is reached, the workflow stops.
        n_simulations: int
            Number of simulations to run in each wave.
        n_test_samples: int
            Number of input parameters to test for implausibility with the emulator.
            Parameters to simulate are sampled from this NROY subset.
        max_retries: int
            Maximum number of times to try to generate `n_simulations` NROY parameters.
            That is the maximum number of times to repeat the following steps:
                - draw `n_test_samples` parameters (use cloud sampling if possible)
                - use emulator to make predictions for those parameters
                - score implausibility of parameters given predictions
                - identify NROY parameters within this set
        scaling_factor: float
            The standard deviation of the Gaussian to sample from in cloud sampling is
            set to: `parameter range * scaling_factor`.
        refit_emulator_on_last_wave: bool
            Whether to refit the emulator after the last wave. Defaults to True.
        refit_on_all_data: bool
            Whether to refit the emulator on all available data after each wave
            or just the data from the most recent simulation run. Defaults to True.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        if resume_wave == True:
            self.nroy_samples = torch.load("nroy_samples.pt", map_location="cpu").to(self.device)
            last_wave = int(torch.load("last_wave.pt", map_location="cpu"))
            start_i = last_wave + 1
            print(start_i)
        else:
            start_i = 0

        self.wave_results = []
        for i in range(start_i, n_waves):
            if i == 1:
                self.threshold = 3
            if i == 2:
                self.threshold = 3

            logger.info("Running history matching wave %d/%d", i + 1, n_waves)
            refit_emulator = i != n_waves - 1 or refit_emulator_on_last_wave
            test_x, impl_scores = self.run(
                n_simulations=n_simulations,
                n_test_samples=n_test_samples,
                max_retries=max_retries,
                scaling_factor=scaling_factor,
                refit_emulator=refit_emulator,
                refit_on_all_data=refit_on_all_data,
            )

            if len(test_x) < n_simulations or len(impl_scores) < n_simulations:
                msg = (
                    f"Not enough parameters or impl scores generated in wave {i + 1}"
                    f"/{n_waves}. Stopping history matching workflow. Results are "
                    f"stored until wave {i}/{n_waves}."
                )
                logger.warning(msg)
                break

            self.wave_results.append((test_x, impl_scores))
            self.plot_wave((len(self.wave_results) - 1), fname=f"200000_wave_{(len(self.wave_results) - 1)}_exercise.png")

            # Get NROY points from impl scores and check fraction
            nroy_x = self.get_nroy(impl_scores, test_x)
            nroy_frac = nroy_x.shape[0] / test_x.shape[0]
            logger.info(
                "Wave %d/%d: NROY fraction is %.2f%%",
                i + 1,
                n_waves,
                nroy_frac * 100,
            )

            torch.save(int(i), "last_wave.pt")

            if nroy_frac > frac_nroy_stop:
                logger.info(
                    "Stopping history matching workflow at wave %d/%d "
                    "with NROY fraction %.2f%% > %.2f%%",
                    i + 1,
                    n_waves,
                    nroy_frac * 100,
                    frac_nroy_stop * 100,
                )
                break

        return self.wave_results

    def plot_run(
        self,
        test_parameters: TensorLike,
        impl_scores: TensorLike,
        set_simulator_axis_limits: bool = True,
        ref_val: dict[str, float] | None = None,
        title: str = "History Matching Results",
        fname: str | None = None,
    ) -> None | Figure:
        """
        Plot results of a single history matching run.

        Parameters
        ----------
        test_parameters: TensorLike
            A tensor of tested input parameters [n_samples, n_inputs].
        impl_scores: TensorLike
            A tensor of implausibility scores for the tested input parameters.
        set_simulator_axis_limits: bool
            Whether to keep the simulator parameter ranges as axis limits.
        ref_val:dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        title: str
            Title for the plot.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        test_parameters_plausible = self.get_nroy(impl_scores, test_parameters)
        impl_scores_plausible = self.get_nroy(impl_scores, impl_scores)

        df = pd.DataFrame(
            test_parameters_plausible[:, self.parameter_idx],
            columns=self.calibration_params,  # pyright: ignore[reportArgumentType]
        )
        df["Implausibility"] = impl_scores_plausible.cpu().numpy().mean(axis=1)
        g = sns.PairGrid(df, vars=self.calibration_params, corner=True)

        norm = Normalize(
            vmin=df["Implausibility"].min(),  # pyright: ignore[reportArgumentType]
            vmax=df["Implausibility"].max(),  # pyright: ignore[reportArgumentType]
        )
        cmap = plt.cm.get_cmap("viridis")

        # added
        n_params = len(self.calibration_params)
        ncols = 4
        nrows = int(np.ceil(n_params / ncols))

        plt.rcParams.update({
            "font.size": 26,  # base font size
            "axes.titlesize": 28,  # subplot title
            "axes.labelsize": 26,  # axis labels
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "axes.linewidth": 2.5,
            "lines.linewidth": 2.2,
        })

        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4.5 * nrows), sharey=True)
        axes = axes.flatten()

        for i, param in enumerate(self.calibration_params):
            ax = axes[i]
            x = df[param].to_numpy()
            y = df["Implausibility"].to_numpy()

            # Compute 2D density using Gaussian KDE
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]  # sort for clean layering

            sc = ax.scatter(x, y, c=z, cmap=cmap, s=15, alpha=0.8)

            if ref_val is not None and param in ref_val:
                ax.axvline(ref_val[param], color="red", linestyle="--", label="True value")

            if set_simulator_axis_limits:
                ax.set_xlim(self.simulator.parameters_range[param])

            ax.set_xlabel(param)
            if i % ncols == 0:
                ax.set_ylabel("Implausibility")
            ax.grid(True, linestyle="--", alpha=0.3)

        # Hide unused subplots if parameter count not divisible by ncols
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # cbar = fig.colorbar(sc, ax=axes, shrink=0.7, label="Point density")

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if fname is None:
            return display_figure(fig)
        fig.savefig(fname, bbox_inches="tight")
        return None

        # def scatter_continuous(x, y, **kwargs):
        #     ax = plt.gca()
        #     sc = ax.scatter(
        #         x,
        #         y,
        #         c=df.loc[x.index, "Implausibility"],
        #         cmap=cmap,
        #         norm=norm,
        #         s=15,
        #         alpha=0.7,
        #     )
        #     # Set axis limits if available
        #     if set_simulator_axis_limits:
        #         ax.set_xlim(self.simulator.parameters_range[x.name])
        #         ax.set_ylim(self.simulator.parameters_range[y.name])
        #     return sc
        #
        # def diag_hist(x, **kwargs):
        #     ax = plt.gca()
        #     sns.histplot(x, kde=False, color="gray", ax=ax)
        #     # Set axis limits if available
        #     if set_simulator_axis_limits:
        #         ax.set_xlim(self.simulator.parameters_range[x.name])
        #
        # g.map_lower(scatter_continuous)
        # g.map_diag(diag_hist)
        #
        # # Add reference points
        # if ref_val is not None:
        #     for i, parami in enumerate(self.calibration_params):
        #         for j, paramj in enumerate(self.calibration_params):
        #             if j < i:  # lower triangle only
        #                 ax = g.axes[i, j]
        #                 ax.scatter(
        #                     ref_val[paramj],
        #                     ref_val[parami],
        #                     color="white",
        #                     s=60,
        #                     edgecolor="black",
        #                     marker="X",
        #                     zorder=5,
        #                     label=(
        #                         "True value"
        #                         if (i == len(self.calibration_params) - 1 and j == 0)
        #                         else None
        #                     ),
        #                 )
        #
        # # Colorbar
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # plt.colorbar(sm, ax=plt.gcf().axes, shrink=0.7, label="Implausibility")
        #
        # # Global legend (handles all subplots)
        # handles, labels = g.axes[-1, 0].get_legend_handles_labels()
        # g.fig.legend(handles, labels, loc="upper right", frameon=True)
        # g.fig.suptitle(title, fontsize=16)
        #
        # if fname is None:
        #     return display_figure(g.fig)
        # g.savefig(fname, bbox_inches="tight")
        # return None

    def plot_wave(
        self,
        wave: int,
        set_simulator_axis_limits: bool = True,
        ref_val: dict[str, float] | None = None,
        fname: str | None = None,
    ) -> None | Figure:
        """
        Plot results for a specific wave.

        Parameters
        ----------
        wave: int
            The wave number to plot (0-indexed).
        set_simulator_axis_limits: bool
            Whether to keep the simulator parameter ranges as axis limits.
        ref_val: dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        test_parameters, impl_scores = self.get_wave_results(wave)
        return self.plot_run(
            test_parameters,
            impl_scores,
            set_simulator_axis_limits,
            ref_val,
            f"Results for Wave {wave}",
            fname,
        )

    def get_wave_results(self, wave: int) -> tuple[TensorLike, TensorLike]:
        """
        Get results for a specific wave.

        Parameters
        ----------
        wave: int
            The wave number to get results for (0-indexed).

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        assert self.wave_results, "No wave results, run `run_waves()` first."
        assert 0 <= wave < len(self.wave_results), f"Wave {wave} not available."

        return self.wave_results[wave]

    def plot_wave_evolution(
        self, param, ref_val: dict[str, float] | None = None, fname: str | None = None
    ) -> None | Figure:
        """
        Plot evolution of parameter distributions across all waves.

        Parameters
        ----------
        param: str
            The parameter to plot the evolution for.
        ref_val: dict[str, float] | None
            Optional dictionary of true parameter values to mark on the plots.
        fname: str | None
            Optional filename to save the plot to. If None, the plot is displayed.

        Returns
        -------
        None | Figure
            If `fname` is provided, saves the plot to the file and returns None.
            If `fname` is None, displays the plot and returns the plot figure.
        """
        all_df = []
        for wave_idx, (test_parameters, impl_scores) in enumerate(self.wave_results):
            test_parameters_plausible = self.get_nroy(impl_scores, test_parameters)
            impl_scores_plausible = self.get_nroy(impl_scores, impl_scores)

            # Create DataFrame
            df = pd.DataFrame(
                test_parameters_plausible[:, self.parameter_idx],
                columns=self.calibration_params,  # pyright: ignore[reportArgumentType]
            )
            df["Implausibility"] = impl_scores_plausible.mean(axis=1)  # pyright: ignore[reportCallIssue]
            df["Wave"] = wave_idx

            all_df.append(df)

        # Concatenate all waves into a single DataFrame
        result_df = pd.concat(all_df, ignore_index=True)

        fig = plt.figure(figsize=(8, 5))
        sns.boxplot(data=result_df, x="Wave", y=param)

        # Add horizontal line at true value
        if ref_val is not None:
            plt.axhline(
                ref_val[param],
                color="red",
                linestyle="--",
                linewidth=2,
                label="True value",
            )

        plt.title(f"Distribution of {param} by Wave")
        plt.xlabel("Wave")
        plt.ylabel(param)
        plt.tight_layout()

        # Add global legend only once (first plot)
        plt.legend(loc="upper right", frameon=True)

        if fname is None:
            return display_figure(fig)
        plt.savefig(f"{param}_wave_evolution.png", dpi=300, bbox_inches="tight")
        return None
