import logging
import multiprocessing
import warnings
import numpy as np
from scipy.stats import gaussian_kde
import os
import joblib
from joblib.externals.loky import get_reusable_executor
from autoemulate.core.model_selection import evaluate, r2_metric
from autoemulate.core.model_selection import bootstrap
from joblib.externals.loky import get_reusable_executor
import gc
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

EMULATOR_OUTPUT_NAMES = [
    "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
    "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
    "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
    "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
    "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
    "Max_LA_Pressure_Mitral_Opening", "LA_Pre_Atrial_Contraction_Volume", "RA_Pre_Atrial_Contraction_Volume",
    "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
    "PaO2", "PaCO2",
]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _cleanup_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _resolve_emulator_model(model_or_name):
    """Accept both string model names and callable/class model references."""
    if isinstance(model_or_name, str):
        return get_emulator_class(model_or_name)
    if not isinstance(model_or_name, type):
        return model_or_name.__class__
    return model_or_name

def _transformed_emulator_kwargs(result, device):
    """Build constructor kwargs from either an AutoEmulate Result or TransformedEmulator."""
    model_or_name = getattr(result, "model_name", None)
    if not isinstance(model_or_name, str):
        model_or_name = getattr(result, "model", result)
        if not isinstance(model_or_name, str) and not isinstance(model_or_name, type):
            model_or_name = model_or_name.__class__

    params = getattr(result, "params", None)
    if params is None:
        params = getattr(result, "model_params", None)
    if params is None:
        params = {}

    return {
        "model": _resolve_emulator_model(model_or_name),
        "x_transforms": getattr(result, "x_transforms", None),
        "y_transforms": getattr(result, "y_transforms", None),
        "device": device,
        **params,
    }

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

    @staticmethod
    def _exercise_output_names() -> list[str]:
        return [
            "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
            "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
            "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
            "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
            "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
            "Max_LA_Pressure_Mitral_Opening", "LA_Pre_Atrial_Contraction_Volume", "RA_Pre_Atrial_Contraction_Volume",
            "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
            "PaO2", "PaCO2"]

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
        if self.rank == 1:
            rank_values = implausibility.max(dim=1).values
        else:
            # rank-th largest is the (n_outputs - rank + 1)-th value in ascending order.
            kth = implausibility.shape[1] - self.rank + 1
            rank_values = implausibility.kthvalue(kth, dim=1).values

        return rank_values <= self.threshold

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

        # Calculate implausibility
        return torch.abs(self.obs_means - pred_means) / torch.sqrt(Vs)

    @staticmethod
    def _safe_ratio_denominator(denominator: TensorLike, eps: float = 1e-8) -> TensorLike:
        # Prevent divide-by-zero when max and min are very close.
        eps_tensor = torch.full_like(denominator, eps)
        signed_eps = torch.where(denominator < 0, -eps_tensor, eps_tensor)
        return torch.where(denominator.abs() < eps, signed_eps, denominator)

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
        atrial_ratio_bounds: tuple[float, float] | None = None,
        atrial_ratio_min_probability: float = 0.0,
        atrial_ratio_mc_samples: int = 128,
        device: DeviceLike | None = None,
        random_seed: int | None = None,
        log_level: str = "debug",
        run_dir: str = ".",
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
        atrial_ratio_bounds: tuple[float, float] | None
            Optional acceptable range for the derived atrial contraction ratio.
        atrial_ratio_min_probability: float
            Minimum predictive probability required for the ratio to lie in range.
        atrial_ratio_mc_samples: int
            Monte Carlo samples used to propagate emulator uncertainty to the ratio.
        device: DeviceLike | None
            The device to use. If None, the default torch device is returned.
        random_seed: int | None
            Optional random seed for reproducibility. If None, no seed is set.
        log_level: str
            The logging level to use. One of: "debug", "info", "warning", "error",
            "critical", "progress_bar" (default).
        run_dir: str
            Directory where generated NROY samples, emulator files, and wave
            artifacts are saved and loaded.
        """
        super().__init__(observations, threshold, model_discrepancy, rank, device)
        self.simulator = simulator
        if random_seed is not None:
            set_random_seed(seed=random_seed)
        self.logger, self.progress_bar = get_configured_logger(log_level)
        self.run_dir = os.path.abspath(os.fspath(run_dir))
        os.makedirs(self.run_dir, exist_ok=True)

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
        self._current_wave_idx: int | None = None
        self._last_completed_wave_idx: int | None = None
        self._save_wave_artifacts = True
        self._wave_artifacts_dir = self.run_dir
        self._last_wave_train_points: TensorLike | None = None
        self._last_rejected_simulation_points: TensorLike | None = None
        self._generate_samples_artifact_idx = 0

        # Save names and indices of parameters to calibrate
        self.calibration_params = calibration_params or list(
            simulator.parameters_range.keys()
        )
        self.parameter_idx = [
            self.simulator.get_parameter_idx(param) for param in self.calibration_params
        ]
        # Derived atrial ratio is enforced as an interval constraint, not a point target.
        self.atrial_ratio_bounds = atrial_ratio_bounds
        self.atrial_ratio_min_probability = atrial_ratio_min_probability
        self.atrial_ratio_mc_samples = atrial_ratio_mc_samples

    @staticmethod
    def _to_numpy(array: TensorLike | np.ndarray) -> np.ndarray:
        if torch.is_tensor(array):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def _run_path(self, *parts: str) -> str:
        return os.path.join(self.run_dir, *parts)

    def _resolve_artifact_dir(self, path: str) -> str:
        path = os.fspath(path)
        if os.path.isabs(path):
            return path
        if path in ("", "."):
            return self.run_dir
        return self._run_path(path)

    def _resolve_output_path(self, path: str) -> str:
        path = os.fspath(path)
        if os.path.isabs(path):
            return path
        return self._run_path(path)

    def _wave_number(self) -> int | None:
        if self._current_wave_idx is None:
            return None
        return self._current_wave_idx + 1

    def _save_wave_numpy_artifacts(
        self,
        test_x: TensorLike,
        impl_scores: TensorLike,
    ) -> None:
        if not self._save_wave_artifacts:
            return

        wave_number = self._wave_number()
        if wave_number is None:
            return

        os.makedirs(self._wave_artifacts_dir, exist_ok=True)
        nroy_mask = self._create_nroy_mask(impl_scores)
        nroy_points = test_x[nroy_mask]

        np.save(
            os.path.join(self._wave_artifacts_dir, f"test_params_wave_{wave_number}.npy"),
            self._to_numpy(test_x),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"impl_scores_wave_{wave_number}.npy"),
            self._to_numpy(impl_scores),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"nroy_mask_wave_{wave_number}.npy"),
            self._to_numpy(nroy_mask),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"nroy_points_wave_{wave_number}.npy"),
            self._to_numpy(nroy_points),
        )

        if self._last_wave_train_points is not None:
            np.save(
                os.path.join(self._wave_artifacts_dir, f"train_points_wave_{wave_number}.npy"),
                self._to_numpy(self._last_wave_train_points),
            )

    def _save_emulator_prediction_artifacts(
        self,
        test_x: TensorLike,
        mean_tensor: TensorLike,
        output_names: list[str],
    ) -> None:
        if not self._save_wave_artifacts:
            return

        wave_number = self._wave_number()
        if wave_number is None:
            return

        self._generate_samples_artifact_idx += 1
        attempt_number = self._generate_samples_artifact_idx

        os.makedirs(self._wave_artifacts_dir, exist_ok=True)
        suffix = f"wave_{wave_number}_attempt_{attempt_number}"

        np.save(
            os.path.join(self._wave_artifacts_dir, f"emulator_test_x_pre_filter_{suffix}.npy"),
            self._to_numpy(test_x),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"emulator_mean_predictions_pre_filter_{suffix}.npy"),
            self._to_numpy(mean_tensor),
        )
        np.save(
            os.path.join(self._wave_artifacts_dir, f"emulator_output_names_{suffix}.npy"),
            np.asarray(output_names),
        )

    def _estimate_ratio_interval_probability(
        self,
        min_mean: TensorLike,
        min_var: TensorLike,
        max_mean: TensorLike,
        max_var: TensorLike,
        pre_mean: TensorLike,
        pre_var: TensorLike,
    ) -> TensorLike:
        if self.atrial_ratio_bounds is None:
            return torch.ones_like(min_mean)

        lower, upper = self.atrial_ratio_bounds
        n_mc = self.atrial_ratio_mc_samples
        chunk_size = _env_int("HM_RATIO_MC_CHUNK_SIZE", 32_768)
        probabilities = torch.empty_like(min_mean)

        for start in range(0, min_mean.shape[0], chunk_size):
            end = min(start + chunk_size, min_mean.shape[0])
            min_slice = min_mean[start:end]
            max_slice = max_mean[start:end]
            pre_slice = pre_mean[start:end]

            # Sample from emulator marginals and estimate P(lower <= ratio <= upper).
            min_draws = min_slice[:, None] + min_var[start:end].clamp(min=0).sqrt()[:, None] * torch.randn(
                min_slice.shape[0], n_mc, device=min_slice.device, dtype=min_slice.dtype
            )
            max_draws = max_slice[:, None] + max_var[start:end].clamp(min=0).sqrt()[:, None] * torch.randn(
                max_slice.shape[0], n_mc, device=max_slice.device, dtype=max_slice.dtype
            )
            pre_draws = pre_slice[:, None] + pre_var[start:end].clamp(min=0).sqrt()[:, None] * torch.randn(
                pre_slice.shape[0], n_mc, device=pre_slice.device, dtype=pre_slice.dtype
            )

            denom = self._safe_ratio_denominator(max_draws - min_draws)
            ratio = (pre_draws - min_draws) / denom
            in_band = (max_draws > min_draws) & (ratio >= lower) & (ratio <= upper)
            probabilities[start:end] = in_band.float().mean(dim=1)

        return probabilities

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

        # Only use non-constant parameters for means and bounds to sample from.
        nroy_params_to_sample = self.nroy_samples[:, sample_params_idx]
        stdev = (
                        nroy_params_to_sample.max(dim=0).values
                        - nroy_params_to_sample.min(dim=0).values
                ) * scaling_factor

        # Shuffle the order of means to sample from
        num_means = nroy_params_to_sample.shape[0]
        perm = torch.randperm(num_means, device=nroy_params_to_sample.device)

        # Determine how many samples to draw for each mean, handle remainder
        min_samples_per_mean = n // num_means
        remainder_to_sample = n % num_means

        # Determine number of parallel jobs
        n_jobs = 64 # multiprocessing.cpu_count()  # use all cores

        # Split permuted means into batches
        chunk_size = math.ceil(num_means / n_jobs)
        batches = [nroy_params_to_sample[perm][i:i + chunk_size] for i in range(0, num_means, chunk_size)]

        # precompute once outside the loop:
        low_all = torch.tensor([b[0] for b in bounds.values()], device=self.device)
        high_all = torch.tensor([b[1] for b in bounds.values()], device=self.device)
        low = low_all[sample_params_idx]
        high = high_all[sample_params_idx]
        std = torch.clamp(stdev, min=1e-12)

        # Precompute these once (outside sample_batch)
        sample_idx_t = torch.tensor(sample_params_idx, device=self.device, dtype=torch.long)

        if constant_params:
            const_idx_t = torch.tensor(list(constant_params.keys()), device=self.device, dtype=torch.long)
            const_vals_t = torch.tensor(list(constant_params.values()), device=self.device, dtype=low.dtype)
        else:
            const_idx_t, const_vals_t = None, None

        param_dim = len(bounds)
        eps = 1e-7

        def _phi(x):
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        def _phi_inv(u):
            return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

        def truncated_normal_1d(mean, std, low, high, n_samples):
            """
            mean/std/low/high: [d]
            returns: [n_samples, d] all within [low, high]
            """
            eps = 1e-7
            std = torch.clamp(std, min=1e-12)

            a = (low - mean) / std
            b = (high - mean) / std

            pa = torch.clamp(_phi(a), eps, 1 - eps)
            pb = torch.clamp(_phi(b), eps, 1 - eps)

            # sample uniformly between CDF(low) and CDF(high)
            u = torch.rand((n_samples, mean.numel()), device=mean.device, dtype=mean.dtype)
            u = pa + u * (pb - pa)
            u = torch.clamp(u, eps, 1 - eps)

            z = _phi_inv(u)
            x = mean + std * z

            # numerical safety
            return torch.clamp(x, low, high)


        def sample_batch(batch, batch_idx):
            outs = []
            for j, mean in enumerate(batch):
                i = batch_idx * chunk_size + j
                n_samples = min_samples_per_mean + (1 if i < remainder_to_sample else 0)

                x_nonconst = truncated_normal_1d(mean, std, low, high, n_samples)  # [n_samples, d_nonconst]

                full = torch.empty((n_samples, param_dim), device="cpu", dtype=x_nonconst.dtype)
                if const_idx_t is not None:
                    full[:, const_idx_t] = const_vals_t.to(x_nonconst.dtype)
                full[:, sample_idx_t] = x_nonconst

                outs.append(full)

            # print(f"==============Batch {batch_idx + 1} done")
            return torch.cat(outs, dim=0) if outs else torch.empty((0, param_dim), device="cpu")

        results = Parallel(n_jobs=n_jobs)(
            delayed(sample_batch)(batch, idx) for idx, batch in enumerate(batches)
        )
        get_reusable_executor().shutdown(wait=True)
        print(f"==============Batch done")
        return torch.cat(results, dim=0)


    def pre_wave_train_emulators(self, n_simulations: int = 4096, refit_on_all_data: bool = False) -> None:
        """
        Pre-wave step: generate hybrid samples, run them through the simulator,
        train one emulator per output, and save them under run_dir/Emulator_exercise_only/.

        This must be called BEFORE run_waves(). It populates train_x / train_y
        and creates the initial emulators that wave 0 will load.

        Parameters
        ----------
        n_simulations: int
            Number of samples to generate, simulate, and train emulators on.
        refit_on_all_data: bool
            Whether to refit on all accumulated data (True) or just this batch.
        """

        x_train_path = "X_train.pt"
        y_train_path = "Y_train.pt"
        if os.path.exists(x_train_path) and os.path.exists(y_train_path):
            print("=" * 60)
            print(f"PRE-WAVE: Loading existing {x_train_path} and {y_train_path}")
            print("=" * 60)
            x = torch.load(x_train_path, map_location=self.device)
            y = torch.load(y_train_path, map_location=self.device)
        else:
            print("=" * 60)
            print("PRE-WAVE: Generating hybrid samples for initial emulator training")
            print("=" * 60)

            samples = self.simulator.sample_inputs(n_simulations).to(self.device)
            self._current_wave_idx = 0

            x, y = [], []
            for chunk in samples.split(2048):
                x_chunk, y_chunk = self.simulate(chunk)
                x.append(x_chunk)
                y.append(y_chunk)

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            torch.save(x, self._run_path("X_train.pt"))
            torch.save(y, self._run_path("Y_train.pt"))

        self.train_x = x
        self.train_y = y

        # Train and save one emulator per output
        output_names_full = self._exercise_output_names()

        def fit_one_initial_output(j, target_name, X_fit_all, Y_fit_all, parameter_idx, result, device):
            X_fit = X_fit_all
            Y_fit = Y_fit_all[:, j:j + 1]

            x_fit = X_fit[:, parameter_idx]
            n = x_fit.shape[0]
            g = torch.Generator(device=x_fit.device)
            g.manual_seed(42)
            perm = torch.randperm(n, generator=g, device=x_fit.device)

            n_test = max(1, int(round(0.2 * n)))
            x_train, y_train = x_fit[perm[n_test:]], Y_fit[perm[n_test:]]
            x_test, y_test = x_fit[perm[:n_test]], Y_fit[perm[:n_test]]

            emulator = TransformedEmulator(
                x_train.float(),
                y_train.float(),
                **_transformed_emulator_kwargs(result, device),
            )
            emulator.fit(x_train, y_train)

            (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
                emulator,
                x_test.float(),
                y_test.float(),
                n_bootstraps=100,
                device=device,
            )

            print(
                f"[{j + 1}/{len(output_names_full)}] {target_name} "
                f"R² test: {r2_mean:.4f} (±{r2_std:.4f}) | "
                f"RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})"
            )

            parent = os.path.join("Emulator_exercise_only", target_name)
            os.makedirs(parent, exist_ok=True)
            path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
            joblib.dump(emulator, path1)
            # print(f"  Saved to {path1}")
            return target_name

        Parallel(n_jobs=25)(
            delayed(fit_one_initial_output)(
                j, target_name, self.train_x, self.train_y,
                self.parameter_idx, self.result, self.device,
            )
            for j, target_name in enumerate(output_names_full)
        )
        get_reusable_executor().shutdown(wait=True)

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
        use_raw_model = self.nroy_samples is None
        # Generate `n` parameter samples (use simulator if have no NROY samples)
        if use_raw_model:
            test_x = self.simulator.sample_inputs(n).to(self.device)
            # +/-20 %
            parent = self._run_path("Emulator_exercise_only")
            # parent = "DGSM_Exercise_Paper/HM_fifth_90_Exercise_Only/Emulator_exercise_only"
            # parent = "Emulator_initial_V_tot"
            # # +/-50%
            # parent = "Emulator_Paper_same_1000"
        else:
            test_x = self.cloud_sample(n, scaling_factor).to(self.device)
            parent = self._run_path("Emulator_exercise_only_wave")
            # parent = "Emulator_wave_V_tot"

        output_names = [
            "Heart_Rate", "Systolic_Pressure", "Diastolic_Pressure", "EDV", "ESV",
            "Max_RV_Volume", "Min_RV_Volume", "Max_RV_Pressure", "Min_RV_Pressure", "Min_RA_Volume",
            "Max_RA_Volume", "Max_RA_Pressure_Atrial_contraction",
            "Max_RA_Pressure_Tricuspid_Opening", "Min_LA_Volume",
            "Max_LA_Volume", "Max_LA_Pressure_Atrial_contraction",
            "Max_LA_Pressure_Mitral_Opening", "LA_Pre_Atrial_Contraction_Volume", "RA_Pre_Atrial_Contraction_Volume",
            "LV_Pressure_Deriv", "RV_Pressure_Deriv", "Tidal_Volume", "Minute_Ventilation",
            "PaO2", "PaCO2"]

        models = {}
        for name in output_names:
            folder = name
            path1 = os.path.join(parent, folder, f"GaussianProcessMatern32_{name}_best.joblib")
            models[name] = joblib.load(path1)

        means = {}
        variances = {}

        for name in output_names:
            target_emulator = models[name]

            with torch.no_grad():
                means[name], variances[name] = target_emulator.predict_mean_and_variance(
                    test_x[:, self.parameter_idx]
                )
           # means[name], variances[name] = target_emulator.predict_mean_and_variance(test_x[:, self.parameter_idx])


        #
        # n_jobs = len(output_names)
        # def predict_one_output(name, X):
        #     target_emulator = models[name]
        #
        #     mean, var = target_emulator.predict_mean_and_variance(X)
        #     return name, mean, var
        #
        # results = Parallel(n_jobs=n_jobs)(
        #     delayed(predict_one_output)(name, test_x[:, self.parameter_idx]) for name in output_names)
        #
        # means = {name: mean for name, mean, var in results}
        # variances = {name: var for name, mean, var in results}

        mean_tensor = torch.cat([means[name].reshape(-1, 1) for name in output_names], dim=1)
        var_tensor = torch.cat([variances[name].reshape(-1, 1) for name in output_names], dim=1)

        self._save_emulator_prediction_artifacts(test_x, mean_tensor, output_names)

        get_reusable_executor().shutdown(wait=True)

        # assert adjusted_var_tensor is not None
        impl_scores = self.calculate_implausibility(mean_tensor, var_tensor)
        if self.atrial_ratio_bounds is not None:
            la_ratio_probs = self._estimate_ratio_interval_probability(
                mean_tensor[:, 13], var_tensor[:, 13],
                mean_tensor[:, 14], var_tensor[:, 14],
                mean_tensor[:, 17], var_tensor[:, 17],
            )
            ra_ratio_probs = self._estimate_ratio_interval_probability(
                mean_tensor[:, 9], var_tensor[:, 9],
                mean_tensor[:, 10], var_tensor[:, 10],
                mean_tensor[:, 18], var_tensor[:, 18],
            )
            atrial_ratio_mask = (
                (la_ratio_probs >= self.atrial_ratio_min_probability)
                & (ra_ratio_probs >= self.atrial_ratio_min_probability)
            )
            # The ratio is now enforced through the interval-probability filter below.
            impl_scores[:, 17] = 0.0
            impl_scores[:, 18] = 0.0
        else:
            atrial_ratio_mask = torch.ones(
                mean_tensor.shape[0], dtype=torch.bool, device=mean_tensor.device
            )

        # Filter non-physiological emulator predictions before NROY selection:
        # col 13 = Min_LA_Volume > Vu_la (param 201), col 9 = Min_RA_Volume > Vu_ra (param 203)
        phys_mask = (
            (mean_tensor[:, 13] > 0)
            & (mean_tensor[:, 9] > 0)
            & (mean_tensor[:, 10] > mean_tensor[:, 9])
            & (mean_tensor[:, 14] > mean_tensor[:, 13])
            & atrial_ratio_mask
        )
        # test_x = test_x[phys_mask]
        # mean_tensor = mean_tensor[phys_mask]
        # impl_scores = impl_scores[phys_mask]

        impl_scores[~phys_mask] = 4

        mask = self._create_nroy_mask(impl_scores)

        min_col_13 = mean_tensor[mask, 13].min()
        min_col_17 = mean_tensor[mask, 17].min()
        min_col_18 = mean_tensor[mask, 18].min()

        print("min mean_tensor[:,13] where impl_score < 3:", min_col_13.item())
        print("min adjusted mean_tensor[:,17] where impl_score < 3:", min_col_17.item())
        print("min adjusted mean_tensor[:,18] where impl_score < 3:", min_col_18.item())

        del mean_tensor, var_tensor, phys_mask, mask
        _cleanup_torch_memory()
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

    def _remove_from_nroy_samples(self, rejected_x: TensorLike) -> None:
        """Remove rejected simulated parameter rows from the current NROY cloud."""
        if self.nroy_samples is None or rejected_x.numel() == 0:
            return

        keep_nroy_mask = torch.ones(
            self.nroy_samples.shape[0], dtype=torch.bool, device=self.device
        )
        rejected_x = rejected_x.to(device=self.device, dtype=self.nroy_samples.dtype)

        for rejected in rejected_x:
            matches = torch.where(
                keep_nroy_mask & torch.all(self.nroy_samples == rejected, dim=1)
            )[0]
            if matches.numel() > 0:
                keep_nroy_mask[matches[0]] = False

        self.nroy_samples = self.nroy_samples[keep_nroy_mask]

    def _mark_rejected_simulations_as_ro(
        self, test_x: TensorLike, impl_scores: TensorLike
    ) -> None:
        """Ensure simulator-rejected samples are excluded by later NROY masks."""
        rejected_x = self._last_rejected_simulation_points
        if rejected_x is None or rejected_x.numel() == 0:
            return

        rejected_x = rejected_x.to(device=test_x.device, dtype=test_x.dtype)
        ro_value = torch.as_tensor(
            self.threshold + 1.0, device=impl_scores.device, dtype=impl_scores.dtype
        )
        for rejected in rejected_x:
            matches = torch.all(test_x == rejected, dim=1)
            impl_scores[matches] = ro_value

    def simulate(self, x: TensorLike) -> tuple[TensorLike, TensorLike]:
        """
        Simulate `x` parameter inputs and filter out failed/out-of-target simulations.

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
        get_reusable_executor().shutdown(wait=True)

        y = y.to(self.device)
        x = x.to(self.device)

        # if self._current_wave_idx > 1:
        #     obs_means = self.obs_means.to(device=y.device, dtype=y.dtype)
        #     obs_stds = torch.sqrt(torch.clamp(
        #         self.obs_vars.to(device=y.device, dtype=y.dtype),
        #         min=0.0,
        #     ))
        #     if y.shape[1] != obs_means.shape[1]:
        #         raise ValueError(
        #             f"Simulation output dimension ({y.shape[1]}) does not match "
        #             f"observation target dimension ({obs_means.shape[1]})."
        #         )
        #
        #     # Keep only simulations within target +/- self.threshold observation
        #     # standard deviations. Pre-atrial volume columns are checked through
        #     # the derived atrial ratio instead.
        #     finite_mask = torch.isfinite(y).all(dim=1)
        #     min_ra, max_ra, min_la, max_la = 9, 10, 13, 14
        #     pre_la, pre_ra = 17, 18
        #     target_columns_mask = torch.ones(y.shape[1], dtype=torch.bool, device=y.device)
        #     pre_atrial_columns = torch.tensor([pre_la, pre_ra], device=y.device)
        #     target_columns_mask[pre_atrial_columns] = False
        #     target_band_mask = (
        #         (
        #             y[:, target_columns_mask]
        #             >= obs_means[:, target_columns_mask] - self.threshold * obs_stds[:, target_columns_mask]
        #         ).all(dim=1)
        #         & (
        #             y[:, target_columns_mask]
        #             <= obs_means[:, target_columns_mask] + self.threshold * obs_stds[:, target_columns_mask]
        #         ).all(dim=1)
        #     )
        #
        #     if self.atrial_ratio_bounds is not None:
        #         lower, upper = self.atrial_ratio_bounds
        #         la_ratio = (
        #             (y[:, pre_la] - y[:, min_la])
        #             / self._safe_ratio_denominator(y[:, max_la] - y[:, min_la])
        #         )
        #         ra_ratio = (
        #             (y[:, pre_ra] - y[:, min_ra])
        #             / self._safe_ratio_denominator(y[:, max_ra] - y[:, min_ra])
        #         )
        #         ratio_mask = (
        #             (la_ratio >= lower)
        #             & (la_ratio <= upper)
        #             & (ra_ratio >= lower)
        #             & (ra_ratio <= upper)
        #         )
        #     else:
        #         ratio_mask = torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
        #
        #     target_mask = finite_mask & target_band_mask & ratio_mask
        #
        #     rejected_x = x[~target_mask]
        #     self._last_rejected_simulation_points = rejected_x.detach()
        #     x = x[target_mask]
        #     y = y[target_mask]
        #     self._remove_from_nroy_samples(rejected_x)
        #
        #     if rejected_x.shape[0] > 0:
        #         logger.info(
        #             "Removed %d simulated sample(s) outside target +/- %.4g std band from "
        #             "training data and NROY samples.",
        #             rejected_x.shape[0],
        #             self.threshold,
        #         )
        #
        # else:
        # 3-sigma outlier filter (columnwise)
        col_mean = y.mean(axis=0)
        col_std = y.std(axis=0)
        within = (y >= (col_mean - 3 * col_std)) & (y <= (col_mean + 3 * col_std))
        row_mask = within.all(axis=1)
        x = x[row_mask, :]
        y = y[row_mask, :]


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
        self._last_wave_train_points = None
        self._generate_samples_artifact_idx = 0

        test_parameters_list, impl_scores_list, nroy_parameters_list = (
            [],
            [],
            [torch.empty((0, self.simulator.in_dim), device=self.device)],
        )

        retries = 0
        nroy_total = 0
        while nroy_total < n_simulations:
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
                scaling_factor = 0.05

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
            nroy_total += nroy_parameters.shape[0]

            msg = (
                f"Generated {nroy_parameters.shape[0]} NROY samples on try "
                f"{retries + 1}, have {torch.cat(nroy_parameters_list, 0).shape[0]} "
                f"total NROY samples so far."
            )
            logger.debug(msg)

            retries += 1

        # # Next time that call run(), will sample using these NROY points
        self.nroy_samples = torch.cat(nroy_parameters_list, 0)

        # Randomly pick at most `n_simulations` parameters from NROY to simulate
        nroy_simulation_samples = self.sample_tensor(n_simulations, self.nroy_samples)
        # nroy_params = torch.cat(nroy_parameters_list, dim=0)
        #
        # implaus_tensor = torch.cat(impl_scores_list, 0)
        # nroy_impl = self.get_nroy(implaus_tensor, implaus_tensor)
        #
        # # Rank by worst-output implausibility per sample
        # max_impl_per_sample, _ = nroy_impl.max(dim=1)
        # best_idx = torch.argsort(max_impl_per_sample)
        # nroy_params_sorted = nroy_params[best_idx]
        #
        # # Take the best n_simulations to run through the simulator
        # nroy_simulation_samples = nroy_params_sorted[:n_simulations]
        #
        # # Also update nroy_samples so cloud sampling next wave uses best seeds
        # self.nroy_samples = nroy_params_sorted

        # np.save("check.npy", nroy_simulation_samples)
        # A = np.load("check.npy")[64:66]
        # print(A[:,-4:])
        # A = torch.from_numpy(A)

        # Make predictions using simulator (this updates self.x_train and self.y_train)
        x, y = self.simulate(nroy_simulation_samples)
        #
        # # Keep only simulations whose outputs are within self.threshold observation
        # # standard deviations of the observation means. `obs_vars` stores variances.
        # obs_means = self.obs_means.to(device=y.device, dtype=y.dtype)
        # obs_stds = torch.sqrt(torch.clamp(self.obs_vars.to(device=y.device, dtype=y.dtype), min=0.0))
        # obs_3sd_mask = (
        #     (y >= obs_means - self.threshold * obs_stds)
        #     & (y <= obs_means + self.threshold * obs_stds)
        # ).all(dim=1)
        #
        # if not bool(obs_3sd_mask.all()):
        #     rejected_x = x[~obs_3sd_mask]
        #     x = x[obs_3sd_mask]
        #     y = y[obs_3sd_mask]
        #     self.train_x = x
        #     self.train_y = y
        #
        #     # Remove each rejected simulated parameter set from the NROY cloud too,
        #     # so it cannot seed samples for the next emulator wave.
        #     keep_nroy_mask = torch.ones(
        #         self.nroy_samples.shape[0], dtype=torch.bool, device=self.device
        #     )
        #     for rejected in rejected_x:
        #         matches = torch.where(
        #             keep_nroy_mask & torch.all(self.nroy_samples == rejected, dim=1)
        #         )[0]
        #         if matches.numel() > 0:
        #             keep_nroy_mask[matches[0]] = False
        #     self.nroy_samples = self.nroy_samples[keep_nroy_mask]
        #     print(
        #         f"Removed {rejected_x.shape[0]} simulated sample(s) outside "
        #         f"the observation +/- {self.threshold} std band from training and NROY samples."
        #     )

        # Save on CPU so it's portable across machines/devices. This happens after
        # filtering so resume/cloud sampling uses the updated NROY set.
        torch.save(self.nroy_samples.detach().cpu(), self._run_path("nroy_samples_exercise.pt"))
        self._last_wave_train_points = x.detach().cpu()

        output_names_full = EMULATOR_OUTPUT_NAMES
        wave_number = self._wave_number()
        wave_emulator_root = self._run_path("Emulator_exercise_only_wave")
        snapshot_root = (
            os.path.join(self._wave_artifacts_dir, f"Emulator_wave_{wave_number}")
            if self._save_wave_artifacts and wave_number is not None
            else None
        )

        def fit_one_output(j, target_name, X_fit, Y_fit, parameter_idx, result, device):
            x_fit = X_fit[:, parameter_idx]
            y_fit = Y_fit[:, j:j + 1]

            n = x_fit.shape[0]
            g = torch.Generator(device=x_fit.device)
            g.manual_seed(42)
            perm = torch.randperm(n, generator=g, device=x_fit.device)

            n_test = max(1, int(round(0.2 * n)))
            x_train, y_train = x_fit[perm[n_test:]], y_fit[perm[n_test:]]
            x_test, y_test = x_fit[perm[:n_test]], y_fit[perm[:n_test]]

            emulator = TransformedEmulator(
                x_train.float(), y_train.float(),
                model=get_emulator_class(result.model_name),
                x_transforms=result.x_transforms,
                y_transforms=result.y_transforms,
                device=device,
                **result.params,
            )
            emulator.fit(x_train, y_train)

            (r2_mean, r2_std), (rmse_mean, rmse_std) = bootstrap(
                emulator,
                x_test.float(),
                y_test.float(),
                n_bootstraps=100,  # or None for single split behaviour (if supported)
                device=device,
            )

            print(f"R² test: {r2_mean:.4f} (±{r2_std:.4f}) | RMSE test: {rmse_mean:.4f} (±{rmse_std:.4f})")

            # save
            parent = os.path.join(wave_emulator_root, target_name)
            # parent = os.path.join("Emulator_wave_V_tot", target_name)
            os.makedirs(parent, exist_ok=True)
            #######################################
            with torch.no_grad():
                y_test_emulator_mean, y_test_emulator_variance = emulator.predict_mean_and_variance(
                    x_test.float()
                )

            numpy_artifacts = {
                "x_train.npy": x_train,
                "y_train.npy": y_train,
                "x_test.npy": x_test,
                "y_test.npy": y_test,
                "y_test_emulator_mean.npy": y_test_emulator_mean,
                "y_test_emulator_variance.npy": y_test_emulator_variance,
            }
            for filename, array in numpy_artifacts.items():
                np.save(os.path.join(parent, filename), array.detach().cpu().numpy())
            #############################################
            model_filename = f"GaussianProcessMatern32_{target_name}_best.joblib"
            joblib.dump(emulator, os.path.join(parent, model_filename))
            if snapshot_root is not None:
                snapshot_parent = os.path.join(snapshot_root, target_name)
                os.makedirs(snapshot_parent, exist_ok=True)
                ############
                for filename, array in numpy_artifacts.items():
                    np.save(os.path.join(snapshot_parent, filename), array.detach().cpu().numpy())
                ############
                joblib.dump(emulator, os.path.join(snapshot_parent, model_filename))

            del (
                numpy_artifacts,
                y_test_emulator_mean,
                y_test_emulator_variance,
                x_fit,
                y_fit,
                x_train,
                y_train,
                x_test,
                y_test,
                emulator,
            )
            _cleanup_torch_memory()
            return target_name

        if refit_emulator:
            n_jobs = min(
                len(output_names_full),
                _env_int("HM_EMULATOR_TRAIN_N_JOBS", min(8, len(output_names_full))),
            )
            results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(fit_one_output)(j, target_name, x, y, self.parameter_idx, self.result, self.device)
                for j, target_name in enumerate(output_names_full)
            )
            del results
            get_reusable_executor().shutdown(wait=True)
            _cleanup_torch_memory()
        # for j, target_name in enumerate(output_names_full):
        #     # Optionally refit the emulator using the most recent simulations or all data
        #     if refit_emulator:
        #         # data_msg = "all data" if refit_on_all_data else "most recent data"
        #         # msg = f"Refitting emulator on {data_msg}."
        #         # logger.info(msg)
        #         if refit_on_all_data:
        #             X_fit = self.train_x
        #             Y_fit = self.train_y[:, j:j+1]
        #             self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)
        #         else:
        #             X_fit = x
        #             Y_fit = y[:, j:j+1]
        #             self.refit_emulator(X_fit[:, self.parameter_idx], Y_fit)
        #
        #     parent = os.path.join("Emulator_wave", target_name)
        #     os.makedirs(parent, exist_ok=True)
        #
        #     path1 = os.path.join(parent, f"GaussianProcessMatern32_{target_name}_best.joblib")
        #     joblib.dump(self.emulator, path1)

        # torch.save(x, f"X_train_wave_{(len(self.wave_results) - 1)}_rest_.pt")
        # torch.save(y, f"Y_train_wave_{(len(self.wave_results) - 1)}_rest_.pt")

        # Return test parameters and impl scores for this run/wave. Mark any
        # simulator-rejected samples as RO so saved masks/fractions also exclude them.
        test_parameters = torch.cat(test_parameters_list, 0)
        impl_scores = torch.cat(impl_scores_list, 0)
        self._mark_rejected_simulations_as_ro(test_parameters, impl_scores)
        return test_parameters, impl_scores

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
        save_wave_artifacts: bool = True,
        wave_artifacts_dir: str = ".",
        keep_all_wave_results: bool = True,
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
        save_wave_artifacts: bool
            Whether to save per-wave emulator snapshots and `.npy` artifacts.
        wave_artifacts_dir: str
            Directory where per-wave artifacts are written.
        keep_all_wave_results: bool
            Whether to retain every wave in memory. If False, only the most recent
            wave is kept while artifacts are still written to disk.

        Returns
        -------
        tuple[TensorLike, TensorLike]
            A tensor of tested input parameters and their implausibility scores.
        """
        if resume_wave == True:
            self.nroy_samples = torch.load(self._run_path("nroy_samples_exercise.pt"), map_location="cpu").to(self.device)
            last_wave = int(torch.load(self._run_path("last_wave.pt"), map_location="cpu"))
            start_i = last_wave + 1
            print(start_i)
        else:
            start_i = 0

        self.wave_results = []
        self._last_completed_wave_idx = None
        self._save_wave_artifacts = save_wave_artifacts
        self._wave_artifacts_dir = self._resolve_artifact_dir(wave_artifacts_dir)
        if self._save_wave_artifacts:
            os.makedirs(self._wave_artifacts_dir, exist_ok=True)
        for i in range(start_i, n_waves):
            # 0th wave had 155173
            # if i == 0: # 110599
            #     self.threshold = 3.5 # change
            if i == 1:  # 154081
                self.threshold = 3.5
                n_test_samples = 200000
            if i > 1: # 154081
                self.threshold = 3.0
                n_test_samples = 200000


            # if i == 1: # 110599
            #     self.threshold = 1.5
            # if i == 2: # 110599
            #     self.threshold = 1.25
            # if i == 3: # 110599
            #     self.threshold = 1.125
            # if i == 4: # 154081
            #     self.threshold = 1.0
            # if i == 5: # 49467
            #     self.threshold = 1.0
            #     n_simulations = 5000

            logger.info("Running history matching wave %d/%d", i + 1, n_waves)
            self._current_wave_idx = i
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

            if keep_all_wave_results:
                self.wave_results.append((test_x, impl_scores))
            else:
                self.wave_results = [(test_x, impl_scores)]
            # self.plot_wave((len(self.wave_results) - 1), fname=f"200000_wave_{(len(self.wave_results) - 1)}_rest.png")

            # Get NROY points from impl scores and check fraction
            self._save_wave_numpy_artifacts(test_x, impl_scores)
            nroy_x = self.get_nroy(impl_scores, test_x)
            nroy_frac = nroy_x.shape[0] / test_x.shape[0]
            logger.info(
                "Wave %d/%d: NROY fraction is %.2f%%",
                i + 1,
                n_waves,
                nroy_frac * 100,
            )

            torch.save(int(i), self._run_path("last_wave.pt"))
            self._last_completed_wave_idx = i
            del nroy_x
            _cleanup_torch_memory()

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

        self._current_wave_idx = None
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
        fig.savefig(self._resolve_output_path(fname), bbox_inches="tight")
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
        plt.savefig(self._resolve_output_path(fname), dpi=300, bbox_inches="tight")
        return None
