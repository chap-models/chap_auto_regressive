"""Feature extraction and scaling.

Two concerns live here:

- [`get_series`][chap_ar.transforms.get_series] turns a CHAP ``DataSet`` into the
  dense ``(features, target)`` arrays the network consumes.
- [`ZScaler`][chap_ar.transforms.ZScaler] standardizes those features so that no
  single covariate dominates training.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@dataclass
class ZScaler:
    """Standardizes features to zero mean and unit variance.

    The scaler stores the per-feature mean and standard deviation estimated from
    the training data and applies ``(x - mu) / std`` when called. Standardizing
    keeps features such as rainfall (hundreds of millimetres) and the day-of-year
    position (between 0 and 1) on a comparable scale, which helps the optimizer.

    Attributes:
        mu: Per-feature means, broadcast over locations and time.
        std: Per-feature standard deviations, broadcast over locations and time.
    """

    mu: np.ndarray
    std: np.ndarray

    def __call__(self, x: tuple) -> tuple:
        """Standardize the feature array in a ``(features, ar_target)`` pair.

        Args:
            x: A tuple whose first element is the feature array to scale; any
                remaining elements (e.g. the auto-regressive target) are passed
                through unchanged.

        Returns:
            The same tuple with its feature array standardized.
        """
        i = 0
        return x[:i] + ((x[i] - self.mu) / self.std,) + x[i + 1 :]

    @classmethod
    def from_data(cls, data_set: Any) -> "ZScaler":
        """Fit a scaler from a dataset's feature statistics.

        Args:
            data_set: A [`DataSet`][chap_ar.data_loader.DataSet] whose
                ``predictors(0)`` array provides the features.

        Returns:
            A ``ZScaler`` holding the mean and standard deviation of those
            features over the location and time axes.
        """
        return ZScaler(np.mean(data_set.predictors(0), axis=(0, 1)), np.std(data_set.predictors(0), axis=(0, 1)))


def get_series(data: DataSet[FullData]) -> tuple[np.ndarray, np.ndarray]:
    """Extract dense feature and target arrays from a CHAP dataset.

    For every location the function stacks four features per period — rainfall,
    mean temperature, population, and the day-of-year position — into a
    ``(periods, features)`` matrix, and collects the observed ``disease_cases``
    as the target when present.

    Args:
        data: A CHAP ``DataSet`` of ``FullData`` series, one per location.

    Returns:
        A ``(x, y)`` tuple where ``x`` has shape ``(locations, periods, 4)`` and
        ``y`` has shape ``(locations, periods)``. ``y`` is empty when the input
        carries no ``disease_cases`` (i.e. future data).

    Raises:
        AssertionError: If any feature value is NaN.
    """
    x = []
    y = []
    for series in data.values():
        year_position = [year_position_from_datetime(period.start_timestamp.date) for period in series.time_period]
        x.append(np.array((series.rainfall, series.mean_temperature, series.population, year_position)).T)  # type: ignore
        if hasattr(series, "disease_cases"):
            y.append(series.disease_cases)
    assert not np.any(np.isnan(x))
    return np.array(x), np.array(y)


def year_position_from_datetime(dt: datetime) -> float:
    """Return the within-year position of a date as a fraction in ``[0, 1]``.

    This gives the network a simple, continuous seasonal signal: 1 January is
    near 0 and 31 December is near 1.

    Args:
        dt: The date to convert.

    Returns:
        The day of the year divided by 365.
    """
    day = dt.timetuple().tm_yday
    return day / 365
