"""Feature extraction and scaling.

The model consumes a single tidy :class:`pandas.DataFrame` with one row per
location and time period and the columns ``location``, ``time_period``,
``rainfall``, ``mean_temperature``, ``population`` and (for training) the target
``disease_cases``.

- [`get_series`][chap_auto_regressive.transforms.get_series] turns that frame into the dense
  ``(features, target)`` arrays the network consumes.
- [`ZScaler`][chap_auto_regressive.transforms.ZScaler] standardizes those features so that no
  single covariate dominates training.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ("rainfall", "mean_temperature", "population")


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
            data_set: A [`DataSet`][chap_auto_regressive.data_loader.DataSet] whose
                ``predictors(0)`` array provides the features.

        Returns:
            A ``ZScaler`` holding the mean and standard deviation of those
            features over the location and time axes.
        """
        return ZScaler(np.mean(data_set.predictors(0), axis=(0, 1)), np.std(data_set.predictors(0), axis=(0, 1)))


def location_groups(data: pd.DataFrame) -> Iterator[tuple[Any, pd.DataFrame]]:
    """Yield each location's rows, sorted by time period.

    Locations are yielded in sorted (canonical) label order so that a location
    maps to the same embedding index regardless of the input row order — this
    keeps ``train`` and ``predict`` aligned and matches the legacy model's
    ``DataSet`` ordering. Within each location the rows are sorted by
    ``time_period`` (lexicographic order is chronological for both the monthly
    ``YYYY-MM`` and the weekly ``start/end`` formats).

    Args:
        data: The input frame with a ``location`` and ``time_period`` column.

    Yields:
        ``(location, sub_frame)`` pairs.
    """
    for location, sub in data.groupby("location", sort=True):
        yield location, sub.sort_values("time_period")


def get_series(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract dense feature and target arrays from the input frame.

    For every location the function stacks four features per period — rainfall,
    mean temperature, population, and the day-of-year position — into a
    ``(periods, features)`` matrix, and collects the observed ``disease_cases``
    as the target when the column is present.

    Args:
        data: A tidy frame with one row per location and time period.

    Returns:
        A ``(x, y)`` tuple where ``x`` has shape ``(locations, periods, 4)`` and
        ``y`` has shape ``(locations, periods)``. ``y`` is empty when the frame
        carries no ``disease_cases`` column (i.e. future data).

    Raises:
        AssertionError: If any feature value is NaN.
    """
    has_target = "disease_cases" in data.columns
    xs = []
    ys = []
    for _location, sub in location_groups(data):
        year_position = [year_position_from_period(period) for period in sub["time_period"]]
        xs.append(
            np.array(
                (
                    sub["rainfall"].to_numpy(),
                    sub["mean_temperature"].to_numpy(),
                    sub["population"].to_numpy(),
                    year_position,
                )
            ).T
        )
        if has_target:
            ys.append(sub["disease_cases"].to_numpy())
    x = np.array(xs)
    assert not np.any(np.isnan(x))
    return x, np.array(ys)


def period_start_date(period: str) -> datetime:
    """Return the start date of a CHAP time-period string.

    Args:
        period: A monthly period (``"YYYY-MM"``) or a weekly range
            (``"YYYY-MM-DD/YYYY-MM-DD"``).

    Returns:
        The first day of the period as a ``datetime``.
    """
    text = str(period)
    if "/" in text:
        return datetime.fromisoformat(text.split("/")[0])
    return pd.Period(text).start_time.to_pydatetime()


def year_position_from_period(period: str) -> float:
    """Return the within-year position of a period start as a fraction in ``[0, 1]``.

    This gives the network a simple, continuous seasonal signal: January is near 0
    and December is near 1.

    Args:
        period: A CHAP time-period string.

    Returns:
        The day of the year of the period's start divided by 365.
    """
    return period_start_date(period).timetuple().tm_yday / 365
