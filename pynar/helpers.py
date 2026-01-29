from typing import Union, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from xclim.core.units import convert_units_to
from xclim.indices.helpers import solar_declination
from xclim.core.calendar import select_time, time_bnds, get_calendar, _is_leap_year

__all__ = [
    "day_lengths",
    "select_doy",
    "mask_uncomplete_years"
]

def day_lengths( # Modified from xclim.indices.generic.day_lengths
    dates: xr.DataArray,
    lat: xr.DataArray,
    method: Literal["spencer", "simple"] = "spencer",
    altitude_angle: Union[int, float] = 0
) -> xr.DataArray:
    r"""
    Calculate day-length according to latitude and day of year. By default, the day-lengths are computed assuming
    the light is perceptible from sunrise to sunset (i.e., when sun crosses the horizon). However, the argument
    `altitude_angle` can be used to change this behaviour and compute the day-length for given altitude angle
    (in degrees) above/below the horizon.

    See :py:func:`solar_declination` for the approximation used to compute the solar declination angle.
    Based on :cite:t:`kalogirou_chapter_2014`.

    Parameters
    ----------
    dates : xr.DataArray
        Daily datetime data.
        This function makes no sense with data of other frequency.
    lat : xarray.DataArray
        Latitude coordinate.
    method : {'spencer', 'simple'}
        Which approximation to use when computing the solar declination angle.
        See :py:func:`solar_declination`.
    altitude_angle : int or float, optional
        The altitude angle (in degrees) for which the day length is computed.

    Returns
    -------
    xarray.DataArray, [hours]
        Day-lengths in hours per individual day.

    References
    ----------
    :cite:cts:`kalogirou_chapter_2014`
    """        
    declination = solar_declination(dates, method=method)
    lat = convert_units_to(lat, "rad")
    with np.errstate(invalid="ignore"):
        day_length_hours = ((24 / np.pi) * np.arccos((np.sin(np.radians(altitude_angle)) - np.sin(lat) * np.sin(declination)) / (np.cos(lat) * np.cos(declination))))
    return day_length_hours.assign_attrs(units="h")

def select_doy(
        da: xr.DataArray,
        freq: Optional[str] = "YS",
        start: Optional[Union[int, xr.DataArray]] = None,
        end: Optional[Union[int, xr.DataArray]] = None,
        include_bounds: Optional[list[bool]] = True
) -> xr.DataArray:
    """
    Improve the doys selection method of xclim 'select_time' function by automatically determining
    the start or end day of year based on the frequency of provided doys of year (doy) bounds.
    """
    if start is not None and end is not None:
        return select_time(da, doy_bounds=(start, end), include_bounds=include_bounds)
    elif start is None and end is None:
        return da
    else:
        if freq is None:
            if isinstance(start, xr.DataArray):
                freq = xr.infer_freq(start.time)
            elif isinstance(end, xr.DataArray):
                freq = xr.infer_freq(end.time)
        
        bounds = time_bnds(da.time, freq).resample(time=freq).min()
        start_doys = bounds.isel(bnds=0)
        end_doys = bounds.isel(bnds=1)

        if start_doys[0].dt.year.values != end_doys[0].dt.year.values:
            # If the start and end years are different, we need to adjust the end day of year
            # to account for the leap year.
            is_leap = _is_leap_year(start_doys.time.dt.year.values + 1, get_calendar(start_doys))
            end_doys = end_doys.where(~is_leap, end_doys - pd.Timedelta(days=1))
        
        start_doys = start_doys.dt.dayofyear
        end_doys = end_doys.dt.dayofyear

        if isinstance(start, int):
            start = xr.full_like(start_doys, start)
        elif start is None:
            start = start_doys
        
        if isinstance(end, int):
            end = xr.full_like(end_doys, end)
        elif end is None:
            end = end_doys
    # return start, end
    return select_time(da, doy_bounds=(start, end), include_bounds=include_bounds)

def mask_uncomplete_years(
    da: xr.DataArray,
    freq: Optional[str] = "YS",
    drop: Optional[bool] = True
) -> xr.DataArray:
    bounds = time_bnds(da.time, freq, "1day").resample(time=freq).min()
    
    data_start = da.time.min().values
    data_end = da.time.max().values
    freq_start = bounds.isel(bnds=0, time=0).values
    freq_end = bounds.isel(bnds=1, time=-1).values

    if data_start == freq_start and data_end == freq_end:
        return da
    
    if data_start != freq_start:
        start = bounds.isel(bnds=0, time=1).values
    else :
        start = freq_start

    if data_end != freq_end:
        end = bounds.isel(bnds=1, time=-2).values
    else :
        end = freq_end

    return da.where(
        (da.time >= start) & (da.time <= end),
        drop=drop
    )
