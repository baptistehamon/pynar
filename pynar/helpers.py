from typing import Union, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr

from xclim.core.units import convert_units_to
from xclim.indices.helpers import solar_declination
from xclim.core.calendar import time_bnds, doy_to_days_since, _get_doys

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

def select_doys(
        da: xr.DataArray,
        doy_bounds: tuple[int | xr.DataArray | None, int | xr.DataArray | None],
        include_bounds: Optional[list[bool]] = True
) -> xr.DataArray:
    """
    Improve the doys selection method of xclim 'select_time'.
    """
    def mask_between_doys(
        da: xr.DataArray,
        doy_bounds: tuple[int | xr.DataArray | None, int | xr.DataArray | None],
        include_bounds: tuple[bool, bool] = (True, True),
        include_nans: bool = True,
        ) -> xr.DataArray | xr.Dataset:
        """
        Mask the data outside the day of year bounds.

        Improved version of `xclim.core.calendar.mask_between_doys` dealing with None bounds and
        allowing to choose NaN handling strategy.

        Parameters
        ----------
        da : xr.DataArray or xr.Dataset
            Input data. It must have a time coordinate.
        doy_bounds : 2-tuple of integers or DataArrays or None
            The bounds as (start, end) of the period of interest expressed in day-of-year, integers going from
            1 (January 1st) to 365 or 366 (December 31st).
            If None is passed for one of the bounds, it is considered open (the start or end of the period).
            If DataArrays are passed, they must have the same coordinates on the dimensions they share.
            They may have a time dimension, in which case the masking is done independently for each period
            defined by the coordinate, which means the time coordinate must have an inferable frequency
            (see :py:func:`xr.infer_freq`). Timesteps of the input not appearing in the time coordinate of the
            bounds are masked as "outside the bounds".
        include_bounds : 2-tuple of booleans
            Whether the bounds of `doy_bounds` should be inclusive or not.
        include_nans : bool
            Whether to include values associated with NaN day-of-year in the start or end bounds. If True,
            missing values (nan) in the start and end bounds are replaced with 1 and 366 respectively in the non-temporal
            case and to open bounds in the temporal case.

        Returns
        -------
        xr.DataArray
            Boolean array with the same time coordinate as `da` and any other dimension present on the bounds.
            True value inside the period of interest and False outside.
        """
        if (
            (isinstance(doy_bounds[0], int) or (doy_bounds[0] is None)) and
            (isinstance(doy_bounds[1], int) or (doy_bounds[1] is None))
        ):  # Simple case
            if doy_bounds[0] is None:
                doy_bounds = (1, doy_bounds[1])
            if doy_bounds[1] is None:
                doy_bounds = (doy_bounds[0], 366)
            mask = da.time.dt.dayofyear.isin(_get_doys(*doy_bounds, include_bounds))
        else:
            start, end = doy_bounds
            # Convert None to DataArrays with nans
            if start is None:
                start = xr.full_like(end, 1)
            if end is None:
                end = xr.full_like(start, 366)
            # convert ints to DataArrays
            if isinstance(start, int):
                start = xr.full_like(end, start)
            elif isinstance(end, int):
                end = xr.full_like(start, end)
            # Ensure they both have the same dims
            # align join='exact' will fail on common but different coords, broadcast will add missing coords
            start, end = xr.broadcast(*xr.align(start, end, join="exact"))

            if not include_bounds[0]:
                start += 1
            if not include_bounds[1]:
                end -= 1

            if "time" in start.dims:
                freq = xr.infer_freq(start.time)
                # Convert the doy bounds to a duration since the beginning of each period defined
                # in the bound's time coordinate.
                # Also ensures the bounds share the same time calendar as the input.
                # Any missing value is replaced with the min/max of possible values.
                calkws = {"calendar": da.time.dt.calendar, "use_cftime": (da.time.dtype == "O")}
                start = doy_to_days_since(start.convert_calendar(**calkws))
                end = doy_to_days_since(end.convert_calendar(**calkws))
                if include_nans == "include":
                    start = start.fillna(0)
                    end = end.fillna(366)

                out = []
                # For each period, mask the days since between start and end
                for base_time, indexes in da.resample(time=freq).groups.items():
                    group = da.isel(time=indexes)

                    if base_time in start.time:
                        start_d = start.sel(time=base_time)
                        end_d = end.sel(time=base_time)

                        # select days between start and end for group
                        days = (group.time - base_time).dt.days
                        days = days.where(days >= 0)
                        mask = (days >= start_d) & (days <= end_d)
                    else:  # This group has no defined bounds : put False in the mask
                        # Array with the same shape as the "mask" in the other case : broadcast of time and bounds dims
                        template = xr.broadcast(group.time.dt.day, start.isel(time=0, drop=True))[0]
                        mask = xr.full_like(template, False, dtype="bool")
                    out.append(mask)
                mask = xr.concat(out, dim="time")
            else:  # Only "Spatial" dims, we can't constrain as in days since, so there are two cases
                doys = da.time.dt.dayofyear  # for readability
                if include_nans == "include":
                    # Any missing value is replaced with the min/max of possible values
                    start = start.fillna(1)
                    end = end.fillna(366)
                mask = xr.where(
                    # case 1 : nan in start or end
                    start.isnull() | end.isnull(),
                    False,
                    xr.where(
                        start <= end,
                        # case 2 : start <= end, ROI is within a calendar year
                        (doys >= start) & (doys <= end),
                        # case 3 : start >  end, ROI crosses the new year
                        ~((doys > end) & (doys < start)),
                    ),
                )
        return mask

    if isinstance(include_bounds, bool):
        include_bounds = (include_bounds, include_bounds)
    mask = mask_between_doys(da, doy_bounds, include_bounds, include_nans=False)
    return da.where(mask)

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
