"""Phenology module"""
from typing import Literal, Union, Optional

import numpy as np
import xarray as xr

from pynar.helpers import day_lengths, select_doy

__all__ = [
    "photoperiod",
    "reduction_factor_photoperiod_index",
    "vernalisation_index",
    "reduction_factor_vernalisation_index",
    "crop_effective_temperature",
    "crop_development_unit"
]

def photoperiod(
    dates: xr.DataArray,
    lat: xr.DataArray,
    altitude_angle: Union[int, float] = -6, # default -6 based on STICS 
    method: Literal["spencer", "simple"] = "spencer"
) -> xr.DataArray:
    """
    Calculate the photoperiod in hours from latitude and day of year for a given altitude angle.
    
    Parameters
    ----------
    dates : xr.DataArray
        Daily datetime data.
        This function makes no sense with data of other frequency.
    lat : xarray.DataArray
        Latitude coordinate.
    altitude_angle : int or float, optional
        The altitude angle (in degrees) for which the photoperiod is computed.
        Default is -6 degrees, meaning the light is perceptible from and up to 6 degrees below the horizon (see STICS).
    method : {'spencer', 'simple'}
        Which approximation to use when computing the solar declination angle.
        See :py:func:`solar_declination`.

    Returns
    -------
    xarray.DataArray, [hours]
        Photoperiod in hours per individual day.
    """
    return day_lengths(dates, lat, method=method, altitude_angle=altitude_angle)

def reduction_factor_photoperiod_index( # rfpi
        phoi: xr.DataArray,
        sensiphot: Union[int, float], 
        phobase: Union[int, float],
        phosat: Union[int, float]
) -> xr.DataArray:
    """
    Calculate the reduction factor for photoperiod index.

    Parameters
    ----------
    phoi : xr.DataArray
        Photoperiod (hours).
    sensiphot : int or float
        Amplitude of the photoperiod sensitivity. It ranges from 0 to 1, where 0 equals the maximum
        sensitivity and 1 cancels the effect of photoperiod.
    phobase : int or float
        Value of the base photoperiod (hours).
    phosat : int or float
        Value of the saturation photoperiod (hours).
    
    Returns
    -------
    xr.DataArray
        Reduction factor for photoperiod index ranging from 0 to 1.
    """
    rfpi = (1 - sensiphot) * (phoi - phosat) / (phosat - phobase) + 1
    return rfpi.clip(min=0, max=1).rename('reduction_factor_photoperiod_index')

def vernalisation_index( # jvi
        tas: xr.DataArray,
        optimum_temp: Union[int, float], # tfroid
        thermal_sensi: Union[int, float] # ampfroid
) -> xr.DataArray:
    """
    Calculate the vernalisation index. `optimum_temp` and `thermal_semiamplitude` parameters
    define the range of vernalising activity of the temperature.

    Parameters
    ----------
    tas : xr.DataArray
        Daily mean temperature (degC).
    optimum_temp : int or float
        Optimum vernalisation temperature (degC).
    thermal_sensi : int or float
        Thermal sensivity to vernalisation (degC).

    Returns
    -------
    xr.DataArray
        Vernalisation index ranging from 0 to 1.
    """
    vi = 1 - ((optimum_temp - tas) / thermal_sensi)**2
    return vi.clip(min=0).rename('vernalisation_index')

def reduction_factor_vernalisation_index( # rfvi
        vern_index: xr.DataArray, # jvi
        vern_mindays: Union[int, float], # jvcmini
        vern_ndays: Union[int, float], # jvc
        start_doy: Optional[Union[int, xr.DataArray]] = None,
        freq = 'YS'
) -> xr.DataArray:
    """
    Calculate the reduction factor for vernalisation index.

    Parameters
    ----------
    vern_index : xr.DataArray
        Vernalisation index.
    vern_mindays : int or float
        Minimum number of vernalising days before starting vernalisation process.
    vern_ndays : int or float
        Number of vernalising days required to reach vernalisation requirements.
    start_doy : int, optional
        Start day of year for vernalisation accumulation.
    freq : str, optional
        Resampling frequency for the vernalisation index (default is 'YS' - yearly start).
    
    Returns
    -------
    xr.DataArray
        Reduction factor for vernalisation index ranging from 0 to 1.
    """
    # mask data before the start of vernalisation accumulation
    if start_doy is not None:
        vern_index = select_doy(vern_index, freq=freq, start=start_doy)
    
    rfvi = (vern_index.resample(time=freq).cumsum(dim='time') - vern_mindays) / (vern_ndays - vern_mindays)
    return rfvi.clip(min=0, max=1).rename('vernalisation_reduction_factor')

def crop_effective_temperature( # udevcult
        tas: xr.DataArray, # udevcult
        tmin_thresh: Union[int, float], # tdmin
        tmax_thresh: Union[int, float], # tdmax
        tstop_thresh: Union[int, float] # tcxstop
) -> xr.DataArray:
    """
    Calculate the crop effective temperature.

    The effect of temperature increases linearly from `tmin_thresh` to `tmax_thresh` thresholds,
    and decreases linearly from `tmax_thresh` to `tstop_thresh` thresholds. Unlike in STICS model,
    the crop effective temperature is calculated using the air temperature and not the crop temperature.

    Parameters
    ----------
    tas : xr.DataArray
        Daily mean temperature (degC).
    tmin_thresh : int or float
        Threshold temperature (degC) below which temperature has no effect on crop development.
    tmax_thresh : int or float
        Threshold temperature (degC) with maximum effect on crop development.
    tstop_thresh : int or float
        Threshold temperature (degC) above which temperature has no effect on crop development.
    
    Returns
    -------
    xr.DataArray
        Crop effective temperature.
    """
    cet = xr.where(tas.notnull(), xr.zeros_like(tas), np.nan) # <=> 0 if tas <= tmin_thresh or tas >= tstop_thresh
    cet = xr.where((tas > tmin_thresh) & (tas < tmax_thresh), tas - tmin_thresh, cet)
    cet = xr.where(
        (tas >= tmax_thresh) & (tas < tstop_thresh),
        (tmax_thresh - tmin_thresh) / (tmax_thresh - tstop_thresh) * (tas - tstop_thresh),
        cet
    )
    return cet.rename("crop_effective_temperature").assign_attrs(units='deg C')

def crop_development_unit( # upvt
        crop_temp: xr.DataArray, # udevcult
        rfpi: Optional[xr.DataArray] = None, # rfpi
        rfvi: Optional[xr.DataArray] = None # rfvi
) -> xr.DataArray:
    """
    Calculate the crop development unit.
    Simplified version of the STICS model, where the crop development stress is not considered.

    Parameters
    ----------
    crop_temp : xr.DataArray
        Crop effective temperature.
    rfpi : xr.DataArray, optional
        Reduction factor for photoperiod index.
    rfvi : xr.DataArray, optional
        Reduction factor for vernalisation index.
    
    Returns
    -------
    xr.DataArray
        Crop development unit.
    """
    if rfpi is None:
        rfpi = xr.ones_like(crop_temp, dtype=int)
    if rfvi is None:
        rfvi = xr.ones_like(crop_temp, dtype=int)
    
    cdu = crop_temp * rfpi * rfvi
    return cdu.rename("crop_development_unit").assign_attrs(units='degC d')
