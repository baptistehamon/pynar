"""Phenology module"""
from typing import Literal, Union, Optional, Any

import numpy as np
import xarray as xr

from xclim.core.calendar import get_calendar
from xclim.core.units import convert_units_to
from xclim.indices.generic import compare
import xclim.indices.run_length as rl

from pynar.helpers import day_lengths, select_doy

__all__ = [
    "photoperiod",
    "reduction_factor_photoperiod_index",
    "vernalisation_index",
    "reduction_factor_vernalisation_index",
    "crop_effective_temperature",
    "crop_development_unit",
    "stage_doy",
    "phenological_stage"
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

def stage_doy(
    cdu: xr.DataArray,
    thresh: Union[int, float],
    from_doy: Optional[Union[int, xr.DataArray]] = None,
    freq: str = 'YS',
) -> xr.DataArray:
    """
    Calculate the doy of the phenological stage based on the crop development unit (CDU).

    Parameters
    ----------
    cdu : xr.DataArray
        Crop development unit.
    thresh : int or float
        Threshold value of the crop development unit to determine the phenological stage.
    from_doy : int or xr.DataArray, optional
        Starting day of year for CDU accumulation.
    freq : str, optional
        Resampling frequency for the phenological stage calculation (default YS).
    Returns
    -------
    xr.DataArray
        Day of year of the phenological stage.
    """
    if from_doy is not None:
        cdu = select_doy(cdu, freq=freq, start=from_doy)
    
    cumcdu = cdu.resample(time=freq).cumsum(dim='time').assign_coords(time=cdu.time)

    out = rl.resample_and_rl(
        compare(cumcdu, ">=", thresh),
        resample_before_rl=True,
        compute=rl.first_run,
        window=1,
        coord="dayofyear",
        freq=freq,
        dim="time"
    )
    out.attrs.update(
        long_name=f"Phenological stage date for {thresh} CDU",
        description=f"Day of year when the cumulative crop development unit reaches {thresh} degC d.",
        unit= "",
        is_dayofyear=np.int32(1),
        calendar= get_calendar(cumcdu)
    )
    return out


def phenological_stage(
        tas,
        thresh: int | float,
        params: dict[str, Any],
        is_photoperiod: bool | None = False,
        is_vernalisation: bool | None = False,
        start_cycle_doy: int | None = 275,
        from_doy: Optional[Union[int, xr.DataArray]] = None,
        freq: str = 'YS-OCT'

) -> xr.DataArray:
    """
    Calculate the phenological stage date (doy) based on the cumulative crop development unit (CDU).

    Parameters
    ----------
    tas : xr.DataArray
        Daily mean temperature.
    thresh : int or float
        Threshold value of the crop development unit to determine the phenological stage.
    params : dict[str, Any]
        Dictionary containing the parameters for the phenological model. Required keys:
        - 'tmin_thresh': Threshold temperature below which temperature has no effect on crop development (degC).
        - 'tmax_thresh': Threshold temperature with maximum effect on crop development (degC).
        - 'tstop_thresh': Threshold temperature above which temperature has no effect on crop development (degC).
        Optional keys if `is_photoperiod` is True:
        - 'sensiphot': Amplitude of the photoperiod sensitivity (0 to 1).
        - 'phobase': Value of the base photoperiod (hours).
        - 'phosat': Value of the saturation photoperiod (hours).
        Optional keys if `is_vernalisation` is True:
        - 'optimum_temp': Optimum vernalisation temperature (degC).
        - 'thermal_sensi': Thermal sensitivity to vernalisation (degC).
        - 'vern_mindays': Minimum number of vernalising days before starting vernalisation process.
        - 'vern_ndays': Number of vernalising days required to reach vernalisation requirements.\n
    is_photoperiod : bool, optional
        If True, include the photoperiod effect in the phenological model (default is False).
    is_vernalisation : bool, optional
        If True, include the vernalisation effect in the phenological model (default is False).
    start_cycle_doy : int, optional
        Day of year corresponding to the start of the growing cycle of the crop. Usually corresponds to the sowing date
        for annual crops and the greenup date for perennial crops (default is 275, corresponding to October 1st in Northern Hemisphere).
    from_doy : int or xr.DataArray, optional
        Starting day of year for the crop development unit (CDU) accumulation (corresponds to the previous phenological stage date). If None,
        the `start_cycle_doy` is used.
    freq : str, optional
        Resampling frequency for the phenological stage calculation. Should correspond to the month of the start of the growing season or
        'start_doy' if `is_vernalisation` is True.
        Default is 'YS-OCT' corresponding to Northern Hemisphere (set to 'YS-APR' for Southern Hemisphere).
    """
    def _check_required_params(
            params: dict,
            is_photoperiod: bool = False,
            is_vernalisation: bool = False,
    ):

        required_params = ['tmin_thresh', 'tmax_thresh', 'tstop_thresh']
        if is_photoperiod:
            required_params += ['sensiphot', 'phobase', 'phosat']
        if is_vernalisation:
            required_params += ['optimum_temp', 'thermal_sensi', 'vern_mindays', 'vern_ndays']

        if not all(param in params for param in required_params):
            missing_params = [param for param in required_params if param not in params]
            raise ValueError(f"Required parameters are missing: {', '.join(missing_params)}")
    
    # Check if required parameters are provided
    _check_required_params(params, is_photoperiod, is_vernalisation)

    tas = convert_units_to(tas, 'degC')  # Ensure temperature is in Celsius

    if is_photoperiod:
        phoi = photoperiod(tas['time'], lat=tas['lat'], altitude_angle=-6)
        rfpi = reduction_factor_photoperiod_index(phoi, **{k: params[k] for k in ['sensiphot', 'phobase', 'phosat']})
    else:
        rfpi = None
    
    if is_vernalisation:
        vi = vernalisation_index(tas, **{k: params[k] for k in ['optimum_temp', 'thermal_sensi']})
        rfvi = reduction_factor_vernalisation_index(vi, start_doy=start_cycle_doy, freq=freq, **{k: params[k] for k in ['vern_mindays', 'vern_ndays']})
    else:
        rfvi = None
    
    cet = crop_effective_temperature(tas, **{k: params[k] for k in ['tmin_thresh', 'tmax_thresh', 'tstop_thresh']})
    cdu = crop_development_unit(crop_temp=cet, rfpi=rfpi, rfvi=rfvi)

    if from_doy is None:
        from_doy = start_cycle_doy

    out = stage_doy(cdu, thresh=thresh, freq=freq, from_doy=from_doy)
    out.attrs.update(
        method= f"vernalisation={is_vernalisation}, photoperiod={is_photoperiod}"
    )
    return out
