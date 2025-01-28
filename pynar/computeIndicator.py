import pandas as pd 
import xarray as xr
import numpy as np

def compute_indicator_netcdf(climate_data, NameIndicator, varName, threshold=None, two_years_culture=True, start_stage=None, end_stage=None):
    classMoy = ["mint", "maxt", "meant", "avsorad"]
    classDaySup = ["hdaystmax", "hdaystmin", "hdaystmean", "excraidays"]
    classSum = ["rainsum", "sumetp"]
    classDayInf = ["cdaystmin", "defraidays"]
    classFreqSup = ["raifreq", "excraifreq", "hsfreq"]
    classFreqInf = ["cfreqtmean", "cfreqtmin"]

    # Get time information
    time = climate_data.time

    doy = pd.DatetimeIndex(time.values).dayofyear

    years = pd.DatetimeIndex(time.values).year.unique()
       # Create start and end arrays
    if isinstance(start_stage, (int, float)):
        rast_start = xr.full_like(climate_data[varName].isel(time=0), start_stage)
    
    if isinstance(end_stage, (int, float)):
        rast_end = xr.full_like(climate_data[varName].isel(time=0), end_stage)

    # Process each year
    results = []
    for year in years:
        print(year)
        
        if two_years_culture:
            # Select current and previous year
            mask = ((time.dt.year == year) | (time.dt.year == year-1))
            rast_year = climate_data.sel(time=mask)
            
            if len(rast_year.time) < 367:
                result = xr.full_like(climate_data[varName].isel(time=0), np.nan)
                result = result.drop("time")
                result = result.assign_coords(year=year)
                results.append(result)
                continue
                
        else:
            # Select only current year
            rast_year = climate_data.sel(time=time.dt.year==year)
                    
        if isinstance(start_stage, xr.Dataset):
            rast_start = start_stage.sel(time=year)
            rast_start = rast_start.stage
        if isinstance(end_stage, xr.Dataset):
            rast_end = end_stage.sel(time=year)
            rast_end = rast_end.stage
        
        # Apply calculation
        if NameIndicator in classMoy:
            result = xr.apply_ufunc(
                lambda x, start, end: np.mean(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)]),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            result = result.assign_coords(year=year)

        elif NameIndicator in classDaySup:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] > threshold),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            result = result.assign_coords(year=year)
        elif NameIndicator in classSum:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)]),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            result = result.assign_coords(year=year)
        elif NameIndicator in classDayInf:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] < threshold),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            result = result.assign_coords(year=year)
        elif NameIndicator in classFreqSup:
            result = xr.apply_ufunc(
                lambda x, start, end: (np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] > threshold) * 100) / np.sum((np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            result = result.assign_coords(year=year)
        
        results.append(result)

         
        # Combine results
        if results:
            if len(years) == 1:
                output = xr.concat(results, dim="year").to_dataset(name=NameIndicator)
            else:
                output = xr.concat(results, dim="year").to_dataset()
                output = output.rename({varName: NameIndicator})
            return output
        else:
            return xr.Dataset()
