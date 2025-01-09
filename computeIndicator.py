import pandas as pd 
import xarray as xr
import datetime as dt 
import numpy as np

def compute_indicator_netcdf(climate_data, NameIndicator, varName, threshold=None, two_years_culture=True, start_stage=None, end_stage=None):
    classMoy = ["mint", "maxt", "meant", "avsorad"]
    classDaySup = ["hdaystmax", "hdaystmin", "hdaystmean", "excraidays"]
    classSum = ["rainsum", "sumetp"]
    classDayInf = ["cdaystmin", "defraidays"]
    classFreqSup = ["raifreq", "excraifreq", "hsfreq"]
    classFreqInf = ["cfreqtmean", "cfreqtmin"]
   
    import xarray as xr
    import pandas as pd
    import numpy as np
    
    # Get time information
    time = climate_data.time
    
    # Vérifier si le dataset n'est pas vide
    if len(time) == 0:
        return xr.Dataset()
        
    doy = pd.DatetimeIndex(time.values).dayofyear
    
    # Create start and end arrays
    if isinstance(start_stage, (int, float)):
        rast_start = xr.full_like(climate_data[varName].isel(time=0), start_stage)
    
    if isinstance(end_stage, (int, float)):
        rast_end = xr.full_like(climate_data[varName].isel(time=0), end_stage)
    
    # Get unique years
    years = pd.DatetimeIndex(time.values).year.unique()
    
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
                results.append(result)
                continue
                
        else:
            # Select only current year
            rast_year = climate_data.sel(time=time.dt.year==year)
            if len(rast_year.time) < 367:
                result = xr.full_like(climate_data[varName].isel(time=0), np.nan)
                results.append(result)
                continue
                
        if isinstance(start_stage, xr.DataArray):
            rast_start = start_stage.sel(time=time.dt.year==year)
            
        if isinstance(end_stage, xr.DataArray):
            rast_end = end_stage.sel(time=time.dt.year==year)
            
        # Apply calculation
        if NameIndicator in classMoy:
            result = xr.apply_ufunc(
                lambda x, start, end: np.mean(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)]),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)

        elif NameIndicator in classDaySup:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] > threshold),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
                
        elif NameIndicator in classSum:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)]),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
                
        elif NameIndicator in classDayInf:
            result = xr.apply_ufunc(
                lambda x, start, end: np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] < threshold),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
       
        elif NameIndicator in classFreqSup:
            result = xr.apply_ufunc(
                lambda x, start, end: (np.sum(x[(np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)] > threshold) * 100) / np.sum((np.arange(len(x)) >= start-1) & (np.arange(len(x)) <= end-1)),
                rast_year[varName],
                rast_start,
                rast_end,
                input_core_dims=[['time'], [], []],
                vectorize=True)
            #result = result.assign_coords(year=year)    
        #result = result.assign_coords(year=year)
        results.append(result)

     
        
    # Combine results
    if results:
        output = xr.concat(results, dim="year").to_dataset()
        output = output.rename({varName: NameIndicator})
        return output
    else:
        return xr.Dataset()
