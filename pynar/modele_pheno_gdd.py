import pandas as pd 
import xarray as xr
import numpy as np


def load_parameters (TCXSTOP=None,TDMIN=None,TDMAX=None,SENSIPHOT=None,PHOBASE=None,PHOSTAT=None,TFROID=None,AMPFROID=None,JVCMINI=None ,jvc=None):
       return TCXSTOP ,TDMIN,TDMAX,SENSIPHOT,PHOBASE,PHOSTAT,TFROID,AMPFROID,JVCMINI,jvc
    
def upvt (udevecult,RFPI,RFVI):
    UPVT = udevecult*RFPI*RFVI
    return(UPVT)

def JVI (x,TFROID,AMPFROID):
    jvi = 1 - ((TFROID - x) / AMPFROID)**2
    jvi = jvi.where(jvi > 0, 0)
    return jvi

def photoperiode_calc(zlat, jday):
    maxjd = 365
    alat = zlat/57.296
    zday = jday
    
    # Correction des jours > 365
    zday = np.where(zday > maxjd, zday - 365, zday)
    
    theta1 = 2*np.pi*(zday-80)/365
    theta2 = 0.034*(np.sin((2*np.pi*zday)/365)-np.sin(2*np.pi*80/365))
    theta = theta1 + theta2
    dec = np.arcsin(0.3978 * np.sin(theta))
    x1 = np.cos(alat) * np.cos(dec)
    b = -0.01454 / x1
    x2 = np.tan(alat) * np.tan(dec)
    h = np.arccos(b - x2)
    daylen = 24.0 * h / np.pi
    dl = daylen
    d = -1.0 * 0.10453 / (np.cos(alat) * np.cos(dec))
    p = d - (np.tan(alat) * np.tan(dec))
    p = np.arccos(p)
    photp = 24 * p / np.pi
    
    # Limite à 24h max
    photp = np.minimum(photp, 24)
    
    return photp 



def RFPI_cal (x,SENSIPHOT,PHOSTAT,PHOBASE):
    rfpi =(1 - SENSIPHOT) * (x - PHOSTAT) / (PHOSTAT - PHOBASE) + 1
    rfpi = np.where(rfpi>1,1,rfpi)
    rfpi = np.where(rfpi<SENSIPHOT,SENSIPHOT,rfpi)
    return rfpi


def udevult_comput(x,TDMIN,TDMAX,TCXSTOP):
    
    X=x
    X=np.where(X>0,X,0)
    condition1 = (X > TDMIN) & (X < TDMIN)
    condition2 = (X >= TDMAX) & (X < TCXSTOP)
    condition3 = (X >= TCXSTOP)
    udevc=X
    # Application des conditions
    udevc = np.where(condition1, X - TDMIN, udevc)  # Modifie les valeurs où la condition1 est vraie
    udevc = np.where(condition2, (TDMAX - TDMIN) / (TDMAX - TCXSTOP) * (X - TCXSTOP), udevc)  # Modifie les valeurs où la condition2 est vraie
    udevc = np.where(condition3, 0, udevc) 

    return(udevc)
def RFVI_cal(JVCMINI, jvc, JVI):
    rfvi = (np.cumsum(JVI, axis=0) - JVCMINI) / (jvc - JVCMINI)  # Cumul sur l'axe time
    rfvi = np.clip(rfvi, 0, 1)  # Équivalent à np.where()
    return rfvi


def compute_indice_BBCH(TmoyYear,GDD_BBCH,rfvi,rfpi,TDMIN,TDMAX,TCXSTOP):
    
     
    udevecult = udevult_comput(TmoyYear["tasAdjust"],TDMIN,TDMAX,TCXSTOP)
    UPVT_cell = udevecult * rfvi * rfpi
    UPVTCumul = UPVT_cell.cumsum(dim="time")
    stade = (UPVTCumul > GDD_BBCH).argmax(dim="time")
    stade = stade.where((UPVTCumul > GDD_BBCH).any(dim="time"))

def selection_stage(Tmoy,start_stage,end_stage,latitude,longitude):
    data_compute = xr.apply_ufunc(
        lambda x, start, end: np.where(
        (np.arange(len(x)) < start) | (np.arange(len(x)) > end), 0, x
        ),
        Tmoy,  # Xarray contenant les données de température
        start_stage.stage,  # Xarray contenant les indices de début (start)
        end_stage.stage,    # Xarray contenant les indices de fin (end)
        input_core_dims=[["time"], [], []],  # "time" est la dimension de rast_year
        output_core_dims=[["time"]],  # Le résultat doit conserver la dimension "time"
        vectorize=True,  # Appliquer la fonction à chaque cellule (y, x)
        dask="parallelized",  # Utiliser Dask si nécessaire pour des données volumineuses
        output_dtypes=[np.float32]  # Le type de la sortie (par exemple, float32)
    )
    data_compute = data_compute.transpose("time",latitude,longitude)
    data_compute = data_compute.where(~np.isnan(start_stage.stage))  
    return(data_compute)

def proccess_all_year (tmean,stade, two_years_culture,GDD,latitude,longitude,vernalisation,photoperiode,params,varName):
    TCXSTOP, TDMIN, TDMAX, SENSIPHOT, PHOBASE, PHOSTAT, TFROID, AMPFROID, JVCMINI, jvc = params
    time = tmean.time

    years = pd.DatetimeIndex(time.values).year.unique()
    results = []
    for year in years:
        print(year)
    # Vérifier si on prend une ou deux années
        if two_years_culture:
        # Sélectionner l'année actuelle et l'année précédente
            mask = ((tmean.time.dt.year == year) | (tmean.time.dt.year == year - 1))
            rast_year = tmean.sel(time=mask)
        
        # Vérification du nombre de jours
            if rast_year.sizes["time"] < 367:  # Vérifie la dimension time
                result = xr.full_like(tmean[varName].isel(time=0), np.nan)
                result = result.drop_vars("time")  # Supprimer la dimension temps
                result = result.assign_coords(year=year)
                results.append(result)
                continue
                
        else:
        # Sélectionner uniquement l'année actuelle
            rast_year = tmean.sel(time=tmean.time.dt.year == year)

    # Sélection des périodes de stade de croissance
        start_stage = stade.sel(year=year)
        end_stage = xr.full_like(start_stage, 730)
    
        jours_annee = rast_year["time"].dt.dayofyear 
    #"""Fonction appliquée sur chaque cellule"""
        if photoperiode:
            photoP = xr.apply_ufunc(
                photoperiode_calc,
                rast_year[latitude],                # Raster de latitude (2D: y, x)
                jours_annee,   # On applique aux temps de la cellule
                vectorize=True
            )
            photoP = photoP.transpose("time", latitude)
            photoP = photoP.expand_dims(x=rast_year[longitude].values, axis=-1)
        
            rfpi = xr.apply_ufunc(
                RFPI_cal,
                photoP,
                SENSIPHOT, PHOBASE, PHOSTAT,
            )
        else :
            rfpi =xr.full_like(rast_year,1)
    
        if vernalisation:
            jvi_calc = JVI(rast_year["tasAdjust"], TFROID, AMPFROID)
            JVCMINI_da = xr.full_like(jvi_calc, JVCMINI)  
            jvc_da = xr.full_like(jvi_calc, jvc)
            rfvi = xr.apply_ufunc(
                RFVI_cal,
                JVCMINI_da,  
                jvc_da,      
                jvi_calc,  
                input_core_dims=[["time", latitude, longitude], ["time", latitude, longitude], ["time", latitude, longitude]],
                output_core_dims=[["time", latitude, longitude]],
                vectorize=True
            )
        else : 
            rfvi =xr.full_like(rast_year,1)
    # Appliquer la sélection et le calcul d'indice
        data_compute = selection_stage(rast_year, start_stage, end_stage,latitude=latitude,longitude=longitude)
    
        result = compute_indice_BBCH(data_compute, GDD_BBCH=GDD,rfpi=rfpi, rfvi=rfvi,TDMIN=TDMIN,TDMAX=TDMAX,TCXSTOP=TCXSTOP)

    # Ajouter le résultat
        results.append(result)

# Concaténation des résultats par année
    if isinstance(results, xr.DataArray):
        pheno_date = xr.concat(results, dim="year").to_dataset(name="stage")
    if isinstance(results,xr.Dataset):
        pheno_date = xr.concat(results, dim="year")
        pheno_date = pheno_date.rename_vars(tasAdjust="stage")  
    return(pheno_date)