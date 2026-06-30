#  Import necessary libraries and functions
from .config import models
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import os
from astropy.time import Time
import matplotlib.colors as mcolors
try:
    from rubin_sim.phot_utils import PhotometricParameters, Bandpass
    from rubin_sim.phot_utils import calc_snr_m5, calc_mag_error_m5
    from rubin_scheduler.data import get_data_dir
except ImportError:
    print("Warning: rubin_sim.phot_utils or rubin_scheduler.data not found. Some LSST related functions may not work.")
from astropy.cosmology import Planck18, z_at_value
from lal import MRSUN_SI
import glob
from nmma.em.model import FiestaKilonovaModel
from nmma.em.model import SVDLightCurveModel

'''
Fitting formulae (see https://arxiv.org/abs/2205.08513)
'''

def dyn_ej(a = -9.3335, b = 114.17, d = -337.56, n = 1.5465, M1 = 1.4, R1 = 10, M2 = 1.4, R2 = 10):
    C1 = M1 / (R1 * 1e3 / MRSUN_SI)
    C2 = M2 / (R2 * 1e3 / MRSUN_SI)
    x = (a/C1 + b*(M2**n/M1**n) + d*C1)*M1 + (a/C2 + b*(M1**n/M2**n) + d*C2)*M2
    if x < 0:
        return 0
    else:
        return x/1000
    
def wind_ej(M1, M2, a0=-1.725, deltaa=-2.337, b0=-0.564, deltab=-0.437, c=0.958, d=0.057, beta=5.879, qtrans=0.886, Mtov=1.97, R16=11.137): 
    r16 = R16 * 1e3 / MRSUN_SI # convert km to SI (like in NMMA) 
    Mtresh = (2.38 - 3.606 * (Mtov/r16))*Mtov
    q = M2/M1
    xsi = 0.5 * np.tanh(beta * (q - qtrans))
    a = a0 + deltaa * xsi
    b = b0 + deltab * xsi
    mwind = a * (1 + b * np.tanh( (c - (M1+M2)/Mtresh)/d ))
    mwind = np.maximum(-3.0, mwind)
    return mwind

# mass chirp
def chirp_mass(m1, m2):
    return (m1*m2)**(3/5) / (m1 + m2)**(1/5)

# mass ratio
def mass_ratio(m1, m2):
    return m2 / m1

'''
Class to load and explore the eos files from https://zenodo.org/records/6106130#.Y1pdM9JBxhG (15nsat_cse_uniform_R14)
'''
    
class EOSDataset:
    """
    Class to load and manage EOS files from NMMA/EOS directory.
    Similar to BullaDataset but for equation of state data.
    """
    
    def __init__(self, folder_path):
        """
        Initialize the EOSDataset.
        
        Parameters:
        -----------
        folder_path : str
            Path to the folder containing the EOS .dat files
        """
        self.folder_path = folder_path
        self.files = []
        self.eos_dict = {}
        
        # Load all files
        self._load_all_files()
        
        # Compute statistics
        self._compute_statistics()
    
    def _load_all_files(self):
        """Load all .dat files from the folder (excluding Zone.Identifier files)."""
        all_files = sorted([f for f in glob.glob(os.path.join(self.folder_path, '*.dat'))
                           if not f.endswith('Zone.Identifier')],
                          key=lambda x: int(os.path.basename(x).replace('.dat', '')))
        
        print(f"Loading {len(all_files)} EOS files...")
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            eos_id = int(filename.replace('.dat', ''))
            
            self.files.append(filepath)
            self.eos_dict[eos_id] = {
                'filepath': filepath,
                'filename': filename,
                'eos_id': eos_id,
                'data': None,  # Loaded on demand
                'radius_km': None,
                'mass_solar': None,
                'pressure': None,
                'max_mass': None,
                'radius_at_1_4': None,
                'radius_at_1_6': None
            }
        
        print(f"{len(self.eos_dict)} EOS files indexed")
    
    def _load_eos_data(self, eos_id):
        """Load data for a specific EOS if not already loaded."""
        if eos_id not in self.eos_dict:
            print(f"EOS {eos_id} not found in dataset")
            return None
        
        eos_info = self.eos_dict[eos_id]
        
        if eos_info['data'] is None:
            try:
                # Load data: columns are Radius[km], Mass[Solar Mass], Central_pressure
                data = np.loadtxt(eos_info['filepath'])
                eos_info['data'] = data
                eos_info['radius_km'] = data[:, 0]
                eos_info['mass_solar'] = data[:, 1]
                eos_info['pressure'] = data[:, 2]
                
                # Compute derived quantities
                eos_info['max_mass'] = np.max(data[:, 1])
                
                # Find radius at 1.4 M☉
                try:
                    idx_1_4 = np.argmin(np.abs(data[:, 1] - 1.4))
                    eos_info['radius_at_1_4'] = data[idx_1_4, 0]
                except:
                    eos_info['radius_at_1_4'] = None
                
                # Find radius at 1.6 M☉
                try:
                    idx_1_6 = np.argmin(np.abs(data[:, 1] - 1.6))
                    eos_info['radius_at_1_6'] = data[idx_1_6, 0]
                except:
                    eos_info['radius_at_1_6'] = None
                    
            except Exception as e:
                print(f"Error loading {eos_info['filepath']}: {e}")
                return None
        
        return eos_info
    
    def _compute_statistics(self):
        """Compute statistics across all EOS (lazy loading approach)."""
        print("\nComputing statistics (sampling 500 random EOS)...")
        
        # Sample 500 random EOS to compute statistics
        sample_ids = np.random.choice(list(self.eos_dict.keys()), 
                                     size=min(500, len(self.eos_dict)), 
                                     replace=False)
        
        max_masses = []
        r14_values = []
        r16_values = []
        
        for eos_id in sample_ids:
            info = self._load_eos_data(eos_id)
            if info is not None:
                if info['max_mass'] is not None:
                    max_masses.append(info['max_mass'])
                if info['radius_at_1_4'] is not None:
                    r14_values.append(info['radius_at_1_4'])
                if info['radius_at_1_6'] is not None:
                    r16_values.append(info['radius_at_1_6'])
        
        self.statistics = {
            'max_mass': {
                'min': np.min(max_masses) if max_masses else None,
                'max': np.max(max_masses) if max_masses else None,
                'mean': np.mean(max_masses) if max_masses else None,
                'std': np.std(max_masses) if max_masses else None
            },
            'radius_at_1_4': {
                'min': np.min(r14_values) if r14_values else None,
                'max': np.max(r14_values) if r14_values else None,
                'mean': np.mean(r14_values) if r14_values else None,
                'std': np.std(r14_values) if r14_values else None
            },
            'radius_at_1_6': {
                'min': np.min(r16_values) if r16_values else None,
                'max': np.max(r16_values) if r16_values else None,
                'mean': np.mean(r16_values) if r16_values else None,
                'std': np.std(r16_values) if r16_values else None
            }
        }
    
    def get_eos(self, eos_id, load_data=True):
        """
        Retrieve an EOS by its ID.
        
        Parameters:
        -----------
        eos_id : int
            ID of the EOS (filename without .dat)
        load_data : bool
            If True, load the data from the file
        
        Returns:
        --------
        dict : Information about the EOS (filepath, data, derived quantities)
        """
        if eos_id not in self.eos_dict:
            print(f"EOS {eos_id} not found")
            return None
        
        if load_data:
            return self._load_eos_data(eos_id)
        else:
            return self.eos_dict[eos_id]
    
    def get_random_eos(self, n=1, load_data=True):
        """
        Get n random EOS from the dataset.
        
        Parameters:
        -----------
        n : int
            Number of random EOS to retrieve
        load_data : bool
            If True, load the data
        
        Returns:
        --------
        list : List of EOS info dictionaries
        """
        random_ids = np.random.choice(list(self.eos_dict.keys()), size=n, replace=False)
        
        results = []
        for eos_id in random_ids:
            eos_info = self.get_eos(eos_id, load_data=load_data)
            if eos_info is not None:
                results.append(eos_info)
        
        return results
    
    def find_eos_by_criteria(self, max_mass_min=None, max_mass_max=None,
                            r14_min=None, r14_max=None,
                            r16_min=None, r16_max=None,
                            max_results=None):
        """
        Find EOS matching specified criteria.
        
        Parameters:
        -----------
        max_mass_min, max_mass_max : float
            Range for maximum mass [M☉]
        r14_min, r14_max : float
            Range for radius at 1.4 M☉ [km]
        r16_min, r16_max : float
            Range for radius at 1.6 M☉ [km]
        max_results : int
            Maximum number of results to return
        
        Returns:
        --------
        list : List of matching EOS IDs
        """
        matching_ids = []
        
        for eos_id in self.eos_dict.keys():
            info = self._load_eos_data(eos_id)
            
            if info is None:
                continue
            
            # Check criteria
            if max_mass_min is not None and (info['max_mass'] is None or info['max_mass'] < max_mass_min):
                continue
            if max_mass_max is not None and (info['max_mass'] is None or info['max_mass'] > max_mass_max):
                continue
            if r14_min is not None and (info['radius_at_1_4'] is None or info['radius_at_1_4'] < r14_min):
                continue
            if r14_max is not None and (info['radius_at_1_4'] is None or info['radius_at_1_4'] > r14_max):
                continue
            if r16_min is not None and (info['radius_at_1_6'] is None or info['radius_at_1_6'] < r16_min):
                continue
            if r16_max is not None and (info['radius_at_1_6'] is None or info['radius_at_1_6'] > r16_max):
                continue
            
            matching_ids.append(eos_id)
            
            if max_results is not None and len(matching_ids) >= max_results:
                break
        
        return matching_ids
    
    def print_statistics(self):
        """Display statistics about the dataset."""
        print("\n" + "="*80)
        print("EOS DATASET STATISTICS")
        print("="*80)
        print(f"\nTotal number of EOS: {len(self.eos_dict)}")
        
        if hasattr(self, 'statistics'):
            print("\nSampled statistics (from 100 random EOS):")
            
            for param_name, stats in self.statistics.items():
                print(f"\n{param_name.upper().replace('_', ' ')}:")
                if stats['mean'] is not None:
                    print(f"  • Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"  • Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                else:
                    print("  • No data available")
        
        print("\n" + "="*80)
    
    def plot_mr_curves(self, eos_ids=None, n_random=5, figsize=(10, 7), plot_all=False):
        """
        Plot Mass-Radius curves for selected EOS.
        
        Parameters:
        -----------
        eos_ids : list
            List of EOS IDs to plot (if None, plot random EOS)
        n_random : int
            Number of random EOS to plot if eos_ids is None
        figsize : tuple
            Figure size
        plot_all : bool
            If True, plot all EOS in the dataset
        """
        import matplotlib.pyplot as plt
        
        if eos_ids is None:
            if plot_all:
                eos_list = [self.get_eos(eos_id, load_data=True) for eos_id in self.eos_dict.keys()]
            else:
                eos_list = self.get_random_eos(n=n_random, load_data=True)
        else:
            eos_list = [self.get_eos(eos_id, load_data=True) for eos_id in eos_ids]
            eos_list = [eos for eos in eos_list if eos is not None]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for eos in eos_list:
            if plot_all == True:
                c = plt.cm.viridis(eos['eos_id'] % 256 / 256)
                ax.plot(eos['radius_km'], eos['mass_solar'], 
                       color=c, lw=1, alpha=0.3)
            else:
                ax.plot(eos['radius_km'], eos['mass_solar'], 
                   label=f"EOS {eos['eos_id']} (M_max={eos['max_mass']:.2f})", 
                   lw=2, alpha=0.7)
        
        ax.set_xlabel('Radius [km]', fontsize=14)
        ax.set_ylabel('Mass [$M_\\odot$]', fontsize=14)
        ax.set_title('Mass-Radius Relations', fontsize=16)
        if plot_all == False:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(7.8, 18)
        ax.set_ylim(0, 3.5)
        
        plt.tight_layout()
        plt.show()
    
    def __len__(self):
        """Number of EOS in the dataset."""
        return len(self.eos_dict)
    
    def __repr__(self):
        return f"EOSDataset({len(self)} EOS files from {self.folder_path})"
    
'''
Functions used to generate synthetic lc data
'''
def abs_to_app_mag(mag_abs, distance_mpc):
    distance_modulus = 5 * np.log10(distance_mpc) + 25 # distance in Mpc
    mag_app = {}
    for filter, mags in mag_abs.items():
        mag_app[filter] = mags + distance_modulus
    return mag_app

def add_noise_error(mag, noise_level=0, max_error_level=0.1):
    noisy = {}
    errors = {}
    for filter, mags in mag.items():
        noise = np.random.normal(0, noise_level, size=mags.shape)
        noisy[filter] = mags + noise
        err = []
        for noisi in noise:
            # find the smallest error level that would still contains the true mag without the noise
            if abs(noisi) > max_error_level:
                err_i = abs(noisi) + np.random.uniform(0, 0.1) # if we have a lot of noise, we put an error level that is slightly bigger than the noise to have a difference between the error level and the noise
            else:
                err_i = np.random.uniform(abs(noisi), max_error_level) # min value of the error at the noise level to be sure that the true mag is within the error bars, max value is the max error level
            err.append(err_i)

        errors[filter] = np.array(err)
    return noisy, errors

def format_nmma_data_v2(times, mag, errors, filters, trigger_iso = '2025-01-01T00:00:00'):
    data_dict = {}
    # format NMMA : ISOTIME, BAND, MAG, MAG_ERR : 2017-08-18T00:00:00.000 ps1::g 17.41000 0.02000
    # convert svd time to NMMA time format
    iso_times = []
    trigger_dt = pd.to_datetime(trigger_iso)
    trigger_mjd  = trigger_dt.to_julian_date() - 2400000.5  # time of 1st detection -> use it as a filter to remove points before the trigger if timeshift is negative
    # convert trigger in MJD
    print("Trigger ISO:", trigger_iso)
    trigger_dt = pd.to_datetime(trigger_iso)
    for t in times:
        iso_dt = trigger_dt + pd.to_timedelta(t, unit='D')
        if iso_dt.to_julian_date() - 2400000.5 < trigger_mjd:
            iso_times.append('NaN') # if the time is before the trigger, we put NaN to be able to remove it later
            continue
        iso_str = iso_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        iso_times.append(iso_str)
    print("Light Curve generated\nISO times sample:", iso_times)
    # construire le format NMMA
    data_rows = []
    for filter in filters:
        for i, t in enumerate(times):
            if iso_times[i] == 'NaN':
                continue
            data_rows.append([iso_times[i], filter, mag[filter][i], errors[filter][i]])
    df = pd.DataFrame(data_rows)
    return df, trigger_mjd

# def a unified generate_synth_lc function that takes a model name and parameters, and generates the synthetic light curve using the appropriate model type (FIESTA or SVD)
def generate_synth_lc(model_name, 
                      params,
                      obs_duration=10.0,
                      pts_per_day=4,
                      delay=0.0,
                      jitter=0.0,
                      filters=["ps1::g", "ps1::r", "ps1::i"],
                      noise=0.0,
                      max_error=0.1,
                      trigger_iso='2025-01-01T00:00:00',
                      detection_limit_dict=None,
                      save=False,
                      filename="synth_lightcurve.dat",
                      svd_path=None):
    """
    
    Generate synthetic lightcurves using SVDLightCurveModel (Ka2017 and bu2019lm) and FIESTA surrogates.

    Parameters
    ----------
    model_name : str
        Name of the SVD model to be used.
    params : dict
        Dictionary of model parameters. (timeshift and redshift are not needed)
    filters : list
        List of filters to be used for the lightcurve generation. (All sncosmo filters are supported for FIESTA surrogates, for SVD models they are limited)
    noise : float
        Standard deviation of the Gaussian noise to be added.
    max_error : float
        Maximum error level to be added to the magnitudes.
    trigger_iso : str
        ISOT formatted trigger time.
    pts_per_day : int
        Number of observation points per day.
    obs_duration : int
        Duration of the observation in days.
    delay : float
        Delay in days to be added to the sample times (positive delay means that the first point will be after the trigger time = timeshift)
    jitter : float
        Maximum jitter to be added to the sample times in days.
    save : bool
        If True, save the generated data to a file.
    filename : str
        Filename to save the generated data if save is True.
    detection_limit_dict : dict or None
        Dictionary specifying detection limits for each filter. If None, no detection limits are applied.
    svd_path : str
        Path to the directory where NMMA's SVD models are stored locally. 
    
    Returns
    -------
    lc_df : pd.DataFrame
        DataFrame containing the synthetic lightcurve data in NMMA format.
    trigger : float
        Trigger time in MJD.
    
    """
    model_info = models.get(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} not found in the available models.\nList of available models: {list(models.keys())}")
    # check parameters consistency with the model's expected parameters
    expected_params = model_info["params"]
    for key in expected_params.keys():
        if key not in params and key not in ["timeshift", "redshift"]:
            raise ValueError(f"Missing parameter '{key}' for model '{model_name}'. Expected parameters: {list(expected_params.keys())}")
    if "timeshift" in params: # we want to avoid timeshift in the params as it is covered by delay
        del params["timeshift"]
    if "redshift" in params: # we want to avoid redshift in the params as it is covered by distance
        del params["redshift"]
    print(f"Generating synthetic light curve for model: {model_name}")

    # define the observation duration and sampling rate
    sample_times = np.arange(delay, obs_duration, 1/pts_per_day)
    for t in range(len(sample_times)):
        if t == 0:
            continue # we don't want to add jitter to the first point to avoid negative times
        sample_times[t] += np.random.uniform(-jitter, jitter)

    kind = model_info["kind"]
    if kind in ["KN", "GRB"]:
        # Fiesta model
        try:
            model = FiestaKilonovaModel(model_name, filters=filters)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Fiesta model for {model_name}: {e}")
    elif kind == "svd":
        if svd_path is None:
            raise ValueError(f"svd_path must be provided for SVD model {model_name}.")
        try:
            model = SVDLightCurveModel(
                    model=model_name,
                    sample_times=sample_times,
                    svd_path=svd_path,
                    interpolation_type='tensorflow',
                    filters=filters
            )
        except Exception as e:
            print(f"  - {model_name}: error during SVD model creation ->", e)
    # generate the light curve using the appropriate model + noise/error and convert to apparent magnitude
    mag_svd = model.generate_lightcurve(sample_times, params)
    mag_svd_noisy, mag_svd_errors = add_noise_error(mag_svd, noise_level=noise, max_error_level=max_error)
    mag_svd_noisy_app = abs_to_app_mag(mag_svd_noisy, distance_mpc=params["luminosity_distance"])

    # organize for NMMA format
    lc_df, trigger = format_nmma_data_v2(sample_times, mag_svd_noisy_app, mag_svd_errors, filters, trigger_iso=trigger_iso)

    if detection_limit_dict is not None:
        # apply detection limits
        for filter, limit in detection_limit_dict.items():
            filter_mask = lc_df[1] == filter # all rows with this filter
            limit_mask = lc_df[2] > limit # all rows with mag > limit (mag is an inverted scale)
            to_remove = filter_mask & limit_mask # combine masks
            lc_df = lc_df[~to_remove]
    if save:
        # get the directory path from the executed file path and create the directory if it doesn't exist
        dirpath = os.getcwd()
        print(f"Saving synthetic lightcurve data to {dirpath}...")
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            filename = os.path.join(dirpath, filename)
        lc_df.to_csv(filename, sep=' ', index=False, header=False)
    return lc_df, trigger

def generate_synth_lc_lsst(source,
                    model_name,
                    params,
                    save=False,
                    filename='test_lc_lsst.dat',
                    svd_path = None):
    '''
    Generate a synthetic light curve based on the LSST observations of a given source and a given model. The light curve is generated for each filter in the observations and then merged together to have a full light curve with all the filters. The generated light curve is then saved to a file if save is True.

    Parameters
    ----------
    source : dict
        Dictionary containing the LSST observations of the source. It should have the following structure:
        {
            'observations': pd.DataFrame with columns ['filter', 'expMJD', '5sigmaDepth', ...]
        }
        Should be an element of the list of sources returned by the function find_valid_sources().
    model_name : str
        Name of the model to be used for the light curve generation (e.g. 'Bu2026_MLP' or 'Bu2019lm').
    params : dict
        Dictionary containing the parameters of the model to be used for the light curve generation. The keys and values of the dictionary should be coherent with the model used (e.g. for 'Bu2026_MLP', the parameters should be 'log10_mej_dyn', 'log10_mej_wind', 'inclination_EM', 'luminosity_distance', 'v_ej_dyn', 'v_ej_wind', 'Ye_dyn', 'Ye_wind', 'timeshift').
    save : bool
        If True, the generated light curve will be saved to a file with the name specified in filename. If False, the generated light curve will not be saved.
    filename : str
        Name of the file where the generated light curve will be saved if save is True. The file will be saved in the current working directory.
    svd_path : str
        Path to the directory where the SVD models are stored. This is used only if the model_name is not 'Bu2026_MLP' because this model is not an SVD model and is directly instantiated with the FiestaKilonovaModel class.
    Returns
    -------
    full_lc : pd.DataFrame
        The generated synthetic light curve with all the filters merged together.
    '''

    model_info = models.get(model_name)
    if model_info is None:
        raise ValueError(f"Model {model_name} not found in the available models.\nList of available models: {list(models.keys())}")
    # check parameters consistency with the model's expected parameters
    expected_params = model_info["params"]
    for key in expected_params.keys():
        if key not in params and key not in ["timeshift", "redshift"]:
            raise ValueError(f"Missing parameter '{key}' for model '{model_name}'. Expected parameters: {list(expected_params.keys())}")
    if "timeshift" in params: # we want to avoid timeshift in the params as it is covered by delay
        del params["timeshift"]
    if "redshift" in params: # we want to avoid redshift in the params as it is covered by distance
        del params["redshift"]
    print(f"Generating synthetic light curve for model: {model_name}")

    # extract the filter from the obs
    filters = source['observations']['filter'].unique()
    print(f"Filters in the observations: {filters.tolist()}")
    # need to transform the filter names to match the ones used in the model: go from i to ps1::i for example
    if model_info["kind"] == "KN" or model_info["kind"] == "GRB":
        filters = [f"lsst{filt}" for filt in filters]
        try:
            model = FiestaKilonovaModel(model_name, filters=filters)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Fiesta model for {model_name}: {e}")
    else:
        filters = [("sdssu" if filt == "u" else f"ps1::{filt}") for filt in filters] # not lsst filters, we use the ps1 and sdss filters instead due to nmma lack of support for lsst filters in the SVD models
    
    dic = build_dic(source) # get the diff samples times for each filter

    full_lc = None # instanciate a var to store the full light curve (with all the filters merged together)

    # loop over the filters and generate the light curve for each filter
    for filter in dic.keys():
        if len(filter) > 1: 
            continue # skip for the m5 keys (filter key are like i and m5 keys are like i_m5)
        time_samples = dic[filter]
        m5_samples = dic[f"{filter}_m5"]
        print(f"Sample times for filter {filter}:")
        print(time_samples)

        if model_info["kind"] == "KN" or model_info["kind"] == "GRB":
            # generate the light curve for the given filter and time samples
            mag_filter = model.generate_lightcurve(time_samples, params)
        else:
            try:
                svd_model = SVDLightCurveModel(
                        model=model_name,
                        sample_times=time_samples,
                        svd_path=svd_path,
                        interpolation_type='tensorflow',
                        filters=filters
                )
            except Exception as e:
                print(f"  - {model_name}: error during SVD model creation ->", e)
                raise e
            mag_filter = svd_model.generate_lightcurve(time_samples, params)

        # transform the mag to observed mag
        mag_filter = abs_to_app_mag(mag_filter, params["luminosity_distance"])

        # get the mjd date of the first observation in that filter
        mjd_epoch = source['t0_mjd']
        # transform it to ISOT date
        t = Time(mjd_epoch, format='mjd')
        iso_epoch = t.isot

        # transform the mag to observed mag based on the m5 samples
        err = {}
        if model_info["kind"] == "KN" or model_info["kind"] == "GRB":
            key = f"lsst{filter}"
        else:
            key = "sdssu" if filter == "u" else f"ps1::{filter}"
        if key not in mag_filter:
            raise KeyError(f"Missing key in mag_filter: {key}. Available: {list(mag_filter.keys())}")
        err[key] = [None] * len(mag_filter[key])

        for idx, mag in enumerate(mag_filter[key]):
            m5 = m5_samples[idx]
            mag_filter[key][idx], err[key][idx] = get_obs_mag_from_lsst(filter, mag, m5)

        # format the output to match the expected format by nmma
        list_f = [key]
        formatted_lc, _ = format_nmma_data_v2(time_samples, mag_filter, err, list_f, iso_epoch)
        # merge it with previous filters (just append the 'formatted_lc' to each other) 
        if full_lc is None:
            full_lc = formatted_lc
        else:
            full_lc = pd.concat([full_lc, formatted_lc], ignore_index=True)


    if save:
        # get the directory path from the executed file path and create the directory if it doesn't exist
        dirpath = os.getcwd()
        print(f"Saving synthetic lightcurve data to {dirpath}...")
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            filename = os.path.join(dirpath, filename)
        full_lc.to_csv(filename, sep=' ', index=False, header=False)
    return full_lc

"""
Plotting utils for ts-loop
"""
def plot_param_evolution(MODEL, DIR, UL=False, true_merger='2020-01-07T00:00:00.000', minus_num=4, ts_max=-2.5, col_num=5, row_num=5):
    '''
    ### Plot the evolution of the parameters during the ts loop for a given model and grid.
    ### Parameters    
    - ``MODEL`` : str
        Name of the model (e.g. 'Bu2026_MLP')

    - ``DIR`` : str
        Directory where the ts loop results are stored (e.g. 'bu26_1')

    - ``UL`` : bool
        True only if the ts loop was run with upper limits 

    - ``true_merger`` : str
        ISOT time of the true merger (default: '2020-01-07T00:00:00.000')

    - ``minus_num`` : int
        Number of points removed through the ts loop of kn-ts-loop (default: 4)

    - ``ts_max`` : float
        Maximum timeshift value (negative) to be plotted (default: -2.5)

    - ``col_num`` : int
        Number of columns in the plot grid (default: 5) Should be coherent with the dimension of the grid plot

    - ``row_num`` : int
        Number of rows in the plot grid (default: 5) Should be coherent with the dimension of the grid plot

    Returns ``None``
        Saves plot in the grid directory under the subdirectory "plots" with the name "param_evolution.png"
    ---------
    '''
    lc_num = col_num * row_num

    import logging
    logging.getLogger('matplotlib.texmanager').setLevel(logging.WARNING)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'

    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(0, minus_num-1) # from 0 to the max number of points removed -1

    fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
    print(f"Plotting parameter evolution for model {MODEL} in directory {DIR} with UL={UL} and true merger time {true_merger}")
    for idx in range(lc_num):
        BASE_DIR = f"{DIR}/{idx}"
        if os.path.getsize(f"{BASE_DIR}/data{idx}.dat") > 0:
            data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
        else:
            print(f"Warning: Lightcurve data file {BASE_DIR}/data{idx}.dat is empty. Skipping this injection.")
            continue
        data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
        # stock the time list 
        times2 = data[0].unique()
        # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
        if col_num > 1 and row_num > 1:
            ax = axs[idx // col_num, (idx % col_num) * 2]
            axx = axs[idx // col_num, (idx % col_num) * 2 + 1]
        elif row_num > 1 and col_num == 1:
            ax = axs[idx, 0]
            axx = axs[idx, 1]
        else: 
            ax = axs[idx * 2]
            axx = axs[idx * 2 + 1]

        lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
        param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")

        if MODEL == 'Bu2019lm':
            model_param = {
                        "KNphi": param["KNphi"].values[0],
                        "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                        "log10_mej_wind": param["log10_mej_wind"].values[0],
                        "inclination_EM": param["inclination_EM"].values[0],
                        "luminosity_distance": param["luminosity_distance"].values[0]
            }
        elif MODEL == 'Ka2017':
            log10_mej = np.log10(10**param["log10_mej_dyn"].values[0] + 10**param["log10_mej_wind"].values[0]) # we sum the dynamical and wind ejecta to have a total ejecta mass for the Ka2017 model
            model_param = {
                "inclination_EM": param["inclination_EM"].values[0],
                "log10_mej": log10_mej,
                "log10_vej": param["log10_vej"].values[0],
                "log10_Xlan": param["log10_Xlan"].values[0],
                "luminosity_distance": param["luminosity_distance"].values[0]
            }
        elif MODEL == 'Bu2026_MLP':
            model_param = {
                    "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                    "log10_mej_wind": param["log10_mej_wind"].values[0],
                    "luminosity_distance": param["luminosity_distance"].values[0],
                    "inclination_EM": param["inclination_EM"].values[0],
                    "v_ej_dyn": param["v_ej_dyn"].values[0],
                    "v_ej_wind": param["v_ej_wind"].values[0],
                    "Ye_dyn": param["Ye_dyn"].values[0],
                    "Ye_wind": param["Ye_wind"].values[0],
            }
        for band in lc[1].unique():
            band_df = lc[lc[1]==band]
            
            # datetime
            times = pd.to_datetime(band_df[0].values)
            
            # relative time 
            times_days = (times - pd.to_datetime(true_merger)).total_seconds() / (3600 * 24)
            
            # lighter x axis
            axx.errorbar(times_days, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
        axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
        # temp modif for 1 plot
        #txt = "Bu2019lm" if idx == 0 else "Ka2017" if idx == 1 else "Bu2026_MLP"
        #axx.text(0.001, 0.99, txt, transform=axx.transAxes, fontsize=20, verticalalignment='top')
        axx.legend()
        axx.invert_yaxis()
        axx.set_xlabel('Time [days]')
        axx.set_ylabel('Magnitude')
        print("Finished plotting lightcurve for LC", idx)

        for i in range(minus_num):
            SAMPLE_PATH = f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
            if not os.path.exists(SAMPLE_PATH):
                print(f"Warning: Posterior samples file {SAMPLE_PATH} does not exist. Skipping this point.")
                continue
            samples = pd.read_csv(SAMPLE_PATH, delimiter=' ')
            if samples is None or samples.empty:
                print(f"Warning: Posterior samples for LC {idx} with minus {i} are missing or empty. Skipping this point.")
                continue

            if UL:
                ts = pd.to_datetime(times2[i]) - pd.to_datetime(true_merger) # keep the same trigger time as for the original data to see how the timeshift evolves
                ts = -1 *ts.total_seconds() / (3600*24) # convert to days
                adjust = pd.to_datetime(times2[i]) - pd.to_datetime(times2[0]) 
                adjust = -1 * adjust.total_seconds() / (3600*24)
                lower = samples['timeshift'].quantile(0.16) + adjust
                upper = samples['timeshift'].quantile(0.84) + adjust
                median = samples['timeshift'].median() + adjust
                ax.errorbar(ts, median, yerr=[[median - lower], [upper - median]], fmt='v', color=cmap(norm(i)), label=f'true ts: {ts} days')
            else:
                ts = pd.to_datetime(times2[i]) - pd.to_datetime(true_merger) # keep the same trigger time as for the original data to see how the timeshift evolves
                ts = -1 *ts.total_seconds() / (3600*24) # convert to days
                lower = samples['timeshift'].quantile(0.16) 
                upper = samples['timeshift'].quantile(0.84) 
                median = samples['timeshift'].median()
                ax.errorbar(ts, median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)), label=f'true ts: {ts} days')
            ax.plot([ts_max-0.25, 0], [ts_max-0.25, 0], ls='--', color='red', label='perfect recovery')
            ax.set_xlabel('Timeshift [days]')
            ax.set_ylabel('Inferred timeshift [days]')
    OUT_DIR = f"{DIR}/plots"
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/timeshift_evolution.png")
    print(f"Saved timeshift evolution plot in {OUT_DIR}/timeshift_evolution.png")

    # loop for the other parameters as well (corner plots for each analysis with modified data)
    if MODEL == 'Bu2019lm':
        param_range = [(10,200),(15,75),(0,np.pi/2),(-3,-1),(-3,-0.5)] # change as needed based on the prior bounds used for the analysis
        param_names = ['luminosity_distance', 'KNphi', 'inclination_EM', 'log10_mej_dyn', 'log10_mej_wind'] # change as needed based on the parameters used in the analysis
        param_labels = ['D_L [Mpc]', '$\\phi$ [deg]', '$\\iota$ [rad]', '$log_{10} M_{dyn}$ [$M_\\odot$]', '$log_{10} M_{wind}$ [$M_\\odot$]'] # change as needed for the plot labels
    elif MODEL == 'Ka2017':
        param_range = [(10,200),(0, np.pi/2),(-3,-0.5),(-1.5,-0.5),(-9,-1)] # change as needed based on the prior bounds used for the analysis
        param_names = ['luminosity_distance', 'inclination_EM', 'log10_mej', 'log10_vej', 'log10_Xlan'] # change as needed based on the parameters used in the analysis
        param_labels = ['D_L [Mpc]', '$\\iota$ [rad]', '$log_{10} M_{ej}$ [$M_\\odot$]', '$log_{10} v_{ej}$ [c]', '$log_{10} X_{lan}$'] # change as needed for the plot labels
    elif MODEL == 'Bu2026_MLP':
        param_range = [(10,200),(0, np.pi/2),(-3,-0.5),(-3,-0.5),(0.1,0.4),(0.01,0.2),(0.1,0.4),(0.1,0.4)] # change as needed based on the prior bounds used for the analysis
        param_names = ['luminosity_distance', 'inclination_EM', 'log10_mej_dyn', 'log10_mej_wind', 'v_ej_dyn', 'v_ej_wind', 'Ye_dyn', 'Ye_wind'] # change as needed based on the parameters used in the analysis
        param_labels = ['D_L [Mpc]', '$\\iota$ [rad]', '$log_{10} M_{dyn}$ [$M_\\odot$]', '$log_{10} M_{wind}$ [$M_\\odot$]', '$v_{ej,dyn}$ [c]', '$v_{ej,wind}$ [c]', '$Y_{e,dyn}$', '$Y_{e,wind}$'] # change as needed for the plot labels

    for ii, param_name, param_label in zip(range(len(param_range)), param_names, param_labels):
        fig, axs = plt.subplots(ncols=2*col_num, nrows=row_num, figsize=(15*col_num,5*row_num), gridspec_kw={'width_ratios': [0.333, 0.666]*col_num})
        # add fig, ax to do pp plots
        figg, axis = plt.subplots(figsize=(10,10))
        for idx in range(lc_num):
            BASE_DIR = f"{DIR}/{idx}"
            # attribute the left column to timeshift evolution and the right column to lightcurve and set up correctly the axes
            if col_num > 1 and row_num > 1:
                ax = axs[idx // col_num, (idx % col_num) * 2]
                axx = axs[idx // col_num, (idx % col_num) * 2 + 1]
            elif row_num > 1 and col_num == 1:
                ax = axs[idx, 0]
                axx = axs[idx, 1]
            else: 
                ax = axs[idx * 2]
                axx = axs[idx * 2 + 1]
            if os.path.getsize(f"{BASE_DIR}/data{idx}.dat") > 0:
                lc = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None)
            else:
                print(f"Warning: Lightcurve data file {BASE_DIR}/data{idx}.dat is empty. Skipping this injection.")
                continue
            param = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
            if MODEL == 'Bu2019lm':
                model_param = {
                            "KNphi": param["KNphi"].values[0],
                            "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                            "log10_mej_wind": param["log10_mej_wind"].values[0],
                            "inclination_EM": param["inclination_EM"].values[0],
                            "luminosity_distance": param["luminosity_distance"].values[0]
                }
            elif MODEL == 'Ka2017':
                log10_mej = np.log10(10**param["log10_mej_dyn"].values[0] + 10**param["log10_mej_wind"].values[0]) # we sum the dynamical and wind ejecta to have a total ejecta mass for the Ka2017 model
                model_param = {
                    "inclination_EM": param["inclination_EM"].values[0],
                    "log10_mej": log10_mej,
                    "log10_vej": param["log10_vej"].values[0],
                    "log10_Xlan": param["log10_Xlan"].values[0],
                    "luminosity_distance": param["luminosity_distance"].values[0]
                }
            elif MODEL == 'Bu2026_MLP':
                model_param = {
                    "luminosity_distance": param["luminosity_distance"].values[0],
                    "inclination_EM": param["inclination_EM"].values[0],
                    "log10_mej_dyn": param["log10_mej_dyn"].values[0],
                    "log10_mej_wind": param["log10_mej_wind"].values[0],
                    "v_ej_dyn": param["v_ej_dyn"].values[0],
                    "v_ej_wind": param["v_ej_wind"].values[0],
                    "Ye_dyn": param["Ye_dyn"].values[0],
                    "Ye_wind": param["Ye_wind"].values[0]
                }
            for band in lc[1].unique():
                band_df = lc[lc[1]==band]
                
                times = pd.to_datetime(band_df[0].values)
                
                times_days = (times - pd.to_datetime(true_merger)).total_seconds() / (3600 * 24)
                
                axx.errorbar(times_days, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
            axx.text(0.001, 0.99, f"LC {idx}", transform=axx.transAxes, fontsize=20, verticalalignment='top')
            #txt = "Bu2019lm" if idx == 0 else "Ka2017" if idx == 1 else "Bu2026_MLP"
            #axx.text(0.001, 0.99, txt, transform=axx.transAxes, fontsize=20, verticalalignment='top')
            axx.legend()
            axx.invert_yaxis()
            axx.set_xlabel('Time [days]')
            axx.set_ylabel('Magnitude')
            for i in range(minus_num): 
                SAMPLE_PATH = f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
                if not os.path.exists(SAMPLE_PATH):
                    print(f"Warning: Posterior samples file {SAMPLE_PATH} does not exist. Skipping this point.")
                    continue
                samples = pd.read_csv(SAMPLE_PATH, delimiter=' ')
                if samples is None or samples.empty:
                    print(f"Warning: Posterior samples for LC {idx} with minus {i} are missing or empty. Skipping this point.")
                    continue
                truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
                lower = samples[param_name].quantile(0.16)
                upper = samples[param_name].quantile(0.84)
                median = samples[param_name].median()
                if param_name == 'log10_mej': # for the Ka2017 model, we have to sum the dynamical and wind ejecta to have a total ejecta mass for the pp plot
                    truth[param_name] = np.log10(10**truth["log10_mej_dyn"].values[0] + 10**truth["log10_mej_wind"].values[0])
                ax.errorbar(truth[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
                axis.errorbar(truth[param_name].values[0], median, yerr=[[median - lower], [upper - median]], fmt='o', color=cmap(norm(i)))
            ax.plot(param_range[ii], param_range[ii], ls='--', color='red', label='perfect recovery')
            ax.set_xlabel(param_label)
            ax.set_ylabel(f'Inferred {param_label}')
        axis.plot(param_range[ii], param_range[ii], ls='--', color='red', label='perfect recovery')
        axis.set_xlabel(f'Injected {param_label}')
        axis.set_ylabel(f'Inferred {param_label}')
        axis.set_title(f'Injection-recovery plot for {param_label}')
        OUT_DIR = f"{DIR}/plots"
        os.makedirs(OUT_DIR, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{param_name}_evolution.png")
        figg.tight_layout()
        figg.savefig(f"{OUT_DIR}/{param_name}_pp.png")
        print(f"Saved {param_name} evolution plot in {OUT_DIR}/{param_name}_evolution.png")
        print(f"Saved {param_name} pp plot in {OUT_DIR}/{param_name}_pp.png")
    return None
    
'''
LSST utils for synthetic lightcurve
'''
def get_lsst_observations(db_path, full_df=False, n_visits=20):
    '''
    Retrieve LSST observations from a SQLite database. Assumes a table named 'observations' with columns:
    - observationStartMJD (float): Modified Julian Date of the observation start
    - filter (string): Filter used for the observation (e.g., 'u', 'g', 'r', 'i', 'z', 'y')
    - fiveSigmaDepth (float): 5-sigma depth for the observation (limiting magnitude)
    - fieldRA (float): Right Ascension of the observed field
    - fieldDec (float): Declination of the observed field
    '''
    import sqlite3
    con = sqlite3.connect(db_path)
    # randomly select n_visits observations to simulate a realistic observing cadence
    if full_df:
        query = """
        SELECT observationStartMJD as expMJD, fieldRA as _ra, fieldDec as _dec, filter, fiveSigmaDepth 
        FROM observations
        ORDER BY observationStartMJD ASC
        """
    else:
        query = """
    SELECT observationStartMJD as expMJD, fieldRA as _ra, fieldDec as _dec, filter, fiveSigmaDepth 
    FROM observations
    ORDER BY RANDOM()
    LIMIT {n_visits}
    """.format(n_visits=n_visits)
    df_rubin = pd.read_sql_query(query, con)
    con.close()
    df_rubin['filter'] = df_rubin['filter']
    df_rubin = df_rubin.sort_values('expMJD')
    return df_rubin

def get_source_observations(df_lsst, t0_mjd, ra, dec, duration=20, verbose=True):
    '''
    Extract LSST observations that are relevant for a source that exploded at t0_mjd and is located at (ra, dec).
    This function filters the LSST observations to find those that occurred during the expected lifetime of the kilonova (e.g., 20 days) and checks if the source was in the field of view

    Parameters:
    - df_lsst: DataFrame containing LSST observations with columns 'expMJD', '_ra', '_dec', 'filter', 'fiveSigmaDepth'
    - t0_mjd: MJD of the explosion (t0)
    - ra: Right Ascension of the source in degrees
    - dec: Declination of the source in degrees
    - duration: Expected lifetime of the kilonova in days (default is 20)
    - verbose: If True, print information about the filtering process
    '''
    df_lsst['time_days'] = df_lsst['expMJD'] - t0_mjd
    valid_obs = df_lsst[(df_lsst['time_days'] > 0) & (df_lsst['time_days'] < duration)].copy()
    
    if valid_obs.empty:
        if verbose:
            print("No observations during the KN lifetime.")
        return None # Aucune observation pendant la vie de la KN
    # check for the presence of the source in the field of view (simplified as a circular area around the pointing)
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    center = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    pts = SkyCoord(ra=valid_obs["_ra"].to_numpy()*u.deg,
                dec=valid_obs["_dec"].to_numpy()*u.deg, frame="icrs")

    valid_obs["in_fov"] = pts.separation(center).deg < 1.75 # LSST FOV is ~3.5 deg in diameter, so we take half of that as radius
    valid_obs = valid_obs[valid_obs["in_fov"]]

    if valid_obs.empty:
        if verbose:
            print("No observations in the field of view.")
        return None
    return valid_obs

def find_valid_sources(df_lsst, n_sources=5, duration=20, min_observations=3, seed=None):
    '''
    Find n_sources valid sources in the LSST database that have observations during their expected kilonova lifetime and are located within the field of view.

    Parameters:
    - df_lsst: DataFrame containing LSST observations with columns 'expMJD', '_ra', '_dec', 'filter', 'fiveSigmaDepth'
    - n_sources: Number of valid sources to find (default is 5)
    - duration: Expected lifetime of the kilonova in days (default is 20)
    - min_observations: Minimum number of observations required for a source to be considered valid (default is 3)
    - seed: Random seed for reproducibility (default is None)
    '''
    valid_sources = []
    if seed is not None:
        np.random.seed(seed)
    while len(valid_sources) < n_sources:
        # Randomly select a source position and explosion time
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        t0_mjd = np.random.uniform(df_lsst['expMJD'].min(), df_lsst['expMJD'].max() - duration)

        obs = get_source_observations(df_lsst, t0_mjd, ra, dec, duration, verbose=False)
        if obs is not None and len(obs) >= min_observations:
            valid_sources.append({
                'ra': ra,
                'dec': dec,
                't0_mjd': t0_mjd,
                'observations': obs
            })
    
    return valid_sources

def get_obs_mag_from_lsst(filter, model_mag, m5):
    '''
    Compute the observed mag based the 5 sigma depth mag using ``rubin_sim.phot_utils``
    '''
    # get detector status
    phot_params = PhotometricParameters(bandpass=filter)

    # load filter bandpass with lsst's throughtput files
    if filter == "u":
        throughput_dir = os.path.join(get_data_dir(), "throughputs", "sdss")
        bp_filepath = os.path.join(throughput_dir, f"doi_{filter}.dat")
    else:
        throughput_dir = os.path.join(get_data_dir(), "throughputs", "panStarrs")
        bp_filepath = os.path.join(throughput_dir, f"panStarrs_{filter}.dat")
    bp = Bandpass()
    bp.read_throughput(bp_filepath)

    # compute snr and associated error
    snr, gamma = calc_snr_m5(model_mag, bp, m5, phot_params)
    mag_err, _ = calc_mag_error_m5(model_mag, bp, m5, phot_params, gamma=gamma)

    # get real mag from Gaussian distrib
    mag_obs = np.random.normal(loc=model_mag, scale=mag_err)
    is_detected = mag_obs < m5 # (ou SNR > 5)
    if is_detected == False:
        return m5, np.inf # upper lim at m5
    else:
        return mag_obs, mag_err
    
def build_dic(source):
    '''
    Build a dic containing: 
    filter
            associated time samples
    filter_m5
            associated m5 samples
    Used for generate_synth_lc_lsst()
    '''
    observations = source['observations']
    obs_per_band = observations['filter']
    time_arrays = {}

    for band in obs_per_band.unique():
        mask = obs_per_band == band
            
        temp_array = observations[mask]['time_days'].to_numpy()
        m5 = observations[mask]['fiveSigmaDepth'].to_numpy()
        #print(f"Band: {band}, time samples: {temp_array}, m5: {m5}")
            
        time_arrays[band] = temp_array
        time_arrays[f"{band}_m5"] = m5
    return time_arrays

'''
Other utils (for ts-loop)
'''

def regenerate_lc_from_truth(idx, truth_file, out_dir, model, filters, cadence, delay, noise_level, max_error_level=0.4, obs_duration=15, detection_limit_dict=None, jitter=0.0, svd_path="/home/stu_jamsin/jamsin/NMMA/svdmodels", ISOT_trigger="2020-01-07T00:00:00.000"):
    '''
    Regenerate a synthetic light curve from a given truth file and save it to the specified output directory.
    The functions uses the generate_synth_lc functions from peket.kn_side.utils so the parameters are not exactly the same as the ones in the truth file, but they are derived from them. The function is designed to be compatible with the ts-loop.
    Parameters:
    - idx: the index of the light curve to regenerate (used for naming the output files)
    - truth_file: the path to the truth file containing the parameters of the light curve to regenerate
    - out_dir: the directory where the regenerated light curve will be saved (a subdirectory with the name of the idx will be created if it doesn't exist)
    - model: the name of the model to use for generating the light curve (e.g. "Bu2019lm", "Ka2017", "Bu2026_MLP")
    - filters: the list of filters to use for generating the light curve (e.g. ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"])
    - cadence: the cadence (in days) to use for generating the light curve (e.g. 0.5 for one point every 12 hours)
    - delay: the delay (in days) to apply to the light curve (e.g. 0.3 for a delay of 7.2 hours)
    - noise_level: the level of noise to add to the light curve (e.g. 0.2 for a noise level of 0.2 magnitudes)
    - max_error_level: the maximum error level to use for generating the light curve (e.g. 0.4 for a maximum error level of 0.4 magnitudes)
    - obs_duration: the duration (in days) of the observations to generate (e.g. 15 for a duration of 15 days)
    - detection_limit_dict: a dictionary containing the detection limits for each filter (e.g. {'ps1::g': 24.7, 'ps1::r': 24.2, 'ps1::i': 23.8, 'ps1::z': 23.2, 'ps1::y': 22.3})
    - jitter: the level of jitter to add to the light curve (e.g. 0.0 for no jitter, 0.1 for a jitter level of 0.1 magnitudes)
    - svd_path: the path to the directory containing the SVD models to use for generating the light curve (only used for models other than "Bu2026_MLP")
    - ISOT_trigger: the ISOT time of the trigger to use for generating the light curve (e.g. "2020-01-07T00:00:00.000" for a trigger on January 7th, 2020 at 00:00:00)
    '''
    df_truth = pd.read_csv(truth_file)
    
    if model == 'Bu2026_MLP':
        model_param = {
            "log10_mej_dyn": df_truth["log10_mej_dyn"].values[0],
            "log10_mej_wind": df_truth["log10_mej_wind"].values[0],
            "luminosity_distance": df_truth["luminosity_distance"].values[0],
            "inclination_EM": df_truth["inclination_EM"].values[0],
            "v_ej_dyn": df_truth["v_ej_dyn"].values[0],
            "v_ej_wind": df_truth["v_ej_wind"].values[0],
            "Ye_dyn":  df_truth["Ye_dyn"].values[0],
            "Ye_wind": df_truth["Ye_wind"].values[0],
            "redshift": np.array(z_at_value(Planck18.luminosity_distance, df_truth["luminosity_distance"].values[0]*u.Mpc)),
            "timeshift": 0
        }
        lc, _ = generate_synth_lc_fiesta(model_name=model, model_param=model_param, save=1, max_error_level=max_error_level, filters_band=filters, pts_per_day=cadence, delay=delay, filename=out_dir+f"/{idx}/data{idx}.dat", obs_duration=obs_duration, noise_level=noise_level, detection_limit_dict=detection_limit_dict, jitter=jitter, trigger_iso=ISOT_trigger)
        

    elif model == 'Bu2019lm':
        model_param = {
                "KNphi": df_truth["KNphi"].values[0],
                "log10_mej_dyn": df_truth["log10_mej_dyn"].values[0],
                "log10_mej_wind": df_truth["log10_mej_wind"].values[0],
                "inclination_EM": df_truth["inclination_EM"].values[0],
                "luminosity_distance": df_truth["luminosity_distance"].values[0],
                "timeshift": 0
        }
    elif model == 'Ka2017':
        model_param = {
                "luminosity_distance": df_truth["luminosity_distance"].values[0],
                "log10_vej": df_truth["log10_vej"].values[0],
                "log10_Xlan": df_truth["log10_Xlan"].values[0],
                "timeshift": 0,
                "log10_mej": np.log10(10**(df_truth["log10_mej_wind"].values[0]) + 10**(df_truth["log10_mej_dyn"].values[0])), # total ejecta mass for Ka2017
                "inclination_EM": df_truth["inclination_EM"].values[0]
        }
    if model != 'Bu2026_MLP':
        lc, _ = generate_synth_lc_v2(model_name=model, model_param=model_param, save=1, max_error_level=max_error_level, filters_band=filters, pts_per_day=cadence, delay=delay, filename=out_dir+f"/{idx}/data{idx}.dat", obs_duration=obs_duration, noise_level=noise_level, detection_limit_dict=detection_limit_dict, jitter=jitter, svd_path=svd_path, trigger_iso=ISOT_trigger)
    # copy the truth file to the output directory for reference
    df_truth.to_csv(out_dir+f"/{idx}/true{idx}.csv", index=False)
    return lc

import shutil
def duplicate_grid(original_dir, new_dir, num_lc=25):
    os.makedirs(new_dir, exist_ok=True)
    for i in range(num_lc):
        src_csv = f"{original_dir}/{i}/true{i}.csv"
        src_dat = f"{original_dir}/{i}/data{i}.dat"
        dst_csv = f"{new_dir}/{i}/true{i}.csv"
        dst_dat = f"{new_dir}/{i}/data{i}.dat"
        # create subdir for each lc
        os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
        shutil.copy(src_csv, dst_csv)
        shutil.copy(src_dat, dst_dat)
    print(f"Grid duplicated from {original_dir} to {new_dir}.")