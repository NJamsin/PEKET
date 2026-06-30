#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18, z_at_value
import astropy.units as u
from astropy.time import Time
import os
import pandas as pd
import json
from .utils import dyn_ej, wind_ej
from .utils import generate_synth_lc
import argparse
from .config import models, grb_params, kn_params, GRB_FIESTA_PARAM_RANGES, KN_FIESTA_PARAM_RANGES

# adapt the make_grid.py to Bu2026 with NMMA/FIESTA 
def main():
    '''
    Step 0: Get the parser and parse the arguments
    '''
    parser = argparse.ArgumentParser(description="Generate a pseudo-randomly sampled grid of synthetic lightcurves using NMMA/FIESTA with Bu2026 model. And save them as photometic .dat files formatted for nmma's lightcurve-analysis.")
    parser.add_argument("--out-dir", type=str, help="Base directory for the output files.")
    parser.add_argument("--model", type=str, default='Bu2026_MLP', help="Model name to use for generating synthetic lightcurves. (default: Bu2026_MLP). Currently supports 'Bu2026_MLP', 'Bu2019lm' and 'Ka2017'.")
    parser.add_argument("--num-lc", type=int, default=25, help="Number of lightcurves to generate.")
    parser.add_argument("--filters", nargs="+", help="Filters used for observation.")
    parser.add_argument("--eos-path", type=str, help="Path to the EOS file for the fitting formulae.")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise level for the synthetic lightcurves.")
    parser.add_argument("--max-error", type=float, default=0.4, help="Max error level for the synthetic lightcurves.")
    parser.add_argument("--trigger-isot", type=str, default='2020-01-07T00:00:00', help="Trigger time for the synthetic lightcurves.")
    parser.add_argument("--cadence", type=float, default=0.5, help="Cadence in days for the synthetic lightcurves.")
    parser.add_argument("--obs-duration", type=float, default=7, help="Observation duration in days for the synthetic lightcurves.")
    parser.add_argument("--jitter", type=float, default=0.0, help="Jitter in days for the synthetic lightcurves.")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay in days between the trigger and the first observation.")
    parser.add_argument("--detection-limit", nargs='+', help="Detection limits for the synthetic data in the format: filter1:limit1 filter2:limit2 ... (e.g., --detect_limit ps1::g=24.7 ps1::r=24.2)")
    parser.add_argument("--save-json", default=None, action='store_true', help="Whether to save the times and magnitudes for each sample in a json file. (json files will be saved in the same directory as the .dat files)")
    parser.add_argument("--svd-path", type=str, default="/home/stu_jamsin/jamsin/NMMA/svdmodels", help="Path to the directory containing the svd models.")
    parser.add_argument("--Dl", type=float, default=200.0, help="Maximum luminosity distance in Mpc for the synthetic lightcurves. (default: 200.0 Mpc)")

    args = parser.parse_args()

    BASE_DIR = os.path.abspath(args.out_dir)
    MODEL = args.model
    num_lc = args.num_lc
    filters_band = []

    for f in args.filters:
        filters_band.append(f)
    eos_path = args.eos_path
    noise_level = args.noise
    max_error_level = args.max_error
    trigger_isot = args.trigger_isot
    cadence = args.cadence
    obs_duration = args.obs_duration
    jitter = args.jitter
    detect_limit_dict = {}

    if args.detection_limit:
        for limit in args.detection_limit:
            filter_name, limit_value = limit.split('=')
            detect_limit_dict[filter_name] = float(limit_value)

    # define param bounds based on the model
    model_info = models.get(MODEL)
    kind = model_info.get("kind")
    print(f"Generating synthetic lightcurves using model: {MODEL} of kind: {kind}")
    if kind == "GRB":
        param_bounds = GRB_FIESTA_PARAM_RANGES[MODEL]
        param_bounds['luminosity_distance'] = (1.0, args.Dl)  # set the luminosity distance range based on the argument
    else:
        param_bounds = KN_FIESTA_PARAM_RANGES[MODEL]
        param_bounds['luminosity_distance'] = (1.0, args.Dl)  # set the luminosity distance range based on the argument
        param_bounds['mass_1'] = (1.1, 2.2)
        param_bounds['mass_2'] = (1.1, 2.2)

    if len(BASE_DIR) > 60:
        print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    '''
    Step 1: generate the grid of parameters 
    '''
    if kind == "KN" or kind == "svd":
        # setup EOS for the fitting formula 
        eos = np.loadtxt(eos_path)
        r_eos = eos[:,0]  # radius in km
        M_eos = eos[:,1]  # mass in solar masses
        # interpolate to get radius at 1.4 solar masses
        R_16 = np.interp(1.6, M_eos, r_eos) # radius of 1.6 solar masses


    l_bounds = [bounds[0] for bounds in param_bounds.values()]
    u_bounds = [bounds[1] for bounds in param_bounds.values()]

    sampler = qmc.Halton(d=len(param_bounds), scramble=True) # pseudo random sampler for better coverage of the parameter space
    valid_samples = []

    while len(valid_samples) < num_lc:
        # On génère un petit lot
        points = sampler.random(n=200)
        points_scaled = qmc.scale(points, l_bounds, u_bounds)
        
        for i, p in enumerate(points_scaled):
            dic = None
            if kind == "KN" or kind == "svd":
                r1 = np.interp(p[-2], M_eos, r_eos)
                r2 = np.interp(p[-1], M_eos, r_eos)
                mej_dyn = dyn_ej(M1=p[-2], M2=p[-1], R1=r1, R2=r2)
                zeta = np.random.uniform(0.01, 0.5)
                log10_mej_wind = wind_ej(M1=p[-2], M2=p[-1], Mtov=np.max(M_eos), R16=R_16) + np.log10(zeta) # consider between 1 and 50% of the disk mass as wind ejecta
                if mej_dyn > 0 and p[-2] > p[-1]:
                    dic = {k: v for k, v in zip(param_bounds.keys(), p)}
                    dic['zeta'] = zeta
            else:
                dic = {k: v for k, v in zip(param_bounds.keys(), p)}
            if dic is not None:
                valid_samples.append(dic)
                if len(valid_samples) >= num_lc:
                    break
    '''
    Step 2: generate synthetic lightcurves for each parameter combination in the sample and create a huge plot with all the lightcurves for visual check
    '''
    # clean latex warning
    import logging
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('default')
    logging.getLogger('matplotlib.texmanager').setLevel(logging.WARNING)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axs = plt.subplots(5, num_lc // 5, figsize=(10*(num_lc // 5), 6*(num_lc // 5)), sharex=True, sharey=True)
    for i, sample in enumerate(valid_samples):
        witness = num_lc // 5

        model_param = {n: v for n, v in zip(param_bounds.keys(), sample.values())}

        # save model param to csv for reference
        OUT_DIR = f"{BASE_DIR}/{i}"
        os.makedirs(OUT_DIR, exist_ok=True)
        true_dic = {n: v for n, v in zip(param_bounds.keys(), sample.values())}
            
        param_df = pd.DataFrame([true_dic])
        param_df.to_csv(f"{OUT_DIR}/true{i}.csv", index=False)   
        print(f"Generating synthetic lightcurve {i+1}/{num_lc}...")

        data_nmma_svd, trig = generate_synth_lc(
                model_name=MODEL,
                params=model_param,
                filters=filters_band,
                noise=noise_level,
                max_error=max_error_level,
                trigger_iso=trigger_isot,
                pts_per_day=cadence,
                obs_duration=obs_duration,
                jitter=jitter,
                delay=args.delay,
                save=True,
                filename=f"{OUT_DIR}/data{i}.dat",
                detection_limit_dict=detect_limit_dict,
                svd_path=args.svd_path
            )

        if args.save_json:
            print(f"Saving times and magnitudes for sample {i+1} in a json file...")
        # save the times and magnitudes for this sample in a json file"
            # transform data_nmma_svd to a dict with filter names as keys and list of magnitudes as values, and times as a list of iso strings
            tt = (str(t) for t in data_nmma_svd[0])
            times = [Time(t, format='isot', scale='utc').mjd for t in tt]
            # Create a JSON-safe version of model_param by converting ndarrays to standard lists/floats
            json_safe_params = {
                key: (val.tolist() if isinstance(val, np.ndarray) else val) 
                for key, val in model_param.items()
            }
            magnitudes = {filter_name: data_nmma_svd[data_nmma_svd[1]==filter_name][2].values.tolist() for filter_name in filters_band}
            with open(f"{BASE_DIR}/{i}/data{i+1}.json", "w") as f:
                json.dump({"times": times, "magnitudes": magnitudes, "parameters": json_safe_params}, f, indent=4)

        # plot part 
        print(f"Plotting lightcurve {i+1}/{num_lc}...")
        row = i // (num_lc // 5)
        col = i % (num_lc // 5)
        ax = axs[row, col]
        for band in data_nmma_svd[1].unique():
            band_df = data_nmma_svd[data_nmma_svd[1]==band]
            times = pd.to_datetime(band_df[0].values)
            ax.errorbar(times, band_df[2], yerr=band_df[3], fmt='o', label=band, ls='-')
        if col == 0:
            ax.set_ylabel('Magnitude')  
        ax.legend(loc='upper right')
        ax.text(0.005, 0.99, f"LC {i}", transform=ax.transAxes, fontsize=20, verticalalignment='top')
        if row == num_lc // 5: # only set x label for the bottom row
            ax.set_xlabel('Time [days]')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
        ax.invert_yaxis() # invert y axis for magnitude
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/all_lightcurves.png", dpi=150)
    plt.close(fig)

    print(f"All synthetic lightcurves generated and saved in {BASE_DIR}.")

    return 0

if __name__ == "__main__":
    sys.exit(main())