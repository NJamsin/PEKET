#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg') # change backend to avoid display error
import pandas as pd
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import corner
import gc
import sys
import bilby
from astropy.time import Time
import argparse
import traceback

def main():
    '''
    0. SETUP PARSER
    '''
    parser = argparse.ArgumentParser(description="Perform inference on the light curve data with different time shifts and plot the corner plots for each analysis.")
    parser.add_argument('--idx', default=0, type=int, help='Index of the injection to analyze (following the same structure as the lc generated with kn-make-grid)')
    parser.add_argument('--grid-dir', type=str, help='Path to the grid directory')
    parser.add_argument('--model', type=str, default='Bu2019lm', help='EM model to use for the inference (default: Bu2019lm)')
    parser.add_argument('--svd-path', type=str, help='Path to the SVD models (not needed for Bu2026_MLP)')
    parser.add_argument('--prior-file', type=str, help='Path to the prior file')
    parser.add_argument('--minus-pts', type=int, default=2, help='Number of time points to remove for the inference (default: 2, update as needed, up to the number of time points - 1)')
    parser.add_argument('--add-ul', default=False, action='store_true', help='Whether to add an upper limit instead of the removed points (default: False)')
    parser.add_argument('--true-merger-time', type=str, default='2020-01-07T00:00:00', help='True merger time to use for the timeshift calculation (default: 2025-01-01T00:00:00.000, keep the same trigger time for all analyses to see how the timeshift evolves)')
    parser.add_argument('--nlive', type=int, default=1024, help='Number of live points for the nested sampling (default: 1024, update prior bounds if needed)')
    parser.add_argument('--resampling', default=False, action='store_true', help='Whether to run the resampling with gwem-resampling after the inference (default: False)')
    parser.add_argument('--eos-posterior', type=str, default='/home/stu_jamsin/jamsin/add_files/posterior_probability.txt', help='Path to the EOS posterior probability file for the GW samples generation (default: /home/stu_jamsin/jamsin/add_files/posterior_probability.txt, update as needed)')
    parser.add_argument('--eos-path', type=str, default='/home/stu_jamsin/jamsin/NMMA/EOS/15nsat_cse_uniform_R14/macro/', help='Path to the EOS models for the resampling (default: /home/stu_jamsin/jamsin/NMMA/EOS/15nsat_cse_uniform_R14/macro/, update as needed)')
    parser.add_argument('--GW-prior', type=str, default='/home/stu_jamsin/jamsin/NMMA/priors/GWBNS.prior', help='Path to the GW prior file for the resampling (default: /home/stu_jamsin/jamsin/NMMA/priors/GWBNS.prior, update as needed)')
    parser.add_argument('--EM-prior', type=str, default='/home/stu_jamsin/jamsin/NMMA/priors/mespriors/Bu19_GW.prior', help='Path to the EM prior file for the resampling (default: /home/stu_jamsin/jamsin/NMMA/priors/mespriors/Bu19_GW.prior, update as needed)')
    parser.add_argument('--restrict-dist-prior', type=float, default=None, help='Whether to restrict the distance prior, will restrict the prior as true_val +/- value (Default is None)')
    args = parser.parse_args()
    idx = args.idx
    MODEL = args.model
    minus_pts = args.minus_pts
    UL = args.add_ul
    true_merger_time = args.true_merger_time
    nlive = args.nlive
    if MODEL != 'Bu2026_MLP':
        svd_path = args.svd_path
    prior_file = args.prior_file

    BASE_DIR = f"{args.grid_dir}/{idx}"  # change as needed
    if len(BASE_DIR) > 60:
        print("Warning: BASE_DIR path is quite long, which may cause issues with some software. Consider using a shorter path if you encounter errors related to file paths.")
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    # get the env 
    env = os.environ.copy()
    current_env_bin = os.path.join(sys.prefix, 'bin')
    env['PATH'] = current_env_bin + os.pathsep + env.get('PATH', '')
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # to avoid GPU memory issues

    '''
    1. DEFINE FUNCTION TO SAVE CORNER PLOT
    '''
    DTYPE_FLOAT = np.float32 # attempt to reduce memory usage

    def save_corner_plot(samples, truth_row, ts, out_path, title, model=MODEL):
        if model == 'Bu2019lm':
            labels = ['$D_L$', '$\\iota$', '$t_0$', '$log_{10} M_{dyn}$', '$log_{10} M_{wind}$', '$\\phi$']
            cols = ['luminosity_distance','inclination_EM', 'timeshift', 'log10_mej_dyn', 'log10_mej_wind', 'KNphi']
            truth_val = [truth_row['luminosity_distance'].values[0], truth_row['inclination_EM'].values[0],
                        -1*ts, truth_row['log10_mej_dyn'].values[0], truth_row['log10_mej_wind'].values[0], truth_row['KNphi'].values[0]]
        elif model == 'Ka2017':
            labels = ['$D_L$', '$\\iota$', '$t_0$', '$\\log_{10} M_{ej}$', '$\\log_{10} v_{ej}$', '$log_{10} X_{lan}$']
            cols = ['luminosity_distance', 'inclination_EM', 'timeshift', 'log10_mej', 'log10_vej', 'log10_Xlan']
            log10_mej = np.log10(10**truth_row["log10_mej_dyn"].values[0] + 10**truth_row["log10_mej_wind"].values[0])
            truth_val = [truth_row['luminosity_distance'].values[0], truth_row['inclination_EM'].values[0], 
                        -1*ts, log10_mej, truth_row['log10_vej'].values[0], truth_row['log10_Xlan'].values[0]]
        elif model == 'Bu2026_MLP':
            labels = ['$D_L$', '$\\iota$', '$t_0$', '$log_{10} M_{dyn}$', '$log_{10} M_{wind}$','$v^{\\rm{dyn}}_{\\rm{ej}}$', '$v^{\\rm{wind}}_{\\rm{ej}}$', '$Y_{\\mathrm{e}}^{\\rm{dyn}}$', '$Y_{\\mathrm{e}}^{\\rm{wind}}$']
            cols = ['luminosity_distance','inclination_EM', 'timeshift', 'log10_mej_dyn', 'log10_mej_wind', 'v_ej_dyn', 'v_ej_wind', 'Ye_dyn', 'Ye_wind']
            truth_val = [truth_row['luminosity_distance'].values[0], truth_row['inclination_EM'].values[0],
                        -1*ts, truth_row['log10_mej_dyn'].values[0], truth_row['log10_mej_wind'].values[0], truth_row['v_ej_dyn'].values[0], truth_row['v_ej_wind'].values[0], truth_row['Ye_dyn'].values[0], truth_row['Ye_wind'].values[0]]

        # limit to 32 bit float to save memory
        plot_data = samples[cols].astype(np.float32)

        fig = corner.corner(
            plot_data,
            truths=truth_val,
            truth_color='red',
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            label_kwargs={'fontsize': 14},
            smooth=1.0,
            bins=30,
            color='steelblue',
            hist_kwargs={'density': True},
            max_n_ticks=4,
            figsize=(10, 10),
            labelpad=0.03, 
        )

        # get quantiles for annotations
        axes = np.array(fig.axes).reshape((len(cols), len(cols)))
        for i, col in enumerate(cols):
            ax = axes[i, i]
            q16, q50, q84 = plot_data[col].quantile([0.16, 0.5, 0.84])
            inf_txt = rf"${q50:.3f}^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$"
            truth_val = -1*ts if col == 'timeshift' else log10_mej if col == 'log10_mej' else truth_row[col].values[0]
            ax.text(0.3, 1.03, inf_txt, transform=ax.transAxes, ha='center', fontsize=10)
            ax.text(0.8, 1.03, rf"{truth_val:.3f}", transform=ax.transAxes, ha='center', fontsize=10, color='red')

        fig.suptitle(title, fontsize=14)
        fig.savefig(out_path, dpi=150) # [OPTIM] DPI fixe pour contrôler la taille du fichier
        plt.close(fig)
        del plot_data
        gc.collect()

    '''
    1.5 IF RESAMPLING, GENERATE FIDUCIAL GW SAMPLES
    '''
    if args.resampling:
        # create GW samples (see https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/example_files/tools/gwem_resampling/gwsamples_generation.py for ref)
        gw_samples_file = f"{args.grid_dir}/GWsamples.dat" 
        if not os.path.exists(gw_samples_file):
            # Create GWsamples.dat if it does not exist (reuse code from gwsamples_generation.py)

            # load posterior file
            eos_post = np.loadtxt(args.eos_posterior)

            npts = 150000 
            Neos = 5000
            nparams = 3

            ############# [mass1,    mass2,   DL] adjust as needed
            params_low =  [1., 1., 10.]
            params_high = [2.25,      2.25,     200.]

            # 1) create dummy EOS samples with eos_post from nature paper
            EOS_raw = np.arange(0, Neos)  # the gwem_resampling will add one to this
            EOS_samples = np.random.choice(EOS_raw, p=eos_post, size=npts, replace=True)

            # 2) generate samples for masses and distance
            mass_1 = np.random.uniform(params_low[0], params_high[0], size=npts)
            mass_2 = np.random.uniform(params_low[1], params_high[1], size=npts)
            mass_1, mass_2 = np.maximum(mass_1, mass_2), np.minimum(mass_1, mass_2)  
            mass_ratio = mass_2 / mass_1  # mass ratio q < 1 convention is used
            chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
            lum_distance = np.random.uniform(params_low[2], params_high[2], size=npts)

            # 3) create pandas dataframe
            dataset = pd.DataFrame({'mass_1': mass_1, 'mass_2': mass_2, 'chirp_mass': chirp_mass, 'mass_ratio': mass_ratio, 'luminosity_distance': lum_distance, 'EOS': EOS_samples})

            # 4) save GWsamples.dat file
            dataset.to_csv(gw_samples_file, index=False, sep=' ')
            # ensure gw samples file is well formatted
            with open(gw_samples_file, 'r') as f:
                lines = f.readlines()
            with open(gw_samples_file, 'w') as f:
                for line in lines:
                    if len(line.split()) == 6:
                        f.write(line)

            print("GWsamples.dat created.")

    '''
    2. RUN THE ANALYSES WITH DIFFERENT TIME SHIFTS AND SAVE CORNER PLOTS
    '''
    # 1st load the data
    data = pd.read_csv(f"{BASE_DIR}/data{idx}.dat", delimiter=' ', header=None, dtype={0: str, 1: str, 2: DTYPE_FLOAT, 3: DTYPE_FLOAT})
    truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")

    current_prior_file = prior_file
    if args.restrict_dist_prior is not None:
        true_dl = truth['luminosity_distance'].values[0]
        # On évite d'avoir une distance minimale négative
        dl_min = max(1.0, true_dl - args.restrict_dist_prior) 
        dl_max = true_dl + args.restrict_dist_prior
        
        custom_prior_path = f"{BASE_DIR}/custom_dist_{idx}.prior"

        with open(prior_file, 'r') as f:
            prior_lines = f.readlines()
            
        with open(custom_prior_path, 'w') as f:
            for line in prior_lines:
                if line.startswith('luminosity_distance'): # change the D_L line
                    f.write(f"luminosity_distance = Uniform(minimum={dl_min}, maximum={dl_max}, name='luminosity_distance', latex_label='$D_L$')\n")
                else:
                    f.write(line)
        current_prior_file = custom_prior_path
        print(f"Custom prior created with restricted distance [{dl_min:.2f}, {dl_max:.2f}] Mpc at {current_prior_file}")
        if args.resampling:
            # also need to create a custom GW prior for the resampling
            custom_gw_prior_path = f"{BASE_DIR}/custom_GW.prior"
            with open(args.GW_prior, 'r') as f:
                gw_prior_lines = f.readlines()
            with open(custom_gw_prior_path, 'w') as f:
                for line in gw_prior_lines:
                    if line.startswith('luminosity_distance'): # change the D_L line
                        f.write(f"luminosity_distance = Uniform(minimum={dl_min}, maximum={dl_max}, name='luminosity_distance', latex_label='$D_L$')\n")
                    else:
                        f.write(line)
            print(f"Custom GW prior created with restricted distance [{dl_min:.2f}, {dl_max:.2f}] Mpc at {custom_gw_prior_path}")
            args.GW_prior = custom_gw_prior_path

    # sort by time
    data = data.sort_values(by=0, ascending=True).reset_index(drop=True)
    # stock the time list
    times = data[0].unique()

    # get the trigger time for the config file
    trigger_time = Time(times[0]).mjd # assuming the first time point is the trigger time
    # get the filter list for the config file
    filters = data[1].unique().tolist() # assuming the second column contains the filter

    # setup the config file for the command
    if MODEL != 'Bu2026_MLP':
        lc_config = f"""outdir : {BASE_DIR}/minus0
interpolation-type : tensorflow
em-model : {MODEL}
em-transient-class : svd
svd-path : {svd_path} 
label : minus0_{idx}
prior-file :  {current_prior_file}
nlive : {nlive}
sampler : pymultinest
light-curve-data : {BASE_DIR}/data{idx}.dat
trigger-time : {trigger_time}
bestfit : True
filters : {filters}
local-only : False
plot : True
xlim : [-2, 12]
ylim : [24, 16]
"""
    elif MODEL == 'Bu2026_MLP':
        lc_config = f"""outdir : {BASE_DIR}/minus0
interpolation-type : tensorflow 
em-model : Bu2026_MLP
em-transient-class : fiesta_kn
label : minus0_{idx}
prior-file :  {current_prior_file}
nlive : {nlive}
sampler : pymultinest
light-curve-data : {BASE_DIR}/data{idx}.dat
trigger-time : {trigger_time}
bestfit : True
filters : {filters}
local-only : False
plot : True
xlim : [-2, 12]
ylim : [24, 16]
"""
    with open(f"{BASE_DIR}/lc_config.yaml", 'w') as f:
        f.write(lc_config)

    print(f"Configuration file for the analysis with the full data has been saved at {BASE_DIR}/lc_config.yaml")
    print(f"Starting the analysis with the full data for injection {idx}...")
    # run the command
    cmd_lc = ["lightcurve-analysis", f"{BASE_DIR}/lc_config.yaml",
        ]
    subprocess.run(cmd_lc, check=True, cwd=BASE_DIR, env=env) 
    # plot the corner plot for the analysis with the full data
    samples = pd.read_csv(f"{BASE_DIR}/minus0/minus0_{idx}_posterior_samples.dat", delimiter=' ', dtype=DTYPE_FLOAT)
    ts = pd.to_datetime(times[0]) - pd.to_datetime(true_merger_time) # keep the same trigger time as for the original data to see how the timeshift evolves
    ts = ts.total_seconds() / (3600*24) # convert to days
    try:
        save_corner_plot(samples, truth, ts, f"{BASE_DIR}/minus0/corner_minus0_{idx}.png", "Injection analysis with model "+MODEL)
    except Exception as e:
        print(f"Error occurred while saving corner plot: {e}")
        erreur_detaillee = traceback.format_exc()
        print(f"Detailed error :\n{erreur_detaillee}")
    del samples, cmd_lc
    gc.collect()

    for j in range(minus_pts): # change the range as needed (up to the number of time points - 1) /!\ update prior bounds if needed
        if UL:
            filt_list = [data[1][i] for i in range(len(data)) if data[0][i] == times[j]]
            mag_per_filter = {band: data[data[1]==band][2].values for band in data[1].unique()}
            temp_df = pd.DataFrame()
            for f in filt_list:
                dm = mag_per_filter[f][0] - mag_per_filter[f][1]
                if dm > 0:
                    ul = mag_per_filter[f][0] - 0.75 * dm
                else:
                    ul = mag_per_filter[f][0] + 0.75 * dm
                UL = pd.DataFrame([[times[j], f, ul, np.inf]], columns=[0,1,2,3])
                temp_df = pd.concat([temp_df, UL], ignore_index=True)
        # drop the first time point
        dupl = [True if data[0][i] == times[j] else False for i in range(0, len(data))]
        data = data[~pd.Series(dupl)].reset_index(drop=True) # modify the original data for the next iteration  
        if UL:
            # add an UL
            data = pd.concat([data, temp_df], ignore_index=True)
            del temp_df
            gc.collect()
        data.to_csv(f"{BASE_DIR}/data_minus{j+1}.dat", sep=' ', index=False, header=False)
        # compute the timeshift
        try:
            ts = pd.to_datetime(data[0][0]) - pd.to_datetime(true_merger_time) # keep the same trigger time as for the original data to see how the timeshift evolves
            ts = ts.total_seconds() / (3600*24) # convert to days
        except Exception as e:
            print(f"Error occurred while computing timeshift: {e}")
            erreur_detaillee = traceback.format_exc()
            print(f"Detailed error:\n{erreur_detaillee}")
            if len(data[0]) <= 0:
                print(f"No data left after removing {j+1} time points. Stopping the analysis for further time shifts.")
                break
            ts = None
        if MODEL != 'Bu2026_MLP':
            lc_config = f"""outdir : {BASE_DIR}/minus{j+1}
interpolation-type : tensorflow
em-model : {MODEL}
em-transient-class : svd
svd-path : {svd_path} 
label : minus{j+1}_{idx}
prior-file :  {current_prior_file}
nlive : {nlive}
sampler : pymultinest
light-curve-data : {BASE_DIR}/data_minus{j+1}.dat
trigger-time : {Time(data[0][0]).mjd}
bestfit : True
filters : {filters}
local-only : False
plot : True
xlim : [-2, 12]
ylim : [24, 16]
"""
        elif MODEL == 'Bu2026_MLP':
            lc_config = f"""outdir : {BASE_DIR}/minus{j+1}
interpolation-type : tensorflow 
em-model : Bu2026_MLP
em-transient-class : fiesta_kn
label : minus{j+1}_{idx}
prior-file :  {current_prior_file}
nlive : {nlive}
sampler : pymultinest
light-curve-data : {BASE_DIR}/data_minus{j+1}.dat
trigger-time : {Time(data[0][0]).mjd}
bestfit : True
filters : {filters}
local-only : False
plot : True
xlim : [-2, 12]
ylim : [24, 16]
"""
        with open(f"{BASE_DIR}/lc_config_minus{j+1}.yaml", 'w') as f:
            f.write(lc_config)

        # run the command
        print(f"Configuration file for the analysis with the data minus {j+1} time points has been saved at {BASE_DIR}/lc_config_minus{j+1}.yaml")
        print(f"Starting the analysis with the data minus {j+1} time points for injection {idx}...")
        cmd_lc = ["lightcurve-analysis", f"{BASE_DIR}/lc_config_minus{j+1}.yaml",
            ]
        try: # in case the lc has not enough points left for the analysis to run, we catch the error and move on to the next one (also to avoid skipping resampling due to an error)
            subprocess.run(cmd_lc, check=True, cwd=BASE_DIR, env=env)
            # plot the corner plot for the analysis with the full data
            samples = pd.read_csv(f"{BASE_DIR}/minus{j+1}/minus{j+1}_{idx}_posterior_samples.dat", delimiter=' ', dtype=DTYPE_FLOAT)
            try:
                save_corner_plot(samples, truth, ts, f"{BASE_DIR}/minus{j+1}/corner_minus{j+1}_{idx}.png", "Injection analysis with model "+MODEL)
            except Exception as e:
                erreur_detaillee = traceback.format_exc()
                print(f"Detailed error :\n{erreur_detaillee}")
                print(f"Error occurred while saving corner plot: {e}")
        except Exception as e:
            print(f"Error occurred during the analysis for minus {j+1} time points: {e}")
            erreur_detaillee = traceback.format_exc()
            print(f"Detailed error :\n{erreur_detaillee}")
        del samples, cmd_lc
        gc.collect()

    '''
    3. RESAMPLING WITH GWEM-RESAMPLING (OPTIONAL)
    '''
    if args.resampling:
        for i in range(minus_pts+1): 
            print(f"Starting resampling for lc {idx} with minus {i}")
            # set up output directory and EM post file
            OUT_DIR = f"{BASE_DIR}/minus{i}/resamp"
            POST_FILE = f"{BASE_DIR}/minus{i}/minus{i}_{idx}_posterior_samples.dat"
            if not os.path.exists(POST_FILE):
                print(f"Posterior file {POST_FILE} not found. Please run the lightcurve analysis for lc {idx} for minus{i} before running the resampling.")
                continue
            if not os.path.exists(OUT_DIR):
                os.makedirs(OUT_DIR)
            # run the resampling
            cmd_resamp = ["gwem-resampling",
                    "--outdir", OUT_DIR,
                    "--GWsamples", gw_samples_file,
                    "--GWprior", args.GW_prior,
                    "--EMsamples", POST_FILE,
                    "--EOSpath", args.eos_path,
                    "--Neos", "5000",
                    "--EMprior", args.EM_prior,
                    "--nlive", str(nlive)
                ]
            subprocess.run(cmd_resamp, check=True, cwd=BASE_DIR, env=env)
            # do the plot now
            samples = pd.read_csv(f"{OUT_DIR}/posterior_samples.dat", delimiter=' ', dtype=DTYPE_FLOAT)
            truth = pd.read_csv(f"{BASE_DIR}/true{idx}.csv")
            true_q = truth['mass_2'].values[0] / truth['mass_1'].values[0]
            true_chirp = bilby.gw.conversion.component_masses_to_chirp_mass(truth['mass_1'].values[0], truth['mass_2'].values[0])
            truths_list = [true_chirp, true_q, None, None, truth['zeta'].values[0]] 
            samples['mass_1'] = samples['chirp_mass'] * (samples['mass_ratio']**(-3/5)) * ((1 + samples['mass_ratio'])**(1/5))
            samples['mass_2'] = samples['chirp_mass'] * (samples['mass_ratio']**(2/5)) * ((1 + samples['mass_ratio'])**(1/5))
            truths_list2 = [true_chirp, true_q, truth['mass_1'].values[0], truth['mass_2'].values[0]] # adjust as needed for the true EOS index

            fig = corner.corner(
            samples[['chirp_mass', 'mass_ratio', 'EOS', 'alpha', 'zeta']],
            labels=[r'$\mathcal{M}$', r'$q$', 'EOS', r'$\alpha$', r'$\zeta$'],
            truths=truths_list,
            truth_color='red',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={'fontsize': 14, 'pad': 12},
            label_kwargs={'fontsize': 14},
            smooth=1.0,
            bins=30,
            color='steelblue',
            hist_kwargs={'density': True},
            max_n_ticks=4,
            figsize=(12, 12),
            labelpad=0.03,
            )
            # get quantiles for annotations
            quantiles = {}
            for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'EOS', 'alpha', 'zeta']].columns):
                q16, q50, q84 = np.percentile(samples[col], [16, 50, 84])
                quantiles[col] = (q16, q50, q84)

            axes = np.array(fig.axes).reshape(5, 5)
            for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'EOS']].columns):
                ax = axes[j, j]   # Distribution marginale (diagonale)

                q16, q50, q84 = quantiles[col]
                minus = q50 - q16
                plus  = q84 - q50

                # Texte inféré (ligne 1)
                if col == 'EOS':
                    inferred_text = rf"${int(q50)}^{{+{int(plus)}}}_{{-{int(minus)}}}$"
                else:
                    inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

                # Injection (ligne 2)
                truth_val = truths_list[j]
                truth_text = rf"{truth_val:.3f}" if truth_val is not None else "N/A"

                # Clear the automatic title
                ax.set_title("")

                # Add manual 2 lines: one black, one red
                ax.text(
                    0.3, 1.03,
                    inferred_text,
                    ha='center', va='bottom',
                    fontsize=13,
                    transform=ax.transAxes,
                    color='black'
                )
                ax.text(
                    0.8, 1.03,
                    truth_text,
                    ha='center', va='bottom',
                    fontsize=13,
                    transform=ax.transAxes,
                    color='red'
                )
            fig.suptitle(f"Lc {idx} resampling posterior samples (minus {i})", y=1.02, fontsize=20)
            fig.savefig(f"{BASE_DIR}/minus{i}/{idx}_resampling_corner.png", bbox_inches='tight')
            print(f"Resampling and plotting for minus {i} completed for lc {idx}\nPlot saved to {BASE_DIR}/minus{i}/{idx}_resampling_corner.png")

            fig = corner.corner(
            samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']],
            labels=[r'$\mathcal{M}$', r'$q$', r'$M_1$', r'$M_2$'],
            truths=truths_list2,
            truth_color='red',
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={'fontsize': 14, 'pad': 12},
            label_kwargs={'fontsize': 14},
            smooth=1.0,
            bins=30,
            color='steelblue',
            hist_kwargs={'density': True},
            max_n_ticks=4,
            figsize=(12, 12),
            labelpad=0.03,
            )
            # get quantiles for annotations
            quantiles = {}
            for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
                q16, q50, q84 = np.percentile(samples[col], [16, 50, 84])
                quantiles[col] = (q16, q50, q84)

            axes = np.array(fig.axes).reshape(4, 4)
            for j, col in enumerate(samples[['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']].columns):
                ax = axes[j, j]   # Distribution marginale (diagonale)

                q16, q50, q84 = quantiles[col]
                minus = q50 - q16
                plus  = q84 - q50

                # Texte inféré (ligne 1)
                if col == 'EOS':
                    inferred_text = rf"${int(q50)}^{{+{int(plus)}}}_{{-{int(minus)}}}$"
                else:
                    inferred_text = rf"${q50:.3f}^{{+{plus:.3f}}}_{{-{minus:.3f}}}$"

                # Injection (ligne 2)
                truth_val = truths_list2[j]
                truth_text = rf"{truth_val:.3f}" if truth_val is not None else "N/A"

                # Clear the automatic title
                ax.set_title("")

                # Add manual 2 lines: one black, one red
                ax.text(
                    0.3, 1.03,
                    inferred_text,
                    ha='center', va='bottom',
                    fontsize=13,
                    transform=ax.transAxes,
                    color='black'
                )
                ax.text(
                    0.8, 1.03,
                    truth_text,
                    ha='center', va='bottom',
                    fontsize=13,
                    transform=ax.transAxes,
                    color='red'
                )
            fig.suptitle(f"Lc {idx} resampling posterior samples (minus {i})", y=1.02, fontsize=20)
            fig.savefig(f"{BASE_DIR}/minus{i}/{idx}_mass_corner.png", bbox_inches='tight')
            print(f"Resampling and plotting for minus {i} completed for lc {idx}\nPlot saved to {BASE_DIR}/minus{i}/{idx}_mass_corner.png")
    return 0

if __name__ == '__main__':
    sys.exit(main())
