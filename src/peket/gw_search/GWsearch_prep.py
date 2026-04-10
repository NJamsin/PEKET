#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import subprocess
import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
import os
from gwpy.timeseries import TimeSeries
import urllib.request
from gwosc.locate import get_urls
import glob
import yaml
import stat
import argparse
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import gc
from pycbc.types import TimeSeries as PyCBCTimeSeries
from pycbc.noise import noise_from_psd
import pycbc.noise
import pycbc.psd

def main():
    '''
    DEFINE THE NEEDED VAR FROM THE CONFIG FILE
    '''
    parser = argparse.ArgumentParser(description="PyCBC Pipeline Step")
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("--injection", action="store_true", help="Inject a fake signal")
    parser.add_argument("--template-bank", default=None, help="Path to the template bank file if you want to specify it instead of generating through the resampling posterior. This can be useful if you want to use a custom template bank or if you want to skip the template bank generation step for testing purposes.")
    parser.add_argument("--detector-threshold", default=0.5, type=float, help="Minimum antenna response required to launch the search. Default is 0.5, can be useful to avoid long search for time windows where the detectors are barely sensitive to the source.")
    parser.add_argument("--plot-antenna-pattern", default=None, action="store_true", help="If true, will generate an antenna pattern plot for the source location and the injection merger time. Only applied to injections because the merger time is needed for the antenna response.")
    args = parser.parse_args()

    # Dynamically find the Conda bin directory
    import sys
    bin_dir = os.path.dirname(sys.executable)
    pycbc_geom = os.path.join(bin_dir, "pycbc_geom_nonspinbank")
    pycbc_split = os.path.join(bin_dir, "pycbc_hdf5_splitbank")
    pycbc_inspiral = os.path.join(bin_dir, "pycbc_multi_inspiral") # For the bash script later 

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Directory
    BASE_DIR = os.path.abspath(config['Directory']['BASE_DIR'])
    SUFFIX = config['Directory']['run_name']

    # KN data
    KN_detection_date = config['KN_data']['first_detection']
    KN_ra = config['KN_data']['ra']
    KN_dec = config['KN_data']['dec']
    KN_EM_post = config['KN_data']['EM_post_file']
    KN_resamp_post = config['KN_data']['RESAMP_post_file']

    # GW search
    NUM_SPLITS = config['GW_search']['num_splits']
    max_window_size = config['GW_search']['window_size']

    '''
    Step 0: Create the output directory if it doesn't exist
    '''
    os.makedirs(BASE_DIR, exist_ok=True)

    '''
    Step 1: generate the template bank
    '''
    if args.template_bank:
        print(f"Using user-provided template bank at {args.template_bank}. Skipping generation step.")
        OUT_FILE_BANK = args.template_bank
    else:
        sample = pd.read_csv(KN_resamp_post, delimiter=' ', dtype=np.float32)

        # transform the chirp mass and mass ratio to component masses
        m1 = sample['chirp_mass'].values * (1 + sample['mass_ratio'].values)**(1/5) / (sample['mass_ratio'].values)**(3/5)
        m2 = sample['chirp_mass'].values * (1 + sample['mass_ratio'].values)**(1/5) * (sample['mass_ratio'].values)**(2/5)

        # use that to generate the template bank:
        OUT_FILE_BANK = f"{BASE_DIR}/{SUFFIX}_tmplt.hdf"
        # Check if the bank file already exists, if so, skip the generation step
        if os.path.exists(OUT_FILE_BANK):
            print(f"Template bank file {OUT_FILE_BANK} already exists. Skipping generation.")
        else:
            CMD = [pycbc_geom,
                "--min-mass1", f"{np.percentile(m1,16):.4f}",     
                "--max-mass1",  f"{np.percentile(m1, 84):.4f}",     
                "--min-mass2", f"{np.percentile(m2, 16):.4f}",     
                "--max-mass2", f"{np.percentile(m2, 84):.4f}",     
                "--f-low", "30.0",     
                "--f-upper", "2048.0", 
                "--delta-f", "0.01",     
                "--pn-order", "threePointFivePN",     
                "--min-match", "0.97",
                "--psd-model", "aLIGOZeroDetHighPower",     
                "--output-file", f"{OUT_FILE_BANK}", 
                "--verbose"]
            print(f"Generating non-spinning geometric template bank")
            subprocess.run(CMD, check=True, cwd=BASE_DIR)
            print(f"Template bank generated and saved as '{OUT_FILE_BANK}'")

    # Open the geometric bank file to get the numb of template
    bank = h5py.File(OUT_FILE_BANK, 'r')
    num_templates = len(bank['mass1'][:])

    # split it
    TEMPLATE_PER_BANK = int(np.ceil(num_templates / NUM_SPLITS))
    OUT_SPLIT = f"{BASE_DIR}/{SUFFIX}_split"
    os.makedirs(OUT_SPLIT, exist_ok=True)

    # check if the split bank files already exist, if so, skip the splitting step
    existing_split_files = glob.glob(f"{OUT_SPLIT}/split_bank_*.hdf")
    if len(existing_split_files) == NUM_SPLITS:
        print(f"All split bank files already exist in {OUT_SPLIT}. Skipping splitting.")
    else:
        split_CMD = [pycbc_split,
            "--bank-file", f"{OUT_FILE_BANK}",
            "--output-prefix", f"{OUT_SPLIT}/split_bank_",
            "--templates-per-bank", f"{TEMPLATE_PER_BANK}"]

        subprocess.run(split_CMD, check=True, cwd=BASE_DIR)
        print(f"Split template banks generated and saved in '{OUT_SPLIT}'")

    # loop over the split template bank to plot 
    import matplotlib.colors as mcolors
    norm=mcolors.Normalize(vmin=0, vmax=NUM_SPLITS)
    cmap = plt.get_cmap('gist_rainbow')
    col = cmap(np.linspace(0,1,NUM_SPLITS))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(NUM_SPLITS):
        bank = h5py.File(f"{OUT_SPLIT}/split_bank_{i}.hdf", 'r')
        # Plot the template bank masses
        m1 = bank['mass1'][:]
        m2 = bank['mass2'][:]

        ax.scatter(m1, m2, s=5, color=col[i])
    cbar=plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Split Bank Index')
    cbar.set_ticks(np.arange(0.5,NUM_SPLITS+0.5,1))
    ax.set_xlabel(r'Mass 1 ($M_{\odot}$)')
    ax.set_ylabel(r'Mass 2 ($M_{\odot}$)')
    ax.set_title('Template Bank Mass Distribution')
    ax.grid(True)
    if NUM_SPLITS <= 20: # to avoid overcrowding the colorbar ticks
        cbar.set_ticklabels([str(idx) for idx in np.arange(0.,NUM_SPLITS,1)])
    else:
        cbar.set_ticklabels([str(idx) for idx in np.arange(0.,NUM_SPLITS,1)], fontsize=6)
    PLOT_DIR = f"{BASE_DIR}/plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(f"{PLOT_DIR}/{SUFFIX}_template_bank.png")
    plt.close(fig)
    print(f"Template bank mass distribution plot saved as '{SUFFIX}_template_bank.png'")

    '''
    Step 2: define the search window
    '''
    # Load the EM posterior samples
    EM_samp = pd.read_csv(KN_EM_post, delimiter=' ', dtype=np.float32)

    # 1. convert the first detection time to mjd time
    KN_t0 = Time(KN_detection_date, format='isot', scale='utc').mjd

    # 2. Calculate 1 sigma interval (16th and 84th percentiles) of the timeshift samples
    p16, p50, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])

    # 3. Define the search window around the median timeshift, extending to the 1sigma interval
    t_start = KN_t0 + p16
    t_end = KN_t0 + p84

    # convert to gps time
    time_mjd = (t_start, t_end)
    time_gps = Time(time_mjd, format='mjd').gps

    print("\nDefined search window based on EM posterior samples:")
    print(f"MJD time: {time_mjd}")
    print(f"GPS time: {int(time_gps[0])} to {int(time_gps[1])}")

    '''
    Step 3: define sub windows
    '''
    num_banks = NUM_SPLITS

    global_start = int(time_gps[0])
    global_end = int(time_gps[1])
    chunk_length = max_window_size
    overlap = 16 # Accounts for 8s padding at start and 8s at end

    WINDOW_FILE = f"{BASE_DIR}/{SUFFIX}_windows.txt"

    with open(WINDOW_FILE, 'w') as f:
        for bank in range(num_banks):
            current_start = global_start
            while current_start < global_end:
                current_end = min(current_start + chunk_length, global_end)
                tt = (current_start + current_end) // 2 #for the antenna pattern
                # Write: BANK_NUM START_TIME END_TIME
                f.write(f"{bank} {current_start} {current_end} {tt}\n")
                
                if current_end == global_end:
                    break
                
                # Step back by the overlap amount for the next chunk
                current_start = current_end - overlap

    print(f"Generated {WINDOW_FILE}")

    '''
    Step 4: fetch and clean GW data 
    '''
    DATA_DIR = f"{BASE_DIR}/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    h1_cache = f"{DATA_DIR}/{SUFFIX}_H1.lcf"
    l1_cache = f"{DATA_DIR}/{SUFFIX}_L1.lcf"
    detectors = ['H1', 'L1']

    if os.path.exists(h1_cache) and os.path.exists(l1_cache):
        print(f"Cleaned and merged files {h1_cache} and {l1_cache} already exist. Skipping download and preparation.")
    else:
        def robust_get_urls(detector, start, end):
            from gwosc.locate import get_urls
            urls = []
            chunk_size = 86400  # 1 day in seconds
            current_start = start
            
            while current_start < end:
                current_end = min(current_start + chunk_size, end)
                try:
                    # Ask for just this chunk
                    chunk_urls = get_urls(detector, current_start, current_end, format='gwf', sample_rate=4096)
                    for u in chunk_urls:
                        if '4096.gwf' in u and u not in urls: # ensure we only get 4096 files and avoid duplicates (for example the event-specific files)
                            urls.append(u)
                except ValueError:
                    print(f" -> Warning: No public GWOSC data found for {detector} between {int(current_start)} and {int(current_end)}.")
                
                current_start = current_end
                
            return urls
        # The exact GPS times from your bash command
        gps_start = int(time_gps[0]) - 32 # start of the window -32s for padding
        gps_end = int(time_gps[1]) + 32 # end of the window +32s for padding

        downloaded_files = {'H1': [], 'L1': []}

        for ifo in detectors:
            print(f"Locating 4kHz data for {ifo}...")
            # Fetch URLs for the .gwf frame files at 4096 Hz
            urls = robust_get_urls(ifo, gps_start, gps_end)
            
            for url in urls:
                filename = url.split('/')[-1]
                filepath = os.path.join(DATA_DIR, filename)
                
                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                else:
                    print(f"{filename} already exists locally. Skipping.")
                    
                downloaded_files[ifo].append(filepath)

        print("\n--- Download Complete ---")
        print(f"H1 Frame File(s): {','.join(downloaded_files['H1'])}")
        print(f"L1 Frame File(s): {','.join(downloaded_files['L1'])}")

        # clean and merge
        # 1. list og files for each detector
        fichiers_h1 = downloaded_files['H1']

        fichiers_l1 = downloaded_files['L1']

        def preparer_donnees(fichiers, canal, ifo, t_start, t_end, chunk_size=4096):
            print(f"Processing strain data files for {canal}...")

            if args.injection: # prepare the injected signal before processing the data
                print(f" -> Generating injection waveform for {ifo}...")
                inj = config['Injection']
                center_time = t_start + (t_end - t_start) / 2
                merger_time = center_time + inj['time_offset']
                
                hp, hc = get_td_waveform(
                    approximant=inj['approximant'], 
                    mass1=inj['mass1'], 
                    mass2=inj['mass2'],
                    distance=inj['distance'], 
                    delta_t=1.0/4096.0, # supposedely same as data.dt.value
                    f_lower=30.0
                )
                hp.start_time += merger_time
                hc.start_time += merger_time
                
                det = Detector(ifo)
                f_plus, f_cross = det.antenna_pattern(inj['ra'], inj['dec'], inj['polarization'], merger_time)
                # Calculate the total response (Scale of 0 to 1)
                total_response = np.sqrt(f_plus**2 + f_cross**2)
                if args.detector_threshold and total_response < args.detector_threshold:
                    print(f"    *** Antenna response for this injection is {total_response:.2f}, which is below the specified threshold of {args.detector_threshold}. Stopping the search. ***")
                    sys.exit(0)
                ht = det.project_wave(hp, hc, inj['ra'], inj['dec'], inj['polarization'], reference_time=merger_time)
                
                ht_start_time = float(ht.start_time)
                # Calculate absolute end time of the waveform
                ht_end_time = ht_start_time + (len(ht) / ht.sample_rate)

            cache_entries = []
            
            current_start = t_start
            chunk_idx = 0

            while current_start < t_end:
                current_end = min(current_start + chunk_size, t_end)
                # Format: Observatory(H/L) - IFO_Tag - StartTime - Duration .gwf
                out_name = f"{DATA_DIR}/{ifo[0]}-{ifo}_{SUFFIX}-{int(current_start)}-{int(current_end-current_start)}.gwf"
                
                print(f" -> Chunk {chunk_idx}: {int(current_start)} to {int(current_end)}")

                # find files overlapping with the current chunk (we want to pass only those to gwpy to minimize padding issues and speed up the reading)
                overlapping_files = []
                for f in fichiers:
                    basename = os.path.basename(f)
                    # Example format: H-H1_GWOSC_O3b_4KHZ_R1-1262125056-4096.gwf
                    parts = basename.replace('.gwf', '').split('-')
                    file_start = int(parts[-2])
                    file_duration = int(parts[-1])
                    file_end = file_start + file_duration
                    
                    if file_start < current_end and file_end > current_start:
                        overlapping_files.append(f)

                # 2. read files or replace with noise if no files or if gwpy read fails
                if not overlapping_files:
                    # The detector was offline for this entire chunk. Skip reading completely!
                    print(f"    *** No data files found for this chunk. Synthesizing noise... ***")
                    duration = current_end - current_start
                    data = TimeSeries(np.random.normal(0, 1e-22, int(duration * 4096)), 
                                      t0=current_start, sample_rate=4096, name=canal)
                else:
                    try:
                        # Pass ONLY the overlapping files, preventing massive padding leaks
                        data = TimeSeries.read(overlapping_files, canal, start=current_start, end=current_end, pad=np.nan)
                    except Exception as e:
                        print(f"    *** gwpy read failed: {e}. Synthesizing noise... ***")
                        duration = current_end - current_start
                        data = TimeSeries(np.random.normal(0, 1e-22, int(duration * 4096)), 
                                          t0=current_start, sample_rate=4096, name=canal)
                
                # Clean NaNs and Zeros 
                zero_mask = (data.value == 0.0)
                data.value[zero_mask] = np.nan
                print("Replacing NaN values with Gaussian noise...")
                nan_mask = np.isnan(data.value)
                if np.any(nan_mask):
                    valid_data = data.value[~nan_mask]
                    if len(valid_data) > 0:
                        std_bruit = np.std(valid_data) * 1e-3 # inject noise at 0.1% of the std because of the bucket
                    else:
                        # Fallback if the ENTIRE file was empty/zeros
                        std_bruit = 1e-22 # low noise but we loose the "realistic" aspect of the noise (no 100Hz bucket)
                    data.value[nan_mask] = np.random.normal(0, std_bruit, size=np.sum(nan_mask))
                    print(f" -> {np.sum(nan_mask)} values corrected. Gaussian noise injected with std={std_bruit:.2e}.")
                
                data.name = canal

                if args.injection:
                    if ht_end_time > current_start and ht_start_time < current_end:
                        print(f"    -> Adding injection to this chunk...")
                        pycbc_data = data.to_pycbc()
                        # add the injection to the data with pycbc built in method (better than my numpy slicing)
                        pycbc_data = pycbc_data.add_into(ht)
                        # convert back to gwpy TimeSeries for saving
                        try:
                            data = TimeSeries(pycbc_data.numpy(), t0=data.t0.value, dt=pycbc_data.delta_t, channel=canal)
                        except Exception as e:
                            print(f"    *** Failed to convert back to gwpy TimeSeries: {e} ***")
                        print(f"       Injection added! (Merger time: {merger_time}) in chunk {chunk_idx} ({int(current_start)} to {int(current_end)})")
                        # save to a txt file the merger time for later use in the search 
                        with open(f"{BASE_DIR}/{SUFFIX}_injection_time.txt", "w") as f:
                            f.write(f"{merger_time}\n")

                # Write chunk to disk
                data.write(out_name, format='gwf')
                
                # Create LAL Cache Entry Format
                ifo_letter = ifo[0]
                duration = current_end - current_start
                cache_entries.append(f"{ifo_letter} {canal.replace(':', '_')} {int(current_start)} {int(duration)} file://localhost{os.path.abspath(out_name)}")
                
                # Force Memory Cleanup
                del data
                gc.collect()
                
                current_start = current_end
                chunk_idx += 1
                
            # Write out the cache file for PyCBC
            cache_file = f"{DATA_DIR}/{SUFFIX}_{ifo}.lcf"
            with open(cache_file, "w") as f:
                f.write("\n".join(cache_entries) + "\n")
            if args.injection:
                return cache_file, merger_time
            else:
                return cache_file, None

        # because your PyCBC command has a padding of 8 seconds.
        t_start_pycbc = gps_start - 16
        t_end_pycbc = gps_end + 16

        h1_cache, merger_time = preparer_donnees(fichiers_h1, "H1:GWOSC-4KHZ_R1_STRAIN", "H1", t_start_pycbc, t_end_pycbc)
        l1_cache, _ = preparer_donnees(fichiers_l1, "L1:GWOSC-4KHZ_R1_STRAIN", "L1", t_start_pycbc, t_end_pycbc)
        print("Completed! The files are ready for PyCBC.")

        # delete the og file to clean some spaces
        for ifo in detectors:
            for filepath in downloaded_files[ifo]:
                os.remove(filepath)
                print(f"Deleted {filepath}")
        
        # plot the antenna pattern for the injection if requested
        if args.plot_antenna_pattern and args.injection:
            def plot_antenna_pattern(ifo, ra, dec, merger_time, save_path):
                det = Detector(ifo)
                # define a sky position grid
                ra_grid = np.linspace(-np.pi, np.pi, 200)
                dec_grid = np.linspace(-np.pi/2, np.pi/2, 100)
                RA, DEC = np.meshgrid(ra_grid, dec_grid)
                RA_pycbc = RA + np.pi # pycbc convention is 0 to 2pi for RA instead of -pi to pi
                # compute the antenna pattern response for each point in the sky grid
                response_map = np.zeros_like(RA)

                for i in range(RA.shape[0]):
                    for j in range(RA.shape[1]):
                        # Calculate F+ and Fx
                        f_plus, f_cross = det.antenna_pattern(RA_pycbc[i,j], DEC[i,j], 0, merger_time)
                        # Total response
                        response_map[i,j] = np.sqrt(f_plus**2 + f_cross**2)
                # Plotting
                inj_ra_plot = ra - np.pi
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='mollweide')

                # Plot the heat map
                c = ax.pcolormesh(RA, DEC, response_map, cmap='viridis', shading='auto')

                # Plot your injection as a red cross
                ax.plot(inj_ra_plot, dec, 'rx', markersize=5, markeredgewidth=3, label='Injection Location')

                # Formatting
                ax.set_title(f"{ifo} Antenna Response Map at GPS {merger_time}", pad=20)
                ax.grid(True, linestyle='--', alpha=0.5)
                plt.colorbar(c, label='Normalized Total Detector Sensitivity', orientation='horizontal', pad=0.1, aspect=30)
                ax.legend(loc='upper right', numpoints=1)
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Antenna pattern plot saved as '{save_path}'")
            for ifo in detectors:
                plot_antenna_pattern(ifo, KN_ra, KN_dec, merger_time, f"{PLOT_DIR}/{SUFFIX}_{ifo}_antenna_pattern.png")
    '''
    Step 5: Create the .sh and .sub needed to run the PyCBC search on the cluster.
    '''
    # Define the directory for the .sh and .sub files
    CONDOR_FILES = f"{BASE_DIR}/sub_files"
    OUT_DIR = f"{BASE_DIR}/out"
    LOG_DIR = f"{BASE_DIR}/logs"
    # Create the directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CONDOR_FILES, exist_ok=True)
    # set the file names
    sh_filename = f"{CONDOR_FILES}/run_split_search.sh"
    sub_filename = f"{CONDOR_FILES}/split_search.sub"

    ENV_PREFIX = sys.prefix

    # 1. Content for the bash script
    sh_content = f"""#!/bin/bash

    export PATH="{ENV_PREFIX}/bin:$PATH" 

    BANK_NUM=$1
    START_TIME=$2
    END_TIME=$3
    TT=$4

    {pycbc_inspiral} \
        -v \
        --instruments H1 L1 \
        --bank-file {OUT_SPLIT}/split_bank_${{BANK_NUM}}.hdf \
        --channel-name H1:GWOSC-4KHZ_R1_STRAIN L1:GWOSC-4KHZ_R1_STRAIN \
        --frame-cache H1:{h1_cache} L1:{l1_cache} \
        --gps-start-time H1:${{START_TIME}} L1:${{START_TIME}} h1:${{START_TIME}} l1:${{START_TIME}} \
        --gps-end-time H1:${{END_TIME}} L1:${{END_TIME}} h1:${{END_TIME}} l1:${{END_TIME}} \
        --ra {KN_ra} \
        --dec {KN_dec} \
        --trigger-time ${{TT}} \
        --low-frequency-cutoff 30.0 \
        --approximant TaylorF2 \
        --order 7 \
        --sample-rate H1:4096 L1:4096 h1:4096 l1:4096 \
        --pad-data H1:8 L1:8 h1:8 l1:8 \
        --segment-length H1:256 L1:256 h1:256 l1:256 \
        --segment-start-pad H1:8 L1:8 h1:8 l1:8 \
        --segment-end-pad H1:8 L1:8 h1:8 l1:8 \
        --psd-estimation H1:median L1:median h1:median l1:median \
        --psd-segment-length H1:16 L1:16 h1:16 l1:16 \
        --psd-segment-stride H1:8 L1:8 h1:8 l1:8 \
        --psd-inverse-length H1:16 L1:16 h1:16 l1:16 \
        --strain-high-pass H1:20 L1:20 h1:20 l1:20 \
        --autogating-threshold H1:50 L1:50 h1:50 l1:50 \
        --autogating-cluster H1:0.5 L1:0.5 h1:0.5 l1:0.5 \
        --autogating-width H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
        --autogating-pad H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
        --autogating-taper H1:0.25 L1:0.25 h1:0.25 l1:0.25 \
        --coinc-threshold 5.5 \
        --sngl-snr-threshold 4.0 \
        --chisq-bins 16 \
        --cluster-method window \
        --cluster-window 1.0 \
        --output {OUT_DIR}/{SUFFIX}triggers_bank${{BANK_NUM}}_${{START_TIME}}-${{END_TIME}}.hdf
    """

    # 2. Content for the HTCondor submit file
    sub_content = f"""executable = {sh_filename}
    universe   = vanilla

    # Pass the three variables from the text file to the bash script
    arguments  = "$(bank) $(start) $(end) $(tt)"

    # Ensure logs don't overwrite each other + change path (I went for absolute path just to be sure)
    output     = {LOG_DIR}/{SUFFIX}_search_$(bank)_$(start).out
    error      = {LOG_DIR}/{SUFFIX}_search_$(bank)_$(start).err
    log        = {LOG_DIR}/{SUFFIX}_search_cluster.log

    # Request resources (adjust these according to your cluster's limits)
    request_cpus   = 1
    request_memory = 4GB
    request_disk   = 1MB

    # Queue a job for every line in the text file
    queue bank, start, end, tt from {WINDOW_FILE}
    """

    # Write the bash script to disk
    with open(sh_filename, "w") as f:
        f.write(sh_content.strip() + "\n")

    # Automatically make the bash script executable (equivalent to running 'chmod +x')
    st = os.stat(sh_filename)
    os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)

    # Write the submit file to disk
    with open(sub_filename, "w") as f:
        f.write(sub_content.strip() + "\n")

    print(f"Successfully generated '{sh_filename}' and '{sub_filename}'")
    print("Search preparation complete!")
    return 0

if __name__ == '__main__':
    import re
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())