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
from gwosc.timeline import get_segments
import glob
import yaml
import sys
import stat
import argparse
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import gc
from pycbc.types import TimeSeries as PyCBCTimeSeries
from pycbc.noise import noise_from_psd
import pycbc.noise
import pycbc.psd

# Moved functions outside the main to be able to call it outside GWsearch
def preparer_donnees(args, config, DATA_DIR, SUFFIX, BASE_DIR, fichiers, canal, ifo, t_start, t_end, chunk_size=4096):
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
            with open(cache_file, "a") as f:
                f.write("\n".join(cache_entries) + "\n")
            if args.injection:
                return cache_file, merger_time
            else:
                return cache_file, None

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


def get_coincident_segments(detectors, start, end, min_ifos=2, flag_suffix="DATA", margin=0):
    """
    Interroge GWOSC (gwosc.timeline.get_segments) pour connaitre les segments
    "science" de chaque detecteur sur [start, end), puis fait un sweep-line pour
    trouver les segments contigus ou AU MOINS `min_ifos` detecteurs sont actifs
    simultanement.

    Retourne une liste de tuples (seg_start, seg_end, (ifo1, ifo2, ...)) triee
    par seg_start. Le tuple d'ifos donne exactement quels detecteurs utiliser
    pour --instruments sur ce segment (2 ou 3 selon la coincidence).

    flag_suffix: "DATA" pour juste la disponibilite des donnees, ou "CBC_CAT1"
    (voire CAT2/CAT3 en cumulant les segments manuellement) pour un veto qualite
    plus strict adapte a une recherche CBC.

    margin: retire `margin` secondes de part et d'autre de chaque segment brut
    par detecteur avant l'intersection, pour garder une marge de securite par
    rapport au padding/PSD (pad-data + psd-inverse-length typiquement).
    """
    events = []
    for ifo in detectors:
        flag = f"{ifo}_{flag_suffix}"
        try:
            segs = get_segments(flag, int(start), int(end))
        except Exception as e:
            print(f" -> Warning: could not fetch segments for flag {flag}: {e}")
            segs = []
        for s, e_ in segs:
            s_m = s + margin
            e_m = e_ - margin
            if e_m <= s_m:
                continue
            s_c = max(s_m, start)
            e_c = min(e_m, end)
            if e_c <= s_c:
                continue
            events.append((s_c, 1, ifo))
            events.append((e_c, -1, ifo))

    # a un instant identique, on traite d'abord les fins de segment (-1) puis
    # les debuts (1), pour rester conservateur aux bornes.
    events.sort(key=lambda x: (x[0], x[1]))

    active = set()
    raw_segments = []
    prev_t = None
    for t, typ, ifo in events:
        if prev_t is not None and t > prev_t and active:
            raw_segments.append((prev_t, t, tuple(sorted(active))))
        if typ == 1:
            active.add(ifo)
        else:
            active.discard(ifo)
        prev_t = t

    coincident = [(s, e, ifos) for (s, e, ifos) in raw_segments if len(ifos) >= min_ifos]
    return coincident


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
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"], help="Size of the time window to be searched around the expected trigger time, in sigmas. Default is 1.")
    parser.add_argument("--tmplt-sigma", default='1', choices=['1','2','3', "full"], help="Size of the the template bank to be used for template bank generation around the expected trigger time, in sigmas. Default is 1. /!\\ If you specify a custom template bank with --template-bank, this argument will be ignored.")
    parser.add_argument("--ldg-tag", default=None, help="The \"accounting_group\" tag required for submitting to the LDG cluster. If not specified, the pipeline will be generated without the tag.")
    parser.add_argument("--chunk-size", default=3000, type=int)
    parser.add_argument("--detectors", default="H1,L1,V1", help="Comma-separated list of detectors to consider for the search (candidates). Coincidence filtering will still only keep jobs where at least --min-ifos of them were actually on. Default: H1,L1,V1")
    parser.add_argument("--min-ifos", default=2, type=int, help="Minimum number of detectors that must be simultaneously in science mode for a time segment to be analyzed. Default is 2.")
    parser.add_argument("--dq-flag", default="DATA", help="GWOSC timeline flag suffix used to build the per-detector segment lists, e.g. 'DATA' (just strain availability) or 'CBC_CAT1' (adds category-1 data quality vetoes). Default: DATA")
    parser.add_argument("--segment-margin", default=40, type=int, help="Seconds trimmed off both ends of each raw per-detector science segment before computing coincidence, to keep the pad-data/PSD-estimation region inside good data. Default: 40 (8s pad-data + 16s psd-inverse-length + safety margin).")
    parser.add_argument("--min-analysis-length", default=64, type=int, help="Minimum usable duration (s) of a coincident chunk for it to be worth submitting as a job. Shorter leftover pieces are dropped. Default: 64")
    args = parser.parse_args()

    detectors = [d.strip() for d in args.detectors.split(',') if d.strip()]

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
            # used the percentiles dynamically
            if args.tmplt_sigma == "full":
                low_m1 = np.min(m1)
                low_m2 = np.min(m2)
                high_m1 = np.max(m1)
                high_m2 = np.max(m2)
            elif args.tmplt_sigma == '1':
                low_m1 = np.percentile(m1, 15.865)
                low_m2 = np.percentile(m2, 15.865)
                high_m1 = np.percentile(m1, 84.135)
                high_m2 = np.percentile(m2, 84.135)
            elif args.tmplt_sigma == '2':
                low_m1 = np.percentile(m1, 2.275)
                low_m2 = np.percentile(m2, 2.275)
                high_m1 = np.percentile(m1, 97.725)
                high_m2 = np.percentile(m2, 97.725)
            elif args.tmplt_sigma == '3':
                low_m1 = np.percentile(m1, 0.135)
                low_m2 = np.percentile(m2, 0.135)
                high_m1 = np.percentile(m1, 99.865)
                high_m2 = np.percentile(m2, 99.865)
            CMD = [pycbc_geom,
                "--min-mass1", f"{low_m1:.4f}",     
                "--max-mass1",  f"{high_m1:.4f}",     
                "--min-mass2", f"{low_m2:.4f}",     
                "--max-mass2", f"{high_m2:.4f}",     
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
        b_file = f"{OUT_SPLIT}/split_bank_{i}.hdf"
        if not os.path.exists(b_file):
            print(f"Warning: Expected split bank file '{b_file}' not found. Skipping this bank for plotting.")
            continue
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
        cbar.set_ticklabels([str(int(idx)) for idx in np.arange(0.,NUM_SPLITS,1)])
    else:
        cbar.set_ticklabels([str(int(idx)) for idx in np.arange(0.,NUM_SPLITS,1)], fontsize=6)
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
    if args.OSW_sigma == "full":
        p16 = np.min(EM_samp['timeshift'])
        p84 = np.max(EM_samp['timeshift'])
    elif args.OSW_sigma == '1':
        p16, p84 = np.percentile(EM_samp['timeshift'], [15.865, 84.135])
    elif args.OSW_sigma == '2':
        p16, p84 = np.percentile(EM_samp['timeshift'], [2.275, 97.725])
    elif args.OSW_sigma == '3':
        p16, p84 = np.percentile(EM_samp['timeshift'], [0.135, 99.865])

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
    Step 3: define sub windows, restricted to segments where at least
    args.min_ifos detectors were simultaneously in science mode.
    '''
    num_banks = NUM_SPLITS

    global_start = int(time_gps[0])
    global_end = int(time_gps[1])
    chunk_length = max_window_size
    overlap = 16 # Accounts for 8s padding at start and 8s at end

    print(f"\nQuerying GWOSC timeline ({args.dq_flag} flag) for {detectors} between {global_start} and {global_end}...")
    coincident_segments = get_coincident_segments(
        detectors, global_start, global_end,
        min_ifos=args.min_ifos, flag_suffix=args.dq_flag, margin=args.segment_margin
    )

    total_requested = global_end - global_start
    total_coincident = sum(e - s for s, e, _ in coincident_segments)
    print(f" -> {len(coincident_segments)} coincident segment(s) found, "
          f"{total_coincident}s usable out of {total_requested}s requested "
          f"({100.0 * total_coincident / total_requested:.1f}%).")
    for s, e, ifos in coincident_segments:
        print(f"    [{s} - {e}] ({e - s}s) -> {'+'.join(ifos)}")

    if not coincident_segments:
        print(f" *** No segment with at least {args.min_ifos} detectors on was found in this window. Aborting. ***")
        sys.exit(1)

    WINDOW_FILE = f"{BASE_DIR}/{SUFFIX}_windows.txt"

    n_jobs = 0
    with open(WINDOW_FILE, 'w') as f:
        for bank in range(num_banks):
            for seg_start, seg_end, ifos in coincident_segments:
                current_start = seg_start
                while current_start < seg_end:
                    current_end = min(current_start + chunk_length, seg_end)
                    if current_end - current_start < args.min_analysis_length:
                        # leftover piece too short to be worth analyzing, drop it
                        break
                    tt = (current_start + current_end) // 2  # for the antenna pattern
                    ifo_str = ",".join(ifos)
                    # Write: BANK_NUM START_TIME END_TIME TT IFOS
                    f.write(f"{bank} {current_start} {current_end} {tt} {ifo_str}\n")
                    n_jobs += 1

                    if current_end == seg_end:
                        break

                    # Step back by the overlap amount for the next chunk
                    current_start = current_end - overlap

    print(f"Generated {WINDOW_FILE} ({n_jobs} bank/window jobs)")

    '''
    Step 4: fetch and clean GW data 
    '''
    DATA_DIR = f"{BASE_DIR}/data"
    os.makedirs(DATA_DIR, exist_ok=True)

    caches = {ifo: f"{DATA_DIR}/{SUFFIX}_{ifo}.lcf" for ifo in detectors}

    if all(os.path.exists(c) for c in caches.values()):
        print(f"Cleaned and merged files {list(caches.values())} already exist. Skipping download and preparation.")
    else:
        # The exact GPS times from your bash command
        gps_start = int(time_gps[0]) - 32 # start of the window -32s for padding
        gps_end = int(time_gps[1]) + 32 # end of the window +32s for padding

        downloaded_files = {ifo: [] for ifo in detectors}

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
        for ifo in detectors:
            print(f"{ifo} Frame File(s): {','.join(downloaded_files[ifo])}")

        # because your PyCBC command has a padding of 8 seconds.
        t_start_pycbc = gps_start - 16
        t_end_pycbc = gps_end + 16

        merger_time = None
        for ifo in detectors:
            cache_file, mt = preparer_donnees(
                args, config, DATA_DIR, SUFFIX, BASE_DIR, downloaded_files[ifo],
                f"{ifo}:GWOSC-4KHZ_R1_STRAIN", ifo, t_start_pycbc, t_end_pycbc
            )
            caches[ifo] = cache_file
            if mt is not None:
                merger_time = mt
        print("Completed! The files are ready for PyCBC.")

        # delete the og file to clean some spaces
        for ifo in detectors:
            for filepath in downloaded_files[ifo]:
                os.remove(filepath)
                print(f"Deleted {filepath}")
        
        # plot the antenna pattern for the injection if requested
        if args.plot_antenna_pattern and args.injection:
            for ifo in detectors:
                plot_antenna_pattern(ifo, KN_ra, KN_dec, merger_time, f"{PLOT_DIR}/{SUFFIX}_{ifo}_antenna_pattern.png")
    '''
    Step 5: Create the .sh and .sub needed to run the PyCBC search on the cluster.
    The bash script now takes a 5th argument (comma-separated IFO list, e.g.
    "H1,L1" or "H1,L1,V1") and builds the pycbc_multi_inspiral argument lists
    dynamically, so the same job template works for any subset of `detectors`.
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

    # frame-cache declarations for the bash associative array
    frame_cache_decls = "\n    ".join(f'FRAME_CACHES[{ifo}]="{caches[ifo]}"' for ifo in detectors)

    # 1. Content for the bash script
    sh_content = f"""#!/bin/bash

    export PATH="{ENV_PREFIX}/bin:$PATH" 

    BANK_NUM=$1
    START_TIME=$2
    END_TIME=$3
    TT=$4
    IFOS=$5   # comma-separated, e.g. H1,L1 or H1,L1,V1

    export MPLCONFIGDIR="/tmp/matplotlib_noah_${{BANK_NUM}}_${{START_TIME}}"
    export ASTROPY_CACHE_DIR="/tmp/astropy_noah_${{BANK_NUM}}_${{START_TIME}}"
    mkdir -p $MPLCONFIGDIR
    mkdir -p $ASTROPY_CACHE_DIR

    declare -A FRAME_CACHES
    {frame_cache_decls}

    IFS=',' read -ra IFO_ARR <<< "$IFOS"

    INSTR_ARGS="" ; CHAN_ARGS="" ; CACHE_ARGS=""
    GPS_START_ARGS="" ; GPS_END_ARGS="" ; SR_ARGS="" ; PAD_ARGS=""
    SEGLEN_ARGS="" ; SEGSTART_ARGS="" ; SEGEND_ARGS=""
    PSD_EST_ARGS="" ; PSD_SEGLEN_ARGS="" ; PSD_STRIDE_ARGS="" ; PSD_INVLEN_ARGS=""
    HP_ARGS="" ; AG_THRESH_ARGS="" ; AG_CLUSTER_ARGS="" ; AG_WIDTH_ARGS="" ; AG_PAD_ARGS="" ; AG_TAPER_ARGS=""

    for IFO in "${{IFO_ARR[@]}}"; do
        ifo_lower=$(echo "$IFO" | tr '[:upper:]' '[:lower:]')
        INSTR_ARGS="$INSTR_ARGS $IFO"
        CHAN_ARGS="$CHAN_ARGS $IFO:GWOSC-4KHZ_R1_STRAIN"
        CACHE_ARGS="$CACHE_ARGS $IFO:${{FRAME_CACHES[$IFO]}}"
        GPS_START_ARGS="$GPS_START_ARGS $IFO:${{START_TIME}} ${{ifo_lower}}:${{START_TIME}}"
        GPS_END_ARGS="$GPS_END_ARGS $IFO:${{END_TIME}} ${{ifo_lower}}:${{END_TIME}}"
        SR_ARGS="$SR_ARGS $IFO:4096 ${{ifo_lower}}:4096"
        PAD_ARGS="$PAD_ARGS $IFO:8 ${{ifo_lower}}:8"
        SEGLEN_ARGS="$SEGLEN_ARGS $IFO:256 ${{ifo_lower}}:256"
        SEGSTART_ARGS="$SEGSTART_ARGS $IFO:8 ${{ifo_lower}}:8"
        SEGEND_ARGS="$SEGEND_ARGS $IFO:8 ${{ifo_lower}}:8"
        PSD_EST_ARGS="$PSD_EST_ARGS $IFO:median ${{ifo_lower}}:median"
        PSD_SEGLEN_ARGS="$PSD_SEGLEN_ARGS $IFO:16 ${{ifo_lower}}:16"
        PSD_STRIDE_ARGS="$PSD_STRIDE_ARGS $IFO:8 ${{ifo_lower}}:8"
        PSD_INVLEN_ARGS="$PSD_INVLEN_ARGS $IFO:16 ${{ifo_lower}}:16"
        HP_ARGS="$HP_ARGS $IFO:20 ${{ifo_lower}}:20"
        AG_THRESH_ARGS="$AG_THRESH_ARGS $IFO:50 ${{ifo_lower}}:50"
        AG_CLUSTER_ARGS="$AG_CLUSTER_ARGS $IFO:0.5 ${{ifo_lower}}:0.5"
        AG_WIDTH_ARGS="$AG_WIDTH_ARGS $IFO:0.25 ${{ifo_lower}}:0.25"
        AG_PAD_ARGS="$AG_PAD_ARGS $IFO:0.25 ${{ifo_lower}}:0.25"
        AG_TAPER_ARGS="$AG_TAPER_ARGS $IFO:0.25 ${{ifo_lower}}:0.25"
    done

    {pycbc_inspiral} \\
        -v \\
        --instruments $INSTR_ARGS \\
        --bank-file {OUT_SPLIT}/split_bank_${{BANK_NUM}}.hdf \\
        --channel-name $CHAN_ARGS \\
        --frame-cache $CACHE_ARGS \\
        --gps-start-time $GPS_START_ARGS \\
        --gps-end-time $GPS_END_ARGS \\
        --ra {KN_ra} \\
        --dec {KN_dec} \\
        --trigger-time ${{TT}} \\
        --low-frequency-cutoff 30.0 \\
        --approximant TaylorF2 \\
        --order 7 \\
        --sample-rate $SR_ARGS \\
        --pad-data $PAD_ARGS \\
        --segment-length $SEGLEN_ARGS \\
        --segment-start-pad $SEGSTART_ARGS \\
        --segment-end-pad $SEGEND_ARGS \\
        --psd-estimation $PSD_EST_ARGS \\
        --psd-segment-length $PSD_SEGLEN_ARGS \\
        --psd-segment-stride $PSD_STRIDE_ARGS \\
        --psd-inverse-length $PSD_INVLEN_ARGS \\
        --strain-high-pass $HP_ARGS \\
        --autogating-threshold $AG_THRESH_ARGS \\
        --autogating-cluster $AG_CLUSTER_ARGS \\
        --autogating-width $AG_WIDTH_ARGS \\
        --autogating-pad $AG_PAD_ARGS \\
        --autogating-taper $AG_TAPER_ARGS \\
        --coinc-threshold 5.5 \\
        --sngl-snr-threshold 4.0 \\
        --chisq-bins 16 \\
        --cluster-method window \\
        --cluster-window 1.0 \\
        --output {OUT_DIR}/{SUFFIX}triggers_bank${{BANK_NUM}}_${{START_TIME}}-${{END_TIME}}.hdf
    """

    with open(sh_filename, 'w') as f:
        f.write(sh_content)

    # 2. Content for the HTCondor submit file
    ldg_tag_line = f"accounting_group = {args.ldg_tag}" if args.ldg_tag else ""
    import math
    with open(WINDOW_FILE, 'r') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    num_chunks = max(1, math.ceil(total_lines / args.chunk_size))
    
    for i in range(num_chunks):
        chunk_lines = lines[i * args.chunk_size : (i + 1) * args.chunk_size]
        chunk_file = WINDOW_FILE.replace('.txt', f'_chunk_{i}.txt')
        
        with open(chunk_file, 'w') as f:
            f.writelines(chunk_lines)
            
        chunk_sub = os.path.join(CONDOR_FILES, f'split_search_chunk_{i}.sub')
        sub_content = f"""executable = {sh_filename}
universe   = vanilla
arguments  = "$(bank) $(start) $(end) $(tt) $(ifos)"
output     = {LOG_DIR}/{SUFFIX}_search_$(bank)_$(start).out
error      = {LOG_DIR}/{SUFFIX}_search_$(bank)_$(start).err
log        = {LOG_DIR}/{SUFFIX}_search_cluster.log
request_cpus   = 1
request_memory = 4GB
request_disk   = 1MB
{ldg_tag_line}
# --- PROTECTION ANTI-HANG ---
periodic_remove = (JobStatus == 2) && (CurrentTime - EnteredCurrentStatus > 14400)

queue bank, start, end, tt, ifos from {chunk_file}
"""
        with open(chunk_sub, 'w') as f:
            f.write(sub_content.strip() + "\n")
            
    print(f"Generated {num_chunks} chunk files of {args.chunk_size} jobs max for parallel submission.")

    # Automatically make the bash script executable (equivalent to running 'chmod +x')
    st = os.stat(sh_filename)
    os.chmod(sh_filename, st.st_mode | stat.S_IEXEC)

    print(f"Successfully generated chunk files.")
    print("Search preparation complete!")
    return 0

if __name__ == '__main__':
    import re
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())