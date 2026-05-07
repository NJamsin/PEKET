#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GWsearch_significance.py
Estimates FAR and p-value for the top trigger from gw-setup-pipeline.
Uses the Time Slides method on explicitly defined OFF-SOURCE windows.
Reads the .lcf cache files from GWsearch_prep.py and the top trigger stat produced by GWsearch_post.py.
Registered as entry point: gw-search-significance
"""
import sys
import os
import stat
import glob
import subprocess
import time
import re
from collections import defaultdict
import numpy as np
import h5py
import yaml
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
import urllib.request
from peket.gw_search.GWsearch_prep import robust_get_urls, preparer_donnees

from nmma.nmma.post_processing import parser

def plot_timeline(top_time, on_window, n_slides, slide_win_1, slide_win_2, plot_path):
    """
    Generate a timeline plot showing the on-source window, off-source windows, and time slides relative to the top trigger time.
     - top_time: GPS time of the top trigger (float)
     - on_window: tuple (start, end) of the on-source window (GPS times)
     - n-slides: number of time slides generated (1s per slide)
     - slide_win_1: tuple (start, end) of the time slide window for the first off-source (GPS times)
     - slide_win_2: tuple (start, end) of the time slide window for the second off-source (GPS times)
     - plot_path: path to save the generated plot
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Visuallization Y positions for the bars
    y_data = 1
    y_slide = 2
    y_on = 3

    # Utility function
    def rel(t): return t - top_time
    def width(win): return win[1] - win[0]

    # 1. Downloaded Off-Source Data 
    ax.barh(y_data, width(slide_win_1)*n_slides, left=rel(slide_win_1[0]- 2*width(slide_win_1)), height=0.4, 
            color='mediumseagreen', alpha=0.7, label='Background Duration (Timeslides)')
    ax.barh(y_data, width(slide_win_2)*n_slides, left=rel(slide_win_2[0]), height=0.4, 
            color='mediumseagreen', alpha=0.7)

    # 2. Timeslide Windows (Bleu)
    ax.barh(y_slide, width(slide_win_1), left=rel(slide_win_1[0]), height=0.4, 
            color='royalblue', alpha=0.8, label='Off-source Windows')
    ax.barh(y_slide, width(slide_win_2), left=rel(slide_win_2[0]), height=0.4, 
            color='royalblue', alpha=0.8)

    # 3. On-source window (red)
    ax.barh(y_on, width(on_window), left=rel(on_window[0]), height=0.4, 
            color='crimson', alpha=0.9, label='On-Source Window')

    # top candidate
    ax.axvline(0, color='black', linestyle='--', lw=1.5, label=f'Top Trigger (t=0)')

    ax.set_yticks([y_data, y_slide, y_on])
    ax.set_yticklabels(['Timeslides', 'Off-source', 'On-Source'])
    ax.set_xlabel(f'Time [s] relative to top candidate ({top_time:.1f})')
    #ax.set_title('Analysis Windows Timeline', y=1.22)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Timeline plot saved to: {plot_path}")

def count_completed_jobs_from_log(log_path, cluster_id):
    """Count finished Condor job via the log file"""
    if not os.path.exists(log_path):
        return 0
    completed = 0
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip().startswith('005 ('):
                try:
                    event_cluster = int(line.strip().split('(')[1].split('.')[0])
                    if event_cluster == cluster_id:
                        completed += 1
                except (IndexError, ValueError):
                    continue
    return completed

def read_top_trigger_stat(base_dir, suffix):
    """Read the top_candidates.txt file and return the rank_stat of the top trigger."""
    cand_file = os.path.join(base_dir, 'out', f'{suffix}_top_candidates.txt')
    if not os.path.exists(cand_file):
        raise FileNotFoundError(
            f"Top candidates file not found: {cand_file}\n"
            "Make sure GWsearch_post.py ran successfully before this script."
        )
    with open(cand_file, 'r') as f:
        first_line = f.readline().strip()
    
    stat_val, time_val = None, None
    for part in first_line.split('|'):
        if 'Rank Stat' in part:
            stat_val = float(part.split(':')[1].strip())
        if 'Time' in part:
            time_val = float(part.split(':')[1].strip())
    return stat_val, time_val

def collect_background_stats(bg_dir):
    """
    Collects ALL background triggers from off-source windows.
    Returns both the SNRs and their associated times.
    """
    all_snrs = []
    all_times = []
    search_path = os.path.join(bg_dir, '*/*.hdf')
    for fpath in sorted(glob.glob(search_path)):
        try:
            with h5py.File(fpath, 'r') as hf:
                if 'network' not in hf: continue
                
                # get snr
                if 'reweighted_snr' in hf['network']: 
                    snrs = hf['network']['reweighted_snr'][:]
                elif 'coherent_snr' in hf['network']: 
                    snrs = hf['network']['coherent_snr'][:]
                else: 
                    continue
                
                # get times
                if 'end_time' in hf['network']: 
                    times = hf['network']['end_time'][:]
                elif 'time' in hf['network']: 
                    times = hf['network']['time'][:]
                else: 
                    times = np.zeros_like(snrs)

                # append
                if len(snrs) > 0:
                    all_snrs.extend(snrs)
                    all_times.extend(times)
                    
        except Exception as e:
            print(f"  Warning: could not read {fpath}: {e}")
            
    return np.array(all_snrs), np.array(all_times)

def compute_far_pvalue(top_stat, bg_stats, T_bg, T_onsource):
    """
    FAR = (1 + N_louder) / T_background   [Hz = events/s]
    p-value = 1 - exp(-FAR * T_onsource)  [Poisson]
    """
    n_louder = int(np.sum(bg_stats >= top_stat))
    far = (1 + n_louder) / T_bg if T_bg > 0 else np.inf
    p_value = 1.0 - np.exp(-far * T_onsource)
    return far, p_value, n_louder

def generate_timeslides_file(on_source_start, on_source_end, max_size, n_slides, sig_window_file, num_banks, overlap, negative_slide=False, delay=0):
    """Generate the parameter file with 2 off-source windows per slide."""
    with open(sig_window_file, 'w') as f:
        for slide in range(1, n_slides + 1):
            for bank in range(num_banks):
                current_start = on_source_start
                while current_start < on_source_end:
                    current_end = min(current_start + max_size, on_source_end)
                    tt = (current_start + current_end) // 2 #for the antenna pattern
                    # Write: BANK_NUM START_TIME END_TIME
                    if negative_slide:
                        f.write(f"{-(slide + delay)} {bank} {current_start} {current_end} {tt}\n")
                    else:
                        f.write(f"{slide + delay} {bank} {current_start} {current_end} {tt}\n")
                    if current_end == on_source_end:
                        break
                
                    # Step back by the overlap amount for the next chunk
                    current_start = current_end - overlap
    print("Generated time slides file:", sig_window_file, f"with {n_slides} slides × {num_banks} banks in chunks of {max_size}s with {overlap}s overlap. Total jobs: {len(open(sig_window_file).readlines())}")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plot_far_vs_snr(bg_stats, top_stat, T_bg, plot_path, T_onsource):
    """
    Generate a plot of False Alarm Rate (FAR) vs Ranking Statistic (SNR) for the background triggers,
    and indicate the position of the top trigger with its corresponding FAR.
     - bg_stats: array of SNRs from background triggers
     - top_stat: SNR of the top trigger
     - T_bg: Total background time analyzed (in seconds)
     - plot_path: path to save the generated plot
     - T_onsource: Total time of the on-source analysis (in seconds)
     Note: The FAR is computed as (1 + N_louder) / T_bg, where N_louder is the number of background triggers with SNR >= top_stat. The plot will show the distribution of background FAR as a function of SNR, and the top trigger will be highlighted with its FAR. If the top trigger is louder than all background triggers, it will be shown as a point with an upper limit on the FAR (e.g., "< 1/T_bg").
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # positive mask to avoid plotting invalid (negative) SNRs from the background
    valid_bg = bg_stats[bg_stats > 0]

    if len(valid_bg) > 0 and T_bg > 0:
        # 1. rising order 
        sorted_bg = np.sort(valid_bg)
        
        # 2. Cumulative counts of background triggers louder than each SNR threshold
        # Exemple : for the lowest SNR in the background, all triggers are louder (cum_counts = N), for the highest SNR, only 1 trigger is louder (cum_counts = 1)
        cum_counts = np.arange(len(sorted_bg), 0, -1)
        
        # 3. Far computation
        far_bg_yr = (cum_counts / T_bg) * 3.156e7

        # plot the distribution of background FAR vs SNR
        ax.semilogy(sorted_bg, far_bg_yr, color='dimgray', linewidth=2, alpha=0.8)
        bg_patch = mpatches.Patch(color='gray', alpha=0.7, label='Empirical Background')
        # stylizing the plot
        ax.fill_between(sorted_bg, far_bg_yr, 1e-5, color='silver', alpha=0.3)

    # top trigger FAR computation
    n_louder = np.sum(valid_bg >= top_stat)
    
    top_far = (1 + n_louder) / T_bg if T_bg > 0 else np.inf
    top_far_yr = top_far * 3.156e7
    top_fap = 1.0 - np.exp(-top_far * T_onsource) if top_far < np.inf else 1.0

    # If louder than all background triggers, we show it as an upper limit (e.g., "< 1/T_bg") on the plot
    is_limit = (n_louder == 0)
    prefix = "< " if is_limit else ""

    # best candidate point
    ax.scatter([top_stat], [top_far_yr], color='red', s=60, zorder=5, 
               label=f'Top Trigger (FAR {prefix}{top_far_yr:.2e} /yr)')
    trigger_handle = mlines.Line2D([], [], color='red', marker='o', linestyle='None', 
                               markersize=8, label='Top Trigger')
    stats_handle = mlines.Line2D([], [], color='none', marker='None', linestyle='None', 
                             label=f'FAR {prefix}{top_far_yr:.2e} /yr\nFAP: {top_fap:.2e}')

    # "upper limit" arrow if the top trigger is louder than all background triggers
    if is_limit:
        ax.annotate('', xy=(top_stat, top_far_yr * 0.5), xytext=(top_stat, top_far_yr),
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.5), zorder=5)

    # Plot styling
    ax.set_xlabel('Ranking Statistic (reweighted SNR)', fontsize=12)
    ax.set_ylabel('False Alarm Rate (1/yr)', fontsize=12)
    ax.set_title(f'Background FAR vs Ranking Statistic', fontsize=14)
    
    if len(valid_bg) > 0 and T_bg > 0:
        ax.set_ylim(bottom=max(1e-5, 0.1 / (T_bg / 3.156e7)), top=max(far_bg_yr) * 2)

    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    ax.legend(handles=[bg_patch, trigger_handle, stats_handle], loc='upper right', numpoints=1)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  FAR vs SNR plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Estimate FAR and p-value for the top trigger using Time Slides.")
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument("--n-slides", default=300, type=int, help="Number of time slides to generate.")
    parser.add_argument("--run-background", action="store_true", help="Generate AND submit the HTCondor jobs.")
    parser.add_argument("--submit", action="store_true", help="Auto-submit background jobs to HTCondor.")
    parser.add_argument("--monitor", action="store_true", help="Monitor the background jobs.")
    parser.add_argument("--window", default='both', choices=['both', 'before', 'after'], help="Which off-source window(s) to use for background estimation.")
    parser.add_argument("--minimal-log", default=None, action="store_true", help="Reduce logging output to essentials only.")
    parser.add_argument("--max-timeslides", default=4096, type=int, help="Maximum number of slides, corresponds to the additional data duration to be downloaded/prepared (in seconds). Default is 4096s, if window is both, it will be 4096s for each window (total 8192s).")
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"], help="Size of the time window to be searched around the expected trigger time, in sigmas. Default is 1.")
    parser.add_argument("--ldg-tag", default=None, help="The \"accounting_group\" tag required for submitting to the LDG cluster. If not specified, the pipeline will be generated without the tag.")
    parser.add_argument("--delay", default=0, type=int, help="Delay in seconds/timeslides between the limits of the OSW and the start/end of the off-source windows. Useful to break down the background estimation into multiple runs, eg, 200 timeslides delay 0 and timeslides 200 delay 200 to have 400 timeslides in total but split into 2 runs of 200 slides (to avoid too many condor jobs at once).")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    BASE_DIR   = os.path.abspath(config['Directory']['BASE_DIR'])
    SUFFIX     = config['Directory']['run_name']
    NUM_SPLITS = config['GW_search']['num_splits']
    KN_EM_post = config['KN_data']['EM_post_file']
    KN_date    = config['KN_data']['first_detection']
    KN_ra      = config['KN_data']['ra']
    KN_dec     = config['KN_data']['dec']

    bin_dir        = os.path.dirname(sys.executable)
    pycbc_inspiral = os.path.join(bin_dir, "pycbc_multi_inspiral")

    # Recompute on-source window
    EM_samp    = pd.read_csv(KN_EM_post, delimiter=' ', dtype=np.float32)
    KN_t0      = Time(KN_date, format='isot', scale='utc').mjd
    if args.OSW_sigma == "full":
        p16, p84 = EM_samp['timeshift'].min(), EM_samp['timeshift'].max() 
    elif args.OSW_sigma == "1":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])
    elif args.OSW_sigma == "2":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [2.275, 50, 97.725])
    elif args.OSW_sigma == "3":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [0.135, 50, 99.865])
    time_gps   = Time((KN_t0 + p16, KN_t0 + p84), format='mjd').gps
    
    print("On-source window (GPS):", int(time_gps[0]), "-", int(time_gps[1]))

    ON_START   = int(time_gps[0])
    ON_END     = int(time_gps[1])
    WIN_DUR    = ON_END - ON_START

    # Define off-source windows size
    OFF_DUR = WIN_DUR
    OFF1_START = ON_START - OFF_DUR -16 
    OFF1_END   = ON_START -16
    OFF2_START = ON_END +16
    OFF2_END   = ON_END + OFF_DUR + 16 
    add_dur = args.max_timeslides
    if add_dur < args.n_slides:
        print(f"Warning: max_timeslides ({add_dur}s) is less than n_slides ({args.n_slides}). This may lead to insufficient background data for the number of slides requested. Consider increasing max_timeslides or reducing n_slides.")
    DATA_OFF1_START = OFF1_START - add_dur
    DATA_OFF1_END   = OFF1_END
    DATA_OFF2_START = OFF2_START 
    DATA_OFF2_END   = OFF2_END + add_dur
    print(f"Off-source windows (GPS):\n  Window 1: {OFF1_START+add_dur} - {OFF1_END}\n  Window 2: {OFF2_START} - {OFF2_END}")

    BG_SUFFIX = f"{SUFFIX}_background"
    DATA_DIR    = os.path.join(BASE_DIR, 'data')
    DATA_DIR    = os.path.join(DATA_DIR, "background")
    SIG_DIR     = os.path.join(BASE_DIR, 'significance')
    BG_OUT_DIR  = os.path.join(SIG_DIR, 'out')
    PLOT_DIR    = os.path.join(BASE_DIR, 'plots')
    SUB_DIR     = os.path.join(BASE_DIR, 'sub_files')
    LOG_DIR     = os.path.join(BASE_DIR, 'logs')
    os.makedirs(SIG_DIR, exist_ok=True)
    os.makedirs(BG_OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    OUT_SPLIT = os.path.join(BASE_DIR, f'{SUFFIX}_split')
    SIG_WINDOW_FILE_1 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_1.txt')
    SIG_WINDOW_FILE_2 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_2.txt')
    bg_h1_lcf = os.path.join(DATA_DIR, f'{BG_SUFFIX}_H1.lcf')
    bg_l1_lcf = os.path.join(DATA_DIR, f'{BG_SUFFIX}_L1.lcf')

    if not (os.path.exists(bg_h1_lcf) and os.path.exists(bg_l1_lcf)):
        args.injection = False
        args.detector_threshold = 0.0
        print(f"\nDownloading and preparing data for Off-Source Background estimation...")
        detectors = ['H1', 'L1']
        downloaded_files = {'H1': [], 'L1': []}
        
        for ifo in detectors:
            print(f"Locating 4kHz data for {ifo}...")
            urls = robust_get_urls(ifo, DATA_OFF1_START - 16, DATA_OFF1_END + 16)
            print("Downloading files for first off-source window...")
            for url in urls:
                filename = url.split('/')[-1]
                filepath = os.path.join(DATA_DIR, filename)                
                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                else:
                    print(f"{filename} already exists locally. Skipping.")
                downloaded_files[ifo].append(filepath)
            urls = robust_get_urls(ifo, DATA_OFF2_START - 16, DATA_OFF2_END + 16)
            print("Downloading files for second off-source window...")
            for url in urls:
                filename = url.split('/')[-1]
                filepath = os.path.join(DATA_DIR, filename)                
                if not os.path.exists(filepath):
                    print(f"Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                else:
                    print(f"{filename} already exists locally. Skipping.")
                downloaded_files[ifo].append(filepath)
        print("All required data files downloaded successfully.")
        print("Preparing the data for background estimation...")
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['H1'], "H1:GWOSC-4KHZ_R1_STRAIN", "H1", 
            DATA_OFF1_START - 16, DATA_OFF1_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['L1'], "L1:GWOSC-4KHZ_R1_STRAIN", "L1", 
            DATA_OFF1_START - 16, DATA_OFF1_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['L1'], "L1:GWOSC-4KHZ_R1_STRAIN", "L1", 
            DATA_OFF2_START - 16, DATA_OFF2_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['H1'], "H1:GWOSC-4KHZ_R1_STRAIN", "H1", 
            DATA_OFF2_START - 16, DATA_OFF2_END + 16)
        
        if 1 > 2: # just to avoid deletion during test bruh
            print("Cleaning up downloaded raw data files...")
            for ifo in detectors:
                for filepath in downloaded_files[ifo]:
                    os.remove(filepath)
                    print(f"Deleted {filepath}")
                
        print("Extended background data prepared successfully.")
        print("Time covered by background data:")
        print(f"  - Window 1: {DATA_OFF1_START} to {DATA_OFF1_END}")
        print(f"  - Window 2: {DATA_OFF2_START} to {DATA_OFF2_END}")
    else:
        print(f"\nBackground data files already exist for both detectors. Skipping download and preparation.")

    print(f"\n{'='*55}")
    print(f"  PEKET - Significance estimation (Time Slides)")
    print(f"{'='*55}\n")
    
    top_stat, top_time = read_top_trigger_stat(BASE_DIR, SUFFIX)
    print(f"Top trigger ranking stat : {top_stat:.4f} at epoch {top_time:.1f} (GPS)")

    print(f"\nGenerating {args.n_slides} time slides across 2 off-source windows (size {OFF_DUR}s each)...")
    generate_timeslides_file(OFF1_START, OFF1_END, 1000, args.n_slides, SIG_WINDOW_FILE_1, NUM_SPLITS, overlap=16, negative_slide=True) # 16 for 8s padding start/end in pycbc + negative slide for the first window
    generate_timeslides_file(OFF2_START, OFF2_END, 1000, args.n_slides, SIG_WINDOW_FILE_2, NUM_SPLITS, overlap=16, negative_slide=False)

    # merge the two window files into one for easier monitoring and submission (optional, can keep separate if preferred)
    if args.window == 'both':
        merged_window_file = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_all.txt')
        with open(merged_window_file, 'w') as fout:
            for f in [SIG_WINDOW_FILE_1, SIG_WINDOW_FILE_2]:
                with open(f, 'r') as fin:
                    fout.write(fin.read())
        print(f"  Merged time slides file created: {merged_window_file}")

    '''
    timeline_plot_path = os.path.join(PLOT_DIR, f'{SUFFIX}_timeline.png')
    print("\nGenerating timeline plot...")
    on_window = (ON_START, ON_END)
    off_data_1 = (DATA_OFF1_START, DATA_OFF1_END)
    off_data_2 = (DATA_OFF2_START, DATA_OFF2_END)

    slide_win_1 = (OFF1_START, OFF1_END)
    slide_win_2 = (OFF2_START, OFF2_END)
    plot_timeline(top_time, on_window, args.n_slides, slide_win_1, slide_win_2, timeline_plot_path)
    '''
    #print("DEBUG/ early stopping after generating time slides file.")
    #return 0
    if args.run_background:
        # delete the content of the output directory to avoid mixing with previous runs
        import shutil
        if os.path.exists(BG_OUT_DIR):
            shutil.rmtree(BG_OUT_DIR)
        os.makedirs(BG_OUT_DIR, exist_ok=True)
        # Threshold Optimizations
        s_snr_thresh = 4.5
        coinc_threshold = 4 # low limit to get more tirgger for plotting
        
        sh_bg = os.path.join(SUB_DIR, 'run_significance_search.sh')
        if args.window == 'before':
            window_file = SIG_WINDOW_FILE_1
            sub_bg = os.path.join(SUB_DIR, 'significance_search1.sub')
        elif args.window == 'after':
            window_file = SIG_WINDOW_FILE_2
            sub_bg = os.path.join(SUB_DIR, 'significance_search2.sub')
        else:
            window_file = merged_window_file
            sub_bg = os.path.join(SUB_DIR, 'significance_search_both.sub')
        total_bg_jobs = len(open(window_file).readlines())

        sh_content = f"""#!/bin/bash
export PATH="{sys.prefix}/bin:$PATH"

SLIDE=$1
BANK_NUM=$2
START_TIME=$3
END_TIME=$4
TT=$5

mkdir -p {BG_OUT_DIR}/bank_${{BANK_NUM}}

{pycbc_inspiral} \\
    -v \\
    --instruments H1 L1 \\
    --bank-file {OUT_SPLIT}/split_bank_${{BANK_NUM}}.hdf \\
    --channel-name H1:GWOSC-4KHZ_R1_STRAIN L1:GWOSC-4KHZ_R1_STRAIN \\
    --frame-cache H1:{bg_h1_lcf} L1:{bg_l1_lcf} \\
    --gps-start-time H1:${{START_TIME}} L1:${{START_TIME}} h1:${{START_TIME}} l1:${{START_TIME}} \\
    --gps-end-time H1:${{END_TIME}} L1:${{END_TIME}} h1:${{END_TIME}} l1:${{END_TIME}} \\
    --ra {KN_ra} \\
    --dec {KN_dec} \\
    --trigger-time ${{TT}} \\
    --low-frequency-cutoff 30.0 \\
    --approximant TaylorF2 \\
    --order 7 \\
    --sample-rate H1:4096 L1:4096 h1:4096 l1:4096 \\
    --pad-data H1:8 L1:8 h1:8 l1:8 \\
    --segment-length H1:256 L1:256 h1:256 l1:256 \\
    --segment-start-pad H1:8 L1:8 h1:8 l1:8 \\
    --segment-end-pad H1:8 L1:8 h1:8 l1:8 \\
    --psd-estimation H1:median L1:median h1:median l1:median \\
    --psd-segment-length H1:16 L1:16 h1:16 l1:16 \\
    --psd-segment-stride H1:8 L1:8 h1:8 l1:8 \\
    --psd-inverse-length H1:16 L1:16 h1:16 l1:16 \\
    --strain-high-pass H1:20 L1:20 h1:20 l1:20 \\
    --autogating-threshold H1:50 L1:50 h1:50 l1:50 \\
    --autogating-cluster H1:0.5 L1:0.5 h1:0.5 l1:0.5 \\
    --autogating-width H1:0.25 L1:0.25 h1:0.25 l1:0.25 \\
    --autogating-pad H1:0.25 L1:0.25 h1:0.25 l1:0.25 \\
    --autogating-taper H1:0.25 L1:0.25 h1:0.25 l1:0.25 \\
    --sngl-snr-threshold {s_snr_thresh} \\
    --coinc-threshold {coinc_threshold} \\
    --chisq-bins 16 \\
    --cluster-method window \\
    --cluster-window 1.0 \\
    --slide-shift ${{SLIDE}} \\
    --output {BG_OUT_DIR}/bank_${{BANK_NUM}}/{SUFFIX}_bg_bank${{BANK_NUM}}_${{START_TIME}}-${{END_TIME}}_slide${{SLIDE}}.hdf
"""
        with open(sh_bg, 'w') as f:
            f.write(sh_content.strip() + "\n")
        os.chmod(sh_bg, os.stat(sh_bg).st_mode | stat.S_IEXEC)
        log_out = f"{LOG_DIR}/{SUFFIX}_sig_cluster.log"
        
        if getattr(args, 'minimal_log', False):
            output = "/dev/null"
            error  = "/dev/null"
        else:
            output = f"{LOG_DIR}/{SUFFIX}_sig_$(bank)_slide$(slide)_$(window).out"
            error  = f"{LOG_DIR}/{SUFFIX}_sig_$(bank)_slide$(slide)_$(window).err"
        sub_content = f"""executable = {sh_bg}
universe   = vanilla
arguments  = "$(slide) $(bank) $(start) $(end) $(tt)"
output     = {output}
error      = {error}
log        = {log_out}
request_cpus   = 1
request_memory = 4GB
request_disk   = 1MB
queue slide, bank, start, end, tt from {window_file}
"""
        with open(sub_bg, 'w') as f:
            f.write(sub_content.strip() + "\n")
        print(f"  HTCondor files generated: {sh_bg}\n\t\t\t\t{sub_bg}")
        
        if args.submit:
            import subprocess
            try:
                result = subprocess.run(["condor_submit", sub_bg], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(result.stdout, end='')
                if result.stderr: print(result.stderr, end='')
            except subprocess.CalledProcessError as e:
                print("\n" + "!"*65)
                print(" Critical error from condor_submit")
                print("!"*65)
                print(f" Sub file : {sub_bg}")
                print(f"\n --- Condor error --- \n{e.stderr}")
                print("!"*60 + "\n")
                return 1 

            cluster_id = None
            for line in result.stdout.splitlines():
                if "submitted to cluster" in line:
                    cluster_id = int(line.strip().split("cluster")[-1].strip().rstrip('.'))
                    break
            if cluster_id is None:
                raise RuntimeError("Could not extract cluster ID from condor_submit output.")
            
            if args.monitor:
                sig_log = os.path.join(LOG_DIR, f'{SUFFIX}_sig_cluster.log')
                print("-" * 50)
                print("Monitoring background jobs (This will take time due to coherent search)...")
                print("-" * 50)
                completed = 0
                while completed < total_bg_jobs:
                    completed = count_completed_jobs_from_log(sig_log, cluster_id)   
                    percent = int((completed / total_bg_jobs) * 100)
                    bar = '█' * (percent // 5) + '-' * (20 - (percent // 5))
                    sys.stdout.write(f"\r[{bar}] {completed}/{total_bg_jobs} background jobs done ({percent}%)")
                    sys.stdout.flush()
                    if completed < total_bg_jobs: time.sleep(10)
                print("\n  All background jobs completed.")
            else:
                print("  Jobs submitted. Re-run without --run-background once complete.")
                return 0
        else:
            return 0
    
    # ── Calcul FAR / p-value ───────────────────
    print(f"\nCollecting background triggers from {BG_OUT_DIR}...")
    bg_stats, _ = collect_background_stats(BG_OUT_DIR)
    
    if len(bg_stats) == 0:
        print("  No valid background triggers found. Did the Condor jobs finish successfully?")
        return 1

    # compute T_bg by summing the durations of all unique analyzed slide segments
    T_bg = 0
    analyzed_segments = set()
    
    search_path = os.path.join(BG_OUT_DIR, '*/*.hdf')
    for fpath in sorted(glob.glob(search_path)):
        try:
            # outfile name: {SUFFIX}_bg_bank{BANK_NUM}_{START_TIME}-{END_TIME}_slide{SLIDE}.hdf
            basename = os.path.basename(fpath)
            # more robust approach in case of '_' in SUFFIX
            match = re.search(r'_bg_bank\d+_(\d+)-(\d+)_slide(\d+)\.hdf', basename)
            if match:
                start_time = int(match.group(1))
                end_time = int(match.group(2))
                slide = int(match.group(3))
                
                # ignore the bank number (count each one time only !)
                analyzed_segments.add((slide, start_time, end_time))
        except Exception as e:
            print(f"  Warning: could not parse duration from {fpath}: {e}")
            
    # remove padding
    for (slide, start_t, end_t) in analyzed_segments:
        T_bg += max(0, (end_t - start_t) - 16)
    
    far, p_value, n_louder = compute_far_pvalue(top_stat, bg_stats, T_bg, WIN_DUR)
    far_yr = far * 3.156e7

    prefix = "< " if n_louder == 0 else ""

    print(f"\n{'─'*50}")
    print(f"  Top trigger stat     : {top_stat:.4f}")
    print(f"  Louder than top      : {n_louder}")
    print(f"  T_background         : {T_bg:.1f} s  ({T_bg/3.156e7:.3f} yr)")
    print(f"  FAR                  : {prefix}{far:.3e} Hz  ({prefix}{far_yr:.3f} /yr)")
    print(f"  p-value (on-source)  : {prefix}{p_value:.3e}")
    print(f"{'─'*50}\n")

    sig_out = os.path.join(BASE_DIR, 'out', f'{SUFFIX}_significance.txt')
    with open(sig_out, 'w') as f:
        f.write(f"Top stat       : {top_stat:.6f}\n")
        f.write(f"N louder       : {n_louder}\n")
        f.write(f"T background   : {T_bg:.2f} s\n")
        f.write(f"FAR            : {prefix}{far:.6e} Hz\n")
        f.write(f"FAR            : {prefix}{far_yr:.6e} /yr\n")
        f.write(f"p-value        : {prefix}{p_value:.6e}\n")
        f.write(f"Bounding limit : {str(n_louder == 0)}\n")

    plot_far_path = os.path.join(PLOT_DIR, f'{SUFFIX}_far_vs_snr.png')
    print("\nGenerating FAR vs SNR plot...")
    plot_far_vs_snr(bg_stats, top_stat, T_bg, plot_far_path, WIN_DUR)

    print("Significance estimation complete.")
    return 0

if __name__ == '__main__':
    import re
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())