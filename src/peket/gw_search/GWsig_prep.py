#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import stat
import urllib.request
import yaml
import argparse
import pandas as pd
import numpy as np
from astropy.time import Time
from peket.gw_search.GWsearch_prep import robust_get_urls, preparer_donnees

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

def generate_timeslides_file(on_source_start, on_source_end, max_size, n_slides, sig_window_file, num_banks, overlap, negative_slide=False, delay=0):
    with open(sig_window_file, 'w') as f:
        for slide in range(1, n_slides + 1):
            for bank in range(num_banks):
                current_start = on_source_start
                while current_start < on_source_end:
                    current_end = min(current_start + max_size, on_source_end)
                    tt = (current_start + current_end) // 2 
                    if negative_slide:
                        f.write(f"{-(slide + delay)} {bank} {current_start} {current_end} {tt}\n")
                    else:
                        f.write(f"{slide + delay} {bank} {current_start} {current_end} {tt}\n")
                    if current_end == on_source_end:
                        break
                    current_start = current_end - overlap
    print(f"Generated time slides file: {sig_window_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config")
    parser.add_argument("--n-slides", default=300, type=int)
    parser.add_argument("--window", default='both', choices=['both', 'before', 'after'])
    parser.add_argument("--max-timeslides", default=4096, type=int)
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"])
    parser.add_argument("--delay", default=0, type=int)
    parser.add_argument("--ldg-tag", default=None, help="Tag accounting_group pour IGWN")
    parser.add_argument("--chunk-size", default=3000, type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    BASE_DIR = os.path.abspath(config['Directory']['BASE_DIR'])
    SUFFIX = config['Directory']['run_name']
    NUM_SPLITS = config['GW_search']['num_splits']
    KN_EM_post = config['KN_data']['EM_post_file']
    KN_date = config['KN_data']['first_detection']
    KN_ra = config['KN_data']['ra']
    KN_dec = config['KN_data']['dec']

    bin_dir = os.path.dirname(sys.executable)
    pycbc_inspiral = os.path.join(bin_dir, "pycbc_multi_inspiral")

    # OSW calculation
    EM_samp = pd.read_csv(KN_EM_post, delimiter=' ', dtype=np.float32)
    KN_t0 = Time(KN_date, format='isot', scale='utc').mjd
    if args.OSW_sigma == "full":
        p16, p84 = EM_samp['timeshift'].min(), EM_samp['timeshift'].max() 
    elif args.OSW_sigma == "1":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])
    elif args.OSW_sigma == "2":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [2.275, 50, 97.725])
    elif args.OSW_sigma == "3":
        p16, _, p84 = np.percentile(EM_samp['timeshift'], [0.135, 50, 99.865])
    time_gps = Time((KN_t0 + p16, KN_t0 + p84), format='mjd').gps

    ON_START, ON_END = int(time_gps[0]), int(time_gps[1])
    OFF_DUR = ON_END - ON_START

    OFF1_START, OFF1_END = ON_START - OFF_DUR - 16, ON_START - 16
    OFF2_START, OFF2_END = ON_END + 16, ON_END + OFF_DUR + 16 
    DATA_OFF1_START, DATA_OFF1_END = OFF1_START - args.max_timeslides, OFF1_END
    DATA_OFF2_START, DATA_OFF2_END = OFF2_START, OFF2_END + args.max_timeslides

    # Directories
    BG_SUFFIX = f"{SUFFIX}_background"
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'background')
    SIG_DIR = os.path.join(BASE_DIR, 'significance')
    BG_OUT_DIR = os.path.join(SIG_DIR, 'out')
    SUB_DIR = os.path.join(BASE_DIR, 'sub_files')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    for d in [DATA_DIR, SIG_DIR, BG_OUT_DIR, SUB_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    OUT_SPLIT = os.path.join(BASE_DIR, f'{SUFFIX}_split')
    bg_h1_lcf = os.path.join(DATA_DIR, f'{BG_SUFFIX}_H1.lcf')
    bg_l1_lcf = os.path.join(DATA_DIR, f'{BG_SUFFIX}_L1.lcf')

    # ---- 1. DATA PREPARATION ----
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
        
        print("Cleaning up downloaded raw data files...")
        for ifo in detectors:
            for filepath in downloaded_files[ifo]:
                if os.path.exists(filepath):
                    os.remove(filepath)
                
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
    
    # ---- 2. GENERATION DES FENETRES (TIMESLIDES) ----
    SIG_WINDOW_FILE_1 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_1.txt')
    SIG_WINDOW_FILE_2 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_2.txt')
    
    generate_timeslides_file(OFF1_START, OFF1_END, 1000, args.n_slides, SIG_WINDOW_FILE_1, NUM_SPLITS, overlap=16, negative_slide=True, delay=args.delay)
    generate_timeslides_file(OFF2_START, OFF2_END, 1000, args.n_slides, SIG_WINDOW_FILE_2, NUM_SPLITS, overlap=16, negative_slide=False, delay=args.delay)

    merged_window_file = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_all.txt')
    with open(merged_window_file, 'w') as fout:
        for f in [SIG_WINDOW_FILE_1, SIG_WINDOW_FILE_2]:
            with open(f, 'r') as fin: fout.write(fin.read())

    window_file = merged_window_file if args.window == 'both' else (SIG_WINDOW_FILE_1 if args.window == 'before' else SIG_WINDOW_FILE_2)

    # ---- 3. GENERATING .SH & .SUB ----
    sh_bg = os.path.join(SUB_DIR, 'sig_search.sh')
    sub_bg = os.path.join(SUB_DIR, 'sig_search.sub')

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
    --sngl-snr-threshold 4.5 \\
    --coinc-threshold 4 \\
    --chisq-bins 16 \\
    --cluster-method window \\
    --cluster-window 1.0 \\
    --slide-shift ${{SLIDE}} \\
    --output {BG_OUT_DIR}/bank_${{BANK_NUM}}/{SUFFIX}_bg_bank${{BANK_NUM}}_${{START_TIME}}-${{END_TIME}}_slide${{SLIDE}}.hdf
"""
    with open(sh_bg, 'w') as f: f.write(sh_content.strip() + "\n")
    os.chmod(sh_bg, os.stat(sh_bg).st_mode | stat.S_IEXEC)

    # add ldg tag line only if ldg_tag is provided
    ldg_tag_line = f"accounting_group = {args.ldg_tag}" if args.ldg_tag else ""
    
    import math
    with open(window_file, 'r') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    num_chunks = max(1, math.ceil(total_lines / args.chunk_size))
    
    for i in range(num_chunks):
        chunk_lines = lines[i * args.chunk_size : (i + 1) * args.chunk_size]
        chunk_file = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_chunk_{i}.txt')
        with open(chunk_file, 'w') as f:
            f.writelines(chunk_lines)
            
        chunk_sub = os.path.join(SUB_DIR, f'sig_search_chunk_{i}.sub')
        sub_content = f"""executable = {sh_bg}
universe   = vanilla
arguments  = "$(slide) $(bank) $(start) $(end) $(tt)"
output     = {LOG_DIR}/{SUFFIX}_sig_$(bank)_slide$(slide).out
error      = {LOG_DIR}/{SUFFIX}_sig_$(bank)_slide$(slide).err
log        = {LOG_DIR}/{SUFFIX}_sig_cluster.log
request_cpus   = 1
request_memory = 4GB
request_disk   = 1MB
{ldg_tag_line}
periodic_remove = (JobStatus == 2) && (CurrentTime - EnteredCurrentStatus > 14400)

queue slide, bank, start, end, tt from {chunk_file}
"""
        with open(chunk_sub, 'w') as f:
            f.write(sub_content.strip() + "\n")
            
    print(f"Generated {num_chunks} chunk files of {args.chunk_size} jobs max for parallel submission.")
    print("Significance preparation complete!")

if __name__ == '__main__':
    sys.exit(main())