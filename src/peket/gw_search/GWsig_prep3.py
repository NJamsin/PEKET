#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import stat
import math
import urllib.request
import yaml
import argparse
import pandas as pd
import numpy as np
from astropy.time import Time
from peket.gw_search.GWsearch_prep import robust_get_urls, preparer_donnees

# Edge padding eaten on each side of an analysis segment by
# --segment-start-pad / --segment-end-pad (8s + 8s). Also the overlap used
# between consecutive off-source windows so there is no coverage gap.
EDGE_PAD = 16

# One-sided Gaussian tail probabilities (False Alarm Probability targets).
SIGMA_TO_FAP = {
    1: 0.158655,
    2: 0.0227501,
    3: 0.00134990,
    4: 3.16712e-5,
    5: 2.86652e-7,
}


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


def slide_limiter(n_ifos, slide_shift, segment_length):
    """
    Mirror of the bound pycbc_multi_inspiral applies internally when
    --do-shortslides is set (see slide_limiter() in bin/pycbc_multi_inspiral
    on gwastro/pycbc -- checked against the upstream source: identical
    formula and bound).
    Used here only to (a) sanity-check the (slide_shift, segment_length)
    combo BEFORE submitting thousands of jobs, and (b) know how many
    background draws a single job will actually produce, since
    GWsig_post.py needs that same number to compute T_bg correctly.
    """
    if n_ifos == 1:
        print("Warning: --n-ifos=1, can't perform timeslides on a single detector. Setting num_slides=1.")
        return 1
    stride_dur = segment_length / 2
    num_slides = 1 + int(stride_dur / (slide_shift * (n_ifos - 1)))
    low, upp = 1, segment_length
    if not (low <= num_slides <= upp):
        raise ValueError(
            f"(--slide-shift {slide_shift}, --segment-length {segment_length}) "
            f"gives num_slides={num_slides}, outside the allowed range [{low}, {upp}]. "
            "Increase --segment-length (raises the cap) or --slide-shift (raises the step)."
        )
    return num_slides


def windows_durations(start, end, max_size, overlap):
    """
    Same chunking logic as generate_windows_file, but only returns the raw
    duration of each window instead of writing anything to disk. Used to
    estimate how much off-source time will actually end up analyzed, which
    --target-sigma needs to know how much background it can expect.
    """
    durs = []
    current = start
    while current < end:
        c_end = min(current + max_size, end)
        durs.append(c_end - current)
        if c_end == end:
            break
        current = c_end - overlap
    return durs


def resolve_target_sigma(target_sigma, t_onsource, t_analyzed_total, n_ifos,
                          window_max_size, safety_factor, longslide_step):
    """
    Pick (slide_shift, segment_length, num_longslides) so the total
    background time T_bg accumulated across every analyzed off-source
    window is at least `safety_factor` times bigger than the bare 1/FAR
    a `target_sigma`-sigma claim would need. The safety factor matters
    because a louder T_bg than the bare minimum is what lets the
    empirical/extrapolated FAR curve actually resolve that confidence
    level instead of just bounding it from below.

    Strategy, in order:
      1. Try to get there with --do-shortslides alone, by growing
         segment_length (num_slides scales ~linearly with it once
         slide_shift sits at its physical floor).
      2. segment_length is capped at window_max_size - EDGE_PAD, so a
         single off-source window/job can still hold the segment --
         this is exactly the constraint that blows up to "segment-length
         of several thousand seconds" if you try to solve for it without
         the cap.
      3. Whatever T_bg is still missing once segment_length is maxed out
         is made up with a modest number of extra integer-second "long
         slides" between H1/L1 -- one extra HTCondor job per offset per
         window, not one job per shortslide like the old architecture.
    """
    p_target = SIGMA_TO_FAP[target_sigma]
    far_target = -math.log(1 - p_target) / t_onsource    # Hz
    t_bg_target = safety_factor / far_target              # seconds

    segment_length_max = window_max_size - EDGE_PAD
    segment_length_max -= segment_length_max % 2  # keep it even

    # Lowest slide_shift that keeps num_slides inside pycbc's own bound for
    # any segment_length -- going lower trips the assertion inside
    # pycbc_multi_inspiral itself, regardless of segment_length.
    slide_shift = 0.501 / (n_ifos - 1) if n_ifos > 1 else 0.501

    num_slides_max = slide_limiter(n_ifos, slide_shift, segment_length_max)
    t_bg_per_longslide_max = t_analyzed_total * num_slides_max

    if t_bg_per_longslide_max >= t_bg_target:
        # Shortslides alone are enough: don't max out segment_length,
        # just grow it to the minimum that reaches the target.
        num_slides_needed = max(1, math.ceil(t_bg_target / t_analyzed_total))
        segment_length = int(math.ceil(2 * slide_shift * (n_ifos - 1) * (num_slides_needed - 1)))
        segment_length = max(64, min(segment_length_max, segment_length))
        # int()-truncation inside slide_limiter can undershoot this continuous
        # estimate by one stride -- nudge up until the real (integer) bound
        # actually reaches num_slides_needed, or we hit the window ceiling.
        while (segment_length < segment_length_max and
               slide_limiter(n_ifos, slide_shift, segment_length) < num_slides_needed):
            segment_length += 1
        num_longslides = 1
    else:
        segment_length = segment_length_max
        num_longslides = math.ceil(t_bg_target / t_bg_per_longslide_max)
        if num_longslides > 100:
            print(f"  /!\\ Reaching {target_sigma} sigma needs ~{num_longslides} long slides "
                  f"-> {(num_longslides - 1) * longslide_step} s of extra background data to "
                  "download as long-slide headroom. Consider lowering --sigma-safety-factor, "
                  "or accepting an extrapolated (fitted) significance rather than a fully "
                  "empirical one.")

    num_slides_final = slide_limiter(n_ifos, slide_shift, segment_length)
    t_bg_estimate = t_analyzed_total * num_slides_final * num_longslides

    return segment_length, slide_shift, num_longslides, num_slides_final, t_bg_estimate, t_bg_target


def generate_windows_file(on_source_start, on_source_end, max_size, sig_window_file,
                           num_banks, overlap, longslide_offsets):
    """
    longslide_offsets: list of integer-second offsets (always including 0)
    applied to L1 relative to H1 for that window. Each (bank, window) now
    expands into one line per offset, so a single job's worth of
    --do-shortslides background gets multiplied by len(longslide_offsets)
    independent long-slide background draws.
    """
    with open(sig_window_file, 'w') as f:
        for bank in range(num_banks):
            current_start = on_source_start
            while current_start < on_source_end:
                current_end = min(current_start + max_size, on_source_end)
                tt = (current_start + current_end) // 2

                for ls in longslide_offsets:
                    f.write(f"{bank} {current_start} {current_end} {tt} {ls}\n")

                if current_end == on_source_end:
                    break
                current_start = current_end - overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config")
    parser.add_argument("--window", default='both', choices=['both', 'before', 'after'])
    parser.add_argument("--max-extension", default=4096, type=int,
                         help="Maximum duration to extend the off-source window itself (in seconds) -- "
                              "this directly grows the analyzed background duration.")
    parser.add_argument("--OSW-sigma", default='1', choices=['1', '2', '3', "full"])
    parser.add_argument("--ldg-tag", default=None, help="Tag accounting_group pour IGWN")
    parser.add_argument("--chunk-size", default=3000, type=int)
    parser.add_argument("--segment-length", default=512, type=int,
                         help="pycbc_multi_inspiral --segment-length (s). Also caps the max "
                              "number of shortslides obtainable from a single segment. "
                              "Ignored/overwritten if --target-sigma is set.")
    parser.add_argument("--slide-shift", default=0.501, type=float,
                         help="pycbc_multi_inspiral --slide-shift (s). "
                              "Ignored/overwritten if --target-sigma is set.")
    parser.add_argument("--n-ifos", default=2, type=int,
                         help="Number of detectors used (for the shortslide bound check).")
    parser.add_argument("--target-sigma", default=None, choices=[1, 2, 3, 4, 5], type=int,
                         help="If set, auto-pick --slide-shift/--segment-length (and, if that's not "
                              "enough, a number of long slides) to accumulate enough background to "
                              "reach a X sigma FAP. Leave unset to control slide-shift/segment-length "
                              "manually.")
    parser.add_argument("--sigma-safety-factor", default=1.5, type=float,
                         help="With --target-sigma, how many times more background than the bare "
                              "1/FAR minimum to aim for, so the FAR curve actually resolves that "
                              "confidence level instead of just bounding it.")
    parser.add_argument("--num-longslides", default=1, type=int,
                         help="Number of integer-second long-slide offsets per window (including the "
                              "zero-offset baseline), applied between H1 and L1, on top of the internal "
                              "--do-shortslides. 1 = no long slides. Overwritten if --target-sigma needs "
                              "more.")
    parser.add_argument("--longslide-step", default=2, type=int,
                         help="Spacing in integer seconds between consecutive long-slide offsets.")
    parser.add_argument("--longslide-margin", default=0, type=int,
                         help="Extra data (s) to download beyond --max-extension, purely as long-slide "
                              "headroom. Auto-raised if more long slides than this allows are needed. "
                              "NOTE: changing this after background data has already been downloaded "
                              "has no effect -- delete the background data directory to force a "
                              "re-download with the new margin.")
    parser.add_argument("--window-max-size", default=1000, type=int,
                         help="Max duration (s) of a single off-source analysis window/job; also the "
                              "ceiling --target-sigma will respect for --segment-length.") #should read it from the .yaml
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
    DATA_OFF1_START, DATA_OFF1_END = OFF1_START - args.max_extension, OFF1_END
    DATA_OFF2_START, DATA_OFF2_END = OFF2_START, OFF2_END + args.max_extension

    # ---- RESOLVE SLIDE-SHIFT / SEGMENT-LENGTH / LONG-SLIDES ----
    # This has to happen before the data download below, because the
    # number of long slides decides how much extra background data
    # (longslide_margin) needs to be fetched.
    t_onsource = ON_END - ON_START
    durs1 = windows_durations(DATA_OFF1_START, DATA_OFF1_END, args.window_max_size, EDGE_PAD)
    durs2 = windows_durations(DATA_OFF2_START, DATA_OFF2_END, args.window_max_size, EDGE_PAD)
    t_analyzed_before = sum(d - EDGE_PAD for d in durs1)
    t_analyzed_after = sum(d - EDGE_PAD for d in durs2)
    if args.window == 'both':
        t_analyzed_total = t_analyzed_before + t_analyzed_after
    elif args.window == 'before':
        t_analyzed_total = t_analyzed_before
    else:
        t_analyzed_total = t_analyzed_after

    if args.target_sigma is not None:
        (args.segment_length, args.slide_shift, args.num_longslides, num_slides,
         t_bg_estimate, t_bg_target) = resolve_target_sigma(
            args.target_sigma, t_onsource, t_analyzed_total, args.n_ifos,
            args.window_max_size, args.sigma_safety_factor, args.longslide_step,
        )
        print(f"\n--target-sigma {args.target_sigma} requested (safety factor "
              f"{args.sigma_safety_factor}x):")
        print(f"  -> --segment-length {args.segment_length}  --slide-shift {args.slide_shift:.4f}  "
              f"--num-longslides {args.num_longslides}")
        print(f"  -> estimated T_bg ~ {t_bg_estimate:.1f} s vs target {t_bg_target:.1f} s "
              f"({t_bg_estimate / 3.156e7:.3f} yr)")
    else:
        num_slides = slide_limiter(args.n_ifos, args.slide_shift, args.segment_length)

    required_longslide_margin = (args.num_longslides - 1) * args.longslide_step
    args.longslide_margin = max(args.longslide_margin, required_longslide_margin)

    # Long-slide offsets: shift L1 relative to H1, always moving AWAY from
    # the on-source window (earlier for the "before" off-source window,
    # later for the "after" one) so the shift stays inside the freshly
    # downloaded margin and never creeps back toward the on-source time.
    offsets_before = [-k * args.longslide_step for k in range(args.num_longslides)]
    offsets_after = [k * args.longslide_step for k in range(args.num_longslides)]

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

    # Data download bounds: extend further out by longslide_margin, on the
    # same (away-from-on-source) side as the long-slide offsets above, so
    # every long-slide variant has real frame data to read.
    DL_OFF1_START = DATA_OFF1_START - (args.longslide_margin if args.window in ('both', 'before') else 0)
    DL_OFF2_END = DATA_OFF2_END + (args.longslide_margin if args.window in ('both', 'after') else 0)

    # ---- 1. DATA PREPARATION ----
    if not (os.path.exists(bg_h1_lcf) and os.path.exists(bg_l1_lcf)):
        args.injection = False
        args.detector_threshold = 0.0
        print(f"\nDownloading and preparing data for Off-Source Background estimation...")
        detectors = ['H1', 'L1']
        downloaded_files = {'H1': [], 'L1': []}
        for ifo in detectors:
            print(f"Locating 4kHz data for {ifo}...")
            urls = robust_get_urls(ifo, DL_OFF1_START - 16, DATA_OFF1_END + 16)
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
            urls = robust_get_urls(ifo, DATA_OFF2_START - 16, DL_OFF2_END + 16)
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
            DL_OFF1_START - 16, DATA_OFF1_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['L1'], "L1:GWOSC-4KHZ_R1_STRAIN", "L1",
            DL_OFF1_START - 16, DATA_OFF1_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['L1'], "L1:GWOSC-4KHZ_R1_STRAIN", "L1",
            DATA_OFF2_START - 16, DL_OFF2_END + 16)
        preparer_donnees(
            args, config, DATA_DIR, BG_SUFFIX, BASE_DIR, downloaded_files['H1'], "H1:GWOSC-4KHZ_R1_STRAIN", "H1",
            DATA_OFF2_START - 16, DL_OFF2_END + 16)

        print("Cleaning up downloaded raw data files...")
        for ifo in detectors:
            for filepath in downloaded_files[ifo]:
                if os.path.exists(filepath):
                    os.remove(filepath)

        print("Extended background data prepared successfully.")
        print("Time covered by background data:")
        print(f"  - Window 1: {DL_OFF1_START} to {DATA_OFF1_END}")
        print(f"  - Window 2: {DATA_OFF2_START} to {DL_OFF2_END}")
    else:
        print(f"\nBackground data files already exist for both detectors. Skipping download and preparation.")
        print("(If you changed --target-sigma / --num-longslides / --longslide-margin / "
              "--max-extension since the last run, delete the background data directory "
              "and re-run, or the new long slides may read past the available frame data.)")

    print(f"\n{'='*55}")
    print(f"  PEKET - Significance estimation (Short + Long Time Slides)")
    print(f"{'='*55}\n")

    top_stat, top_time = read_top_trigger_stat(BASE_DIR, SUFFIX)
    print(f"Top trigger ranking stat : {top_stat:.4f} at epoch {top_time:.1f} (GPS)")
    print(f"With --segment-length {args.segment_length} and --slide-shift {args.slide_shift}, "
          f"each segment of each job will yield {num_slides} background shortslides, "
          f"x{args.num_longslides} long-slide offset(s) per window.")

    # ---- 2. GENERATION DES FENETRES (UNE LIGNE PAR (BANK, WINDOW, LONGSLIDE)) ----
    SIG_WINDOW_FILE_1 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_1.txt')
    SIG_WINDOW_FILE_2 = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_2.txt')

    generate_windows_file(DATA_OFF1_START, DATA_OFF1_END, args.window_max_size, SIG_WINDOW_FILE_1,
                           NUM_SPLITS, EDGE_PAD, offsets_before)
    generate_windows_file(DATA_OFF2_START, DATA_OFF2_END, args.window_max_size, SIG_WINDOW_FILE_2,
                           NUM_SPLITS, EDGE_PAD, offsets_after)

    merged_window_file = os.path.join(SIG_DIR, f'{SUFFIX}_sig_windows_all.txt')
    with open(merged_window_file, 'w') as fout:
        for f in [SIG_WINDOW_FILE_1, SIG_WINDOW_FILE_2]:
            with open(f, 'r') as fin: fout.write(fin.read())

    window_file = merged_window_file if args.window == 'both' else (SIG_WINDOW_FILE_1 if args.window == 'before' else SIG_WINDOW_FILE_2)

    # ---- 3. GENERATING .SH & .SUB ----
    sh_bg = os.path.join(SUB_DIR, 'sig_search.sh')

    sh_content = f"""#!/bin/bash
export PATH="{sys.prefix}/bin:$PATH"
BANK_NUM=$1
START_TIME=$2
END_TIME=$3
TT=$4
LONGSLIDE=$5

L1_START=$((START_TIME + LONGSLIDE))
L1_END=$((END_TIME + LONGSLIDE))

export MPLCONFIGDIR="/tmp/matplotlib_noah_${{BANK_NUM}}_${{START_TIME}}_${{LONGSLIDE}}"
export ASTROPY_CACHE_DIR="/tmp/astropy_noah_${{BANK_NUM}}_${{START_TIME}}_${{LONGSLIDE}}"
mkdir -p $MPLCONFIGDIR
mkdir -p $ASTROPY_CACHE_DIR

mkdir -p {BG_OUT_DIR}/bank_${{BANK_NUM}}

{pycbc_inspiral} \\
    -v \\
    --instruments H1 L1 \\
    --bank-file {OUT_SPLIT}/split_bank_${{BANK_NUM}}.hdf \\
    --channel-name H1:GWOSC-4KHZ_R1_STRAIN L1:GWOSC-4KHZ_R1_STRAIN \\
    --frame-cache H1:{bg_h1_lcf} L1:{bg_l1_lcf} \\
    --gps-start-time H1:${{START_TIME}} L1:${{L1_START}} h1:${{START_TIME}} l1:${{L1_START}}\\
    --gps-end-time H1:${{END_TIME}} L1:${{L1_END}} h1:${{END_TIME}} l1:${{L1_END}} \\
    --ra {KN_ra} \\
    --dec {KN_dec} \\
    --trigger-time ${{TT}} \\
    --low-frequency-cutoff 30.0 \\
    --approximant TaylorF2 \\
    --order 7 \\
    --sample-rate H1:4096 L1:4096 h1:4096 l1:4096 \\
    --pad-data H1:8 L1:8 h1:8 l1:8 \\
    --segment-length H1:{args.segment_length} L1:{args.segment_length} h1:{args.segment_length} l1:{args.segment_length} \\
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
    --do-shortslides \\
    --slide-shift {args.slide_shift} \\
    --output {BG_OUT_DIR}/bank_${{BANK_NUM}}/{SUFFIX}_bg_bank${{BANK_NUM}}_${{START_TIME}}-${{END_TIME}}_ls${{LONGSLIDE}}.hdf
"""
    with open(sh_bg, 'w') as f: f.write(sh_content.strip() + "\n")
    os.chmod(sh_bg, os.stat(sh_bg).st_mode | stat.S_IEXEC)

    # add ldg tag line only if ldg_tag is provided
    ldg_tag_line = f"accounting_group = {args.ldg_tag}" if args.ldg_tag else ""

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
arguments  = "$(bank) $(start) $(end) $(tt) $(ls)"
output     = {LOG_DIR}/{SUFFIX}_sig_$(bank)_$(start)_$(ls).out
error      = {LOG_DIR}/{SUFFIX}_sig_$(bank)_$(start)_$(ls).err
log        = {LOG_DIR}/{SUFFIX}_sig_cluster.log
request_cpus   = 1
request_memory = 8GB
request_disk   = 1MB
{ldg_tag_line}
periodic_remove = (JobStatus == 2) && (CurrentTime - EnteredCurrentStatus > 14400)

queue bank, start, end, tt, ls from {chunk_file}
"""
        with open(chunk_sub, 'w') as f:
            f.write(sub_content.strip() + "\n")

    print(f"Generated {num_chunks} chunk files of {args.chunk_size} jobs max for parallel submission.")
    print("Significance preparation complete!")

if __name__ == '__main__':
    sys.exit(main())
