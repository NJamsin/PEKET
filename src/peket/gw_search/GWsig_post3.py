#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import glob
import re
import numpy as np
import h5py
import yaml
import argparse
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import concurrent.futures

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

        # memory safety check -> avoid plotting millions of points
        max_plot_points = 100000
        if len(sorted_bg) > max_plot_points:
            stride = len(sorted_bg) // max_plot_points
            sorted_bg_plot = sorted_bg[::stride]
            far_bg_yr_plot = far_bg_yr[::stride]
        else:
            sorted_bg_plot = sorted_bg
            far_bg_yr_plot = far_bg_yr

        # plot the distribution of background FAR vs SNR
        ax.semilogy(sorted_bg_plot, far_bg_yr_plot, color='dimgray', linewidth=2, alpha=0.8)
        bg_patch = mpatches.Patch(color='gray', alpha=0.7, label='Empirical Background')
        # stylizing the plot
        ax.fill_between(sorted_bg_plot, far_bg_yr_plot, 1e-5, color='silver', alpha=0.3)

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

def plot_fitted_far_vs_snr(bg_stats, top_stat, T_bg, plot_path, T_onsource, fit_steps=100, exclude_top_steps=5):
    """ Plot with exponential fit + 5 sig threshold """
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_bg = bg_stats[bg_stats > 0]
    
    # Compute 5 sig threshold
    p_5sigma = 2.8665e-7
    # FAR = -ln(1 - p) / T_onsource
    far_5sigma_Hz = -np.log(1 - p_5sigma) / T_onsource
    far_5sigma_yr = far_5sigma_Hz * 3.156e7

    top_far_yr_extrapolated = None
    top_fap_extrapolated = None

    if len(valid_bg) > 0 and T_bg > 0:
        sorted_bg = np.sort(valid_bg)
        cum_counts = np.arange(len(sorted_bg), 0, -1)
        far_bg_yr = (cum_counts / T_bg) * 3.156e7

        max_plot_points = 100000
        if len(sorted_bg) > max_plot_points:
            stride = len(sorted_bg) // max_plot_points
            sorted_bg_plot = sorted_bg[::stride]
            far_bg_yr_plot = far_bg_yr[::stride]
        else:
            sorted_bg_plot = sorted_bg
            far_bg_yr_plot = far_bg_yr

        # empirical bg
        ax.semilogy(sorted_bg_plot, far_bg_yr_plot, color='dimgray', linewidth=2, alpha=0.8, label='Empirical Background')
        ax.fill_between(sorted_bg_plot, far_bg_yr_plot, 1e-20, color='silver', alpha=0.3)

        # EXP FIT
        x_rounded = np.round(sorted_bg, decimals=2) # Round to 2 decimals to group close SNRs together
        unique_snrs, unique_indices = np.unique(x_rounded, return_index=True)
        y_unique = far_bg_yr[unique_indices]
        sort_idx = np.argsort(unique_snrs)
        x_clean = unique_snrs[sort_idx]
        y_clean = y_unique[sort_idx]
        sort_idx = np.argsort(unique_snrs)

        if len(x_clean) > exclude_top_steps + 10:
            start_idx = max(0, len(x_clean) - exclude_top_steps - fit_steps)
            end_idx = len(x_clean) - exclude_top_steps

            x_data = x_clean[start_idx : end_idx]
            y_data = y_clean[start_idx : end_idx]
        else:
            print("/!\ Not enough background points to perform a robust fit.")
            x_data = x_clean
            y_data = y_clean

        if len(x_data) > 2:
            # y = a * exp(b * x) => ln(y) = ln(a) + b*x
            coefficients = np.polyfit(x_data, np.log(y_data), 1)
            b_fit = coefficients[0]
            a_fit = np.exp(coefficients[1])

            # Extrapolate up to the 5sig thresh
            x_max = max(top_stat * 1.1, np.max(sorted_bg))
            x_fit = np.linspace(x_data[len(x_data)//2], x_max, 1000)
            y_fit = a_fit * np.exp(b_fit * x_fit)

            ax.plot(x_fit, y_fit, label='Fitted Tail (Exponential)', color='red', linestyle='-')

            # Compute extrapolated FAR/FAP
            top_far_yr_extrapolated = a_fit * np.exp(b_fit * top_stat)
            top_far_Hz = top_far_yr_extrapolated / 3.156e7
            top_fap_extrapolated = 1.0 - np.exp(-top_far_Hz * T_onsource)

    # Trace 5 sig line
    ax.axhline(far_5sigma_yr, color='blue', linestyle='--', label=f'5 $\sigma$ threshold ({far_5sigma_yr:.2e} /yr)')
    
    # plot best trigger candidate
    if top_far_yr_extrapolated is not None:
        ax.axvline([top_stat], color='indigo', linestyle='-.', zorder=5, label=f'Top Trigger')
        ax.scatter([top_stat], [top_far_yr_extrapolated], color='indigo', s=60, zorder=6)

    ax.set_xlabel('Ranking Statistic (reweighted SNR)', fontsize=12)
    ax.set_ylabel('False Alarm Rate (1/yr)', fontsize=12)
    ax.set_title(f'Extrapolated Background FAR vs Ranking Statistic', fontsize=14)
    
    if len(valid_bg) > 0 and T_bg > 0: # dyn ylim 
        y_min = min(far_5sigma_yr / 10, top_far_yr_extrapolated / 10 if top_far_yr_extrapolated else 1e-5)
        ax.set_ylim(bottom=y_min, top=max(far_bg_yr) * 5)

    ax.grid(True, which="major", ls="-", alpha=0.5)
    ax.grid(True, which="minor", ls=":", alpha=0.3)
    
    leg = ax.legend(loc='upper right', numpoints=5)
    for line in leg.get_lines():
        line.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Fitted FAR vs SNR plot saved to: {plot_path}")
    
    return top_far_yr_extrapolated, top_fap_extrapolated

def slide_limiter(n_ifos, slide_shift, segment_length):
    """Same formula as in GWsig_prep.py / pycbc_multi_inspiral's internal bound.
    Needed here because each background file now bundles many shortslides
    instead of exactly one: T_bg per analyzed window must be multiplied by
    this number, not counted once per file."""
    if n_ifos == 1:
        return 1
    stride_dur = segment_length / 2
    return 1 + int(stride_dur / (slide_shift * (n_ifos - 1)))

def read_top_trigger_stat(base_dir, suffix):
    cand_file = os.path.join(base_dir, 'out', f'{suffix}_top_candidates.txt')
    if not os.path.exists(cand_file): return None, None
    with open(cand_file, 'r') as f: first_line = f.readline().strip()
    stat_val, time_val = None, None
    for part in first_line.split('|'):
        if 'Rank Stat' in part: stat_val = float(part.split(':')[1].strip())
        if 'Time' in part: time_val = float(part.split(':')[1].strip())
    return stat_val, time_val

def process_single_file(fpath):
    """ Worker function to process one single file and extract its SNRs + Segment Info """
    snrs = []
    segment = None
    
    # parse filename -- now tagged with the long-slide offset too:
    # {suffix}_bg_bank{N}_{start}-{end}_ls{offset}.hdf (offset can be negative)
    # One file still bundles every *short*slide produced internally by
    # --do-shortslides for that (bank, window, long-slide offset).
    basename = os.path.basename(fpath)
    match = re.search(r'_bg_bank\d+_(\d+)-(\d+)_ls(-?\d+)\.hdf', basename)
    if match:
        segment = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
    # read triggers -- unchanged: the 'network' group already concatenates the
    # triggers from every shortslide, tagged individually via 'slide_id'
    try:
        with h5py.File(fpath, 'r') as hf:
            if 'network' in hf:
                snr_data = hf['network'].get('reweighted_snr', hf['network'].get('coherent_snr', []))
                if len(snr_data) > 0:
                    snrs = snr_data[:]
    except Exception:
        pass # Ignore corrupted files silently
        
    return snrs, segment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config")
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"])
    parser.add_argument("--fit-steps", type=int, default=100, help="Number of points to use for the exponential fit (starting from the loudest)")
    parser.add_argument("--exclude-top-steps", type=int, default=5, help="Number of loudest points to exclude from the exponential fit (to avoid glitches)")
    parser.add_argument("--segment-length", default=512, type=int,
                         help="Must match the --segment-length used in GWsig_prep.py / the .sh job.")
    parser.add_argument("--slide-shift", default=0.501, type=float,
                         help="Must match the --slide-shift used in GWsig_prep.py / the .sh job.")
    parser.add_argument("--n-ifos", default=2, type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    BASE_DIR = os.path.abspath(config['Directory']['BASE_DIR'])
    SUFFIX = config['Directory']['run_name']
    SIG_DIR = os.path.join(BASE_DIR, 'significance')
    BG_OUT_DIR = os.path.join(SIG_DIR, 'out')
    PLOT_DIR = os.path.join(BASE_DIR, 'plots')
    for dir in [BASE_DIR, SIG_DIR, BG_OUT_DIR, PLOT_DIR]:
        os.makedirs(dir, exist_ok=True)

    top_stat, top_time = read_top_trigger_stat(BASE_DIR, SUFFIX)
    if top_stat is None:
        print("Error reading the top trigger statistic.")
        return 1

    # Reconstruct WIN_DUR from the EM posterior samples 
    EM_samp = pd.read_csv(config['KN_data']['EM_post_file'], delimiter=' ', dtype=np.float32)
    KN_t0 = Time(config['KN_data']['first_detection'], format='isot', scale='utc').mjd
    if args.OSW_sigma == "full": p16, p84 = EM_samp['timeshift'].min(), EM_samp['timeshift'].max() 
    elif args.OSW_sigma == "1": p16, _, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])
    elif args.OSW_sigma == "2": p16, _, p84 = np.percentile(EM_samp['timeshift'], [2.275, 50, 97.725])
    elif args.OSW_sigma == "3": p16, _, p84 = np.percentile(EM_samp['timeshift'], [0.135, 50, 99.865])
    time_gps = Time((KN_t0 + p16, KN_t0 + p84), format='mjd').gps
    WIN_DUR = int(time_gps[1]) - int(time_gps[0])

    print(f"Locating background triggers in {BG_OUT_DIR}...")
    file_list = glob.glob(os.path.join(BG_OUT_DIR, '*/*.hdf'))
    total_files = len(file_list)
    print(f"Found {total_files} files to process.")

    # Multi-thread reading
    all_snrs = []
    analyzed_segments = set()
    completed = 0

    print(f"Starting parallel data extraction with 32 threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_file, f): f for f in file_list}
        
        for future in concurrent.futures.as_completed(futures):
            snrs, segment = future.result()
            
            if len(snrs) > 0:
                all_snrs.append(snrs)
            if segment is not None:
                analyzed_segments.add(segment)
                
            completed += 1
            if completed % 10000 == 0:
                sys.stdout.write(f"\r  -> Processed {completed}/{total_files} files...")
                sys.stdout.flush()
                
    print(f"\r  -> Processed {completed}/{total_files} files. Done!\n")
    bg_stats = np.concatenate(all_snrs) if all_snrs else np.array([])

    # Each (start,end) window now yields num_slides background draws internally
    # (--do-shortslides), instead of 1 draw per file as before -> T_bg must be
    # scaled accordingly, or it will be underestimated by a factor of num_slides
    # and the FAR will come out artificially loud (this was the ~8 SNR plateau).
    #
    # analyzed_segments now holds (start, end, longslide_offset) triples rather
    # than just (start, end): several template banks sharing the same window
    # AND the same long-slide offset analyze the same underlying noise, so they
    # must be deduped (counted once) -- but different long-slide offsets of the
    # *same* (start, end) window genuinely decorrelate H1 from L1 and produce
    # independent background, so each distinct offset adds its own T_bg instead
    # of being deduped away.
    num_slides = slide_limiter(args.n_ifos, args.slide_shift, args.segment_length)
    print(f"Each analyzed window contributes {num_slides} background shortslides "
          f"(--slide-shift {args.slide_shift}, --segment-length {args.segment_length}), "
          f"across {len(analyzed_segments)} distinct (window, long-slide) combinations.")

    T_bg = 0
    for (start_t, end_t, _longslide) in analyzed_segments:
        T_bg += max(0, (end_t - start_t) - 16) * num_slides # -16 for padding
        
    n_louder = int(np.sum(bg_stats >= top_stat))
    far = (1 + n_louder) / T_bg if T_bg > 0 else np.inf
    p_value = 1.0 - np.exp(-far * WIN_DUR)
    far_yr = far * 3.156e7
    prefix = "< " if n_louder == 0 else ""

    plot_far_path = os.path.join(PLOT_DIR, f'{SUFFIX}_far_vs_snr.png')
    print("Generating Raw FAR vs SNR plot...")
    plot_far_vs_snr(bg_stats, top_stat, T_bg, plot_far_path, WIN_DUR)

    plot_fitted_path = os.path.join(PLOT_DIR, f'{SUFFIX}_fitted_far_vs_snr.png')
    print("Generating Fitted FAR vs SNR plot...")
    far_yr_extrapolated, p_extrapolated = plot_fitted_far_vs_snr(bg_stats, top_stat, T_bg, plot_fitted_path, WIN_DUR, args.fit_steps, args.exclude_top_steps)

    print(f"{'─'*50}")
    print(f"  Top trigger stat     : {top_stat:.4f}")
    print(f"  Louder than top      : {n_louder}")
    print(f"  T_background         : {T_bg:.1f} s  ({T_bg/3.156e7:.3f} yr)")
    print(f"  FAR (Empirical)      : {prefix}{far:.3e} Hz  ({prefix}{far_yr:.3f} /yr)")
    print(f"  p-value (Empirical)  : {prefix}{p_value:.3e}\n")
    print(f"FAR (Extrapolated)      : {far_yr_extrapolated/3.156e7:.6e} Hz\n")
    print(f"FAR (Extrapolated)      : {far_yr_extrapolated:.6e} /yr\n")
    print(f"p-value (Extrapolated)  : {p_extrapolated:.6e}")
    print(f"{'─'*50}\n")

    sig_out = os.path.join(BASE_DIR, 'out', f'{SUFFIX}_significance.txt')
    with open(sig_out, 'w') as f:
        f.write(f"Top stat                : {top_stat:.6f}\n")
        f.write(f"N louder                : {n_louder}\n")
        f.write(f"T background            : {T_bg:.2f} s\n")
        f.write(f"FAR (Empirical)         : {prefix}{far:.6e} Hz\n")
        f.write(f"FAR (Empirical)         : {prefix}{far_yr:.6e} /yr\n")
        f.write(f"p-value (Empirical)     : {prefix}{p_value:.6e}\n")
        f.write(f"Bounding limit          : {str(n_louder == 0)}\n")
        if far_yr_extrapolated is not None:
            f.write(f"FAR (Extrapolated)      : {far_yr_extrapolated/3.156e7:.6e} Hz\n")
            f.write(f"FAR (Extrapolated)      : {far_yr_extrapolated:.6e} /yr\n")
            f.write(f"p-value (Extrapolated)  : {p_extrapolated:.6e}\n")

    print("Significance estimation complete.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
