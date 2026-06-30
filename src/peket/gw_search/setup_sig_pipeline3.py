#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import yaml
import subprocess
import time
import glob
import pandas as pd
import numpy as np
from astropy.time import Time
import math

EDGE_PAD = 16

SIGMA_TO_FAP = {
    1: 0.158655,
    2: 0.0227501,
    3: 0.00134990,
    4: 3.16712e-5,
    5: 2.86652e-7,
}


def slide_limiter(n_ifos, slide_shift, segment_length):
    """Mirror of GWsig_prep.py / pycbc_multi_inspiral's internal bound.
    """
    if n_ifos == 1:
        return 1
    stride_dur = segment_length / 2
    num_slides = 1 + int(stride_dur / (slide_shift * (n_ifos - 1)))
    low, upp = 1, segment_length
    if not (low <= num_slides <= upp):
        raise ValueError(
            f"(--slide-shift {slide_shift}, --segment-length {segment_length}) "
            f"gives num_slides={num_slides}, outside the allowed range [{low}, {upp}]."
        )
    return num_slides


def windows_durations(start, end, max_size, overlap):
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
    """Identical logic to GWsig_prep.py's resolve_target_sigma -- see that
    file for the detailed rationale in comments."""
    p_target = SIGMA_TO_FAP[target_sigma]
    far_target = -math.log(1 - p_target) / t_onsource
    t_bg_target = safety_factor / far_target

    segment_length_max = window_max_size - EDGE_PAD
    segment_length_max -= segment_length_max % 2

    slide_shift = 0.501 / (n_ifos - 1) if n_ifos > 1 else 0.501
    num_slides_max = slide_limiter(n_ifos, slide_shift, segment_length_max)
    t_bg_per_longslide_max = t_analyzed_total * num_slides_max

    if t_bg_per_longslide_max >= t_bg_target:
        num_slides_needed = max(1, math.ceil(t_bg_target / t_analyzed_total))
        segment_length = int(math.ceil(2 * slide_shift * (n_ifos - 1) * (num_slides_needed - 1)))
        segment_length = max(64, min(segment_length_max, segment_length))
        while (segment_length < segment_length_max and
               slide_limiter(n_ifos, slide_shift, segment_length) < num_slides_needed):
            segment_length += 1
        num_longslides = 1
    else:
        segment_length = segment_length_max
        num_longslides = math.ceil(t_bg_target / t_bg_per_longslide_max)

    num_slides_final = slide_limiter(n_ifos, slide_shift, segment_length)
    t_bg_estimate = t_analyzed_total * num_slides_final * num_longslides

    return segment_length, slide_shift, num_longslides, num_slides_final, t_bg_estimate, t_bg_target


def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor DAG for PyCBC Significance Estimation.")
    # base args
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--prep-script", default="gw-sig-prep3", help="Name of the preparation script")
    parser.add_argument("--post-script", default="gw-sig-post3", help="Name of the post-processing script")

    # submit monitor args
    parser.add_argument("--submit", action="store_true", help="Automatically submit the pipeline to HTCondor")
    parser.add_argument("--monitor", action="store_true", help="Monitor the pipeline execution")

    # significance estimation args (passed to prep and post)
    parser.add_argument("--window", default='both', choices=['both', 'before', 'after'], help="Which off-source window(s) to use.")
    parser.add_argument("--max-extension", default=4096, type=int, help="Maximum duration to extend to off source window (in seconds).")
    parser.add_argument("--OSW-sigma", default='1', choices=['1', '2', '3', "full"], help="Search window sigma size.")
    parser.add_argument("--ldg-tag", default=None, help="The accounting_group tag required for IGWN.")
    parser.add_argument("--chunk-size", default=3000, type=int, help="Number of jobs per chunk to bypass Schedd limits")
    parser.add_argument("--fit-steps", type=int, default=100, help="Number of points to use for the exponential fit (starting from the loudest)")
    parser.add_argument("--exclude-top-steps", type=int, default=5, help="Number of loudest points to exclude from the exponential fit (to avoid glitches)")
    parser.add_argument("--segment-length", default=512, type=int, help="pycbc_multi_inspiral --segment-length (s).")
    parser.add_argument("--slide-shift", default=0.501, type=float, help="pycbc_multi_inspiral --slide-shift (s).")
    parser.add_argument("--n-ifos", default=2, type=int)
    parser.add_argument("--target-sigma", default=None, choices=[1, 2, 3, 4, 5], type=int,
                         help="Overwrite \"slide-shift\"/\"segment-length\" (and, if needed, add long "
                              "slides) to ensure accumulating enough background to reach a X sigma FAP. "
                              "Leave unset to control slide-shift/segment-length manually.")
    parser.add_argument("--sigma-safety-factor", default=1.5, type=float,
                         help="With --target-sigma, how many times more background than the bare "
                              "1/FAR minimum to aim for.")
    parser.add_argument("--num-longslides", default=1, type=int,
                         help="Number of integer-second long-slide offsets per window (including the "
                              "zero-offset baseline). 1 = no long slides. Overwritten if --target-sigma "
                              "needs more.")
    parser.add_argument("--longslide-step", default=2, type=int,
                         help="Spacing in integer seconds between consecutive long-slide offsets.")
    parser.add_argument("--longslide-margin", default=0, type=int,
                         help="Extra data (s) to download beyond --max-extension as long-slide headroom.")
    parser.add_argument("--window-max-size", default=1000, type=int,
                         help="Max duration (s) of a single off-source analysis window/job.") # need to be taken from .yaml instead

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = os.path.abspath(config['Directory']['BASE_DIR'])
    suffix = config['Directory']['run_name']

    # Create dir
    os.makedirs(base_dir, exist_ok=True)
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    sub_files_dir = os.path.join(base_dir, 'sub_files')
    os.makedirs(sub_files_dir, exist_ok=True)

    bin_dir = os.path.dirname(sys.executable)

    # --- RESOLVE SLIDE-SHIFT / SEGMENT-LENGTH / LONG-SLIDES (mirrors GWsig_prep.py exactly) ---
    try:
        EM_samp = pd.read_csv(config['KN_data']['EM_post_file'], delimiter=' ', dtype=np.float32)
        KN_t0 = Time(config['KN_data']['first_detection'], format='isot', scale='utc').mjd

        if args.OSW_sigma == "full":
            p16, p84 = EM_samp['timeshift'].min(), EM_samp['timeshift'].max()
        elif args.OSW_sigma == "1":
            p16, _, p84 = np.percentile(EM_samp['timeshift'], [15.865, 50, 84.135])
        elif args.OSW_sigma == "2":
            p16, p84 = np.percentile(EM_samp['timeshift'], [2.275, 97.725])
        elif args.OSW_sigma == "3":
            p16, p84 = np.percentile(EM_samp['timeshift'], [0.135, 99.865])

        time_gps = Time((KN_t0 + p16, KN_t0 + p84), format='mjd').gps
        ON_START, ON_END = int(time_gps[0]), int(time_gps[1])
        OFF_DUR = ON_END - ON_START

        OFF1_START, OFF1_END = ON_START - OFF_DUR - 16, ON_START - 16
        OFF2_START, OFF2_END = ON_END + 16, ON_END + OFF_DUR + 16
        # NB: GWsig_prep.py actually generates its windows over the
        # *extended* DATA_OFF1/DATA_OFF2 range (OFF1/OFF2 widened by
        # --max-extension), not the bare OFF1/OFF2 range. The chunking
        # estimate below has to mirror that, or this script's pre-built
        # DAG ends up with the wrong number of SIG_SEARCH nodes relative
        # to the chunk files GWsig_prep.py will actually write at runtime.
        DATA_OFF1_START, DATA_OFF1_END = OFF1_START - args.max_extension, OFF1_END
        DATA_OFF2_START, DATA_OFF2_END = OFF2_START, OFF2_END + args.max_extension

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
            print(f"--target-sigma {args.target_sigma} requested (safety factor "
                  f"{args.sigma_safety_factor}x): resolved to --segment-length "
                  f"{args.segment_length}, --slide-shift {args.slide_shift:.4f}, "
                  f"--num-longslides {args.num_longslides} "
                  f"(estimated T_bg ~ {t_bg_estimate:.1f} s vs target {t_bg_target:.1f} s)")

        required_longslide_margin = (args.num_longslides - 1) * args.longslide_step
        args.longslide_margin = max(args.longslide_margin, required_longslide_margin)

        num_windows_1 = len(durs1)
        num_windows_2 = len(durs2)

        num_banks = config['GW_search']['num_splits']
        total_jobs = 0
        if args.window in ['both', 'before']:
            total_jobs += num_windows_1 * num_banks * args.num_longslides
        if args.window in ['both', 'after']:
            total_jobs += num_windows_2 * num_banks * args.num_longslides

        num_chunks = max(1, math.ceil(total_jobs / args.chunk_size))
        print(f"Estimated {total_jobs} background search jobs "
              f"({args.num_longslides} long-slide offset(s) x shortslides bundled inside each job).")
    except Exception as e:
        print(f"Warning: Could not pre-calculate jobs, defaulting to 1 chunk. Error: {e}")
        num_chunks = 1
    # -------------------------------------

    def write_sub_file(sub_name, cmd_name):
        sub_path = os.path.join(sub_files_dir, sub_name)
        cmd_path = os.path.join(bin_dir, cmd_name)

        # passing args
        cmd_args = "$(config)"

        if sub_name == "sig_prep.sub":
            cmd_args += (
                f" --window {args.window} --max-extension {args.max_extension} "
                f"--OSW-sigma {args.OSW_sigma} --chunk-size {args.chunk_size} "
                f"--segment-length {args.segment_length} --slide-shift {args.slide_shift} "
                f"--n-ifos {args.n_ifos} --num-longslides {args.num_longslides} "
                f"--longslide-step {args.longslide_step} --longslide-margin {args.longslide_margin} "
                f"--window-max-size {args.window_max_size}"
            )
            if args.ldg_tag: cmd_args += f" --ldg-tag {args.ldg_tag}"
            mem = "2GB"
            disk = "2GB"
        elif sub_name == "sig_post.sub":
            cmd_args += (
                f" --OSW-sigma {args.OSW_sigma} --fit-steps {args.fit_steps} "
                f"--exclude-top-steps {args.exclude_top_steps} --segment-length {args.segment_length} "
                f"--slide-shift {args.slide_shift} --n-ifos {args.n_ifos}"
            )
            mem = "8GB"
            disk = "500MB"

        ldg_line = f"accounting_group = {args.ldg_tag}\n" if args.ldg_tag else ""

        content = f"""executable     = {cmd_path}
arguments      = {cmd_args}
universe       = vanilla

output         = {logs_dir}/{sub_name.replace('.sub', '.out')}
error          = {logs_dir}/{sub_name.replace('.sub', '.err')}
log            = {logs_dir}/sig_pipeline.log

environment    = "PYTHONUNBUFFERED=1"

request_cpus   = 1
request_memory = {mem}
request_disk   = {disk}

{ldg_line}
queue
"""
        with open(sub_path, "w") as f:
            f.write(content)

    # Génération des fichiers .sub (after the slide-shift/target-sigma resolution above,
    # so the resolved values -- not the raw CLI defaults -- get embedded in the .sub files)
    write_sub_file("sig_prep.sub", args.prep_script)
    write_sub_file("sig_post.sub", args.post_script)

    # Génération du DAG
    dag_path = os.path.join(sub_files_dir, "significance.dag")

    search_nodes = ""
    retry_lines = ""
    parents_search = ""
    for i in range(num_chunks):
        chunk_sub = os.path.join(sub_files_dir, f"sig_search_chunk_{i}.sub")
        if not os.path.exists(chunk_sub):
            with open(chunk_sub, "w") as f:
                f.write("# Temporary file for HTCondor DAG validation\n")
        search_nodes += f"JOB SIG_SEARCH_{i} {chunk_sub}\n"
        retry_lines += f"RETRY SIG_SEARCH_{i} 3\n"
        parents_search += f"SIG_SEARCH_{i} "

    dag_content = f"""# Define the nodes for Significance Estimation
JOB SIG_PREP {os.path.join(sub_files_dir, "sig_prep.sub")}
{search_nodes}
FINAL SIG_POST {os.path.join(sub_files_dir, "sig_post.sub")}

{retry_lines}

# Pass the config file path dynamically
VARS SIG_PREP config="{config_path}"
VARS SIG_POST config="{config_path}"

# Define the workflow dependencies
PARENT SIG_PREP CHILD {parents_search}
"""
    with open(dag_path, "w") as f:
        f.write(dag_content)

    if args.submit:
        print(f"Significance Pipeline generated! Automatically submitting to HTCondor...")
        subprocess.run(["condor_submit_dag", "-f", dag_path], check=True)
        print("Submission successful! Check your logs directory for progress.")
    else:
        print(f"Significance Pipeline generated successfully!")
        print(f"To launch your pipeline manually, run: condor_submit_dag {dag_path}")

    # --- CUSTOM PIPELINE MONITOR ---
    if args.monitor:
        print("\n" + "="*50)
        print("PEKET SIGNIFICANCE MONITOR ACTIVE")
        print("Press Ctrl+C at any time to detach and let it run in the background.")
        print("="*50 + "\n")

        sig_prep_out = os.path.join(logs_dir, "sig_prep.out")
        sig_prep_err = os.path.join(logs_dir, "sig_prep.err")
        sig_post_out = os.path.join(logs_dir, "sig_post.out")
        sig_post_err = os.path.join(logs_dir, "sig_post.err")
        bg_out_dir = os.path.join(base_dir, "significance", "out")
        dag_log = f"{dag_path}.dagman.log"

        window_file_name = f'{suffix}_sig_windows_all.txt' if args.window == 'both' else (f'{suffix}_sig_windows_1.txt' if args.window == 'before' else f'{suffix}_sig_windows_2.txt')
        window_file = os.path.join(base_dir, "significance", window_file_name)

        def check_for_critical_errors():
            if os.path.exists(dag_log):
                with open(dag_log, 'r') as dl:
                    log_content = dl.read()
                    if "Job failed" in log_content or "Abnormal termination" in log_content:
                        print("\n\nCRITICAL ERROR: HTCondor reported a job failure in the DAG log!")
                        sys.exit(1)
            for err_file in [sig_prep_err, sig_post_err]:
                if os.path.exists(err_file) and os.path.getsize(err_file) > 0:
                    with open(err_file, 'r') as f:
                        err_text = f.read()

                        is_critical = False
                        if "Traceback" in err_text:
                            is_critical = True
                        elif "Error" in err_text:
                            # On ignore les erreurs si ce sont juste les warnings de cache
                            if "CacheMissingWarning" not in err_text and "MPLCONFIGDIR" not in err_text:
                                is_critical = True

                        if is_critical:
                            print(f"\n\nCRITICAL ERROR IN {os.path.basename(err_file)}:")
                            print(err_text)
                            sys.exit(1)

        try:
            print(f"--- SIGNIFICANCE PREPARATION ---")
            while not os.path.exists(sig_prep_out):
                check_for_critical_errors()
                time.sleep(2)

            with open(sig_prep_out, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        check_for_critical_errors()
                        if "Significance preparation complete!" in line:
                            break
                    else:
                        check_for_critical_errors()
                        time.sleep(3)

            print("\n\n--- PYCBC BACKGROUND ESTIMATION (PARALLEL) ---")
            while not os.path.exists(window_file):
                time.sleep(1)

            # 1. read window file to know which background files to expect (and how many)
            expected_files = set()
            with open(window_file, 'r') as wf:
                for line in wf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bank_num, start_time, end_time, _tt, ls = parts[0], parts[1], parts[2], parts[3], parts[4]
                        # Construction du nom de fichier exact généré par le .sh
                        expected_file = f"{suffix}_bg_bank{bank_num}_{start_time}-{end_time}_ls{ls}.hdf"
                        expected_files.add(expected_file)

            total_jobs = len(expected_files)
            completed = 0

            while completed < total_jobs:
                # 2. scan out dir to see how many of these expected files have been generated
                current_files = set()
                for filepath in glob.glob(f"{bg_out_dir}/bank_*/*.hdf"):
                    current_files.add(os.path.basename(filepath))

                completed = len(expected_files.intersection(current_files))

                percent = int((completed / total_jobs) * 100) if total_jobs > 0 else 0
                bar = '█' * (percent // 5) + '-' * (20 - (percent // 5))
                sys.stdout.write(f"\r[{bar}] {completed}/{total_jobs} Background Slides Completed ({percent}%)")
                sys.stdout.flush()

                if completed < total_jobs:
                    check_for_critical_errors()
                    time.sleep(10)

            print("\n\n--- SIGNIFICANCE POST-PROCESSING ---")
            while not os.path.exists(sig_post_out):
                time.sleep(3)

            with open(sig_post_out, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        check_for_critical_errors()
                    else:
                        check_for_critical_errors()
                        # Fin du job si condor a fini le noeud POST
                        with open(dag_log, 'r') as dl:
                            if "Job SIG_POST completed" in dl.read() or "DAGMAN_FINISHED" in dl.read():
                                break
                        time.sleep(3)

            print("\n\n Significance pipeline completed successfully!")

        except KeyboardInterrupt:
            print("\n\nMonitor detached! Use 'condor_q' to check status later.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
