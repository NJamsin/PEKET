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

def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor DAG for PyCBC Significance Estimation.")
    # base args
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--prep-script", default="gw-sig-prep", help="Name of the preparation script")
    parser.add_argument("--post-script", default="gw-sig-post", help="Name of the post-processing script")
    
    # submit monitor args
    parser.add_argument("--submit", action="store_true", help="Automatically submit the pipeline to HTCondor")
    parser.add_argument("--monitor", action="store_true", help="Monitor the pipeline execution")
    
    # significance estimation args (passed to prep and post)
    parser.add_argument("--n-slides", default=300, type=int, help="Number of time slides to generate.")
    parser.add_argument("--window", default='both', choices=['both', 'before', 'after'], help="Which off-source window(s) to use.")
    parser.add_argument("--delay", default=0, type=int, help="Delay in seconds/timeslides.")
    parser.add_argument("--max-timeslides", default=4096, type=int, help="Maximum number of slides data duration.")
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"], help="Search window sigma size.")
    parser.add_argument("--ldg-tag", default=None, help="The accounting_group tag required for IGWN.")
    parser.add_argument("--chunk-size", default=3000, type=int, help="Number of jobs per chunk to bypass Schedd limits")

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

    def write_sub_file(sub_name, cmd_name):
        sub_path = os.path.join(sub_files_dir, sub_name)
        cmd_path = os.path.join(bin_dir, cmd_name)
        
        # passing args
        cmd_args = "$(config)"
        
        if sub_name == "sig_prep.sub":
            cmd_args += f" --n-slides {args.n_slides} --window {args.window} --delay {args.delay} --max-timeslides {args.max_timeslides} --OSW-sigma {args.OSW_sigma} --chunk-size {args.chunk_size}"
            if args.ldg_tag: cmd_args += f" --ldg-tag {args.ldg_tag}"
            mem = "2GB"
            disk = "2GB"
        elif sub_name == "sig_post.sub":
            cmd_args += f" --OSW-sigma {args.OSW_sigma}"
            mem = "1GB"
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

    # Génération des fichiers .sub
    write_sub_file("sig_prep.sub", args.prep_script)
    write_sub_file("sig_post.sub", args.post_script)

    # Génération du DAG
    dag_path = os.path.join(sub_files_dir, "significance.dag")
    sig_search_sub = os.path.join(sub_files_dir, "sig_search.sub") 
    
# --- MATH BLOCK FOR EXACT CHUNKING ---
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
        
        def count_windows(start, end, max_size, overlap):
            count = 0; current = start
            while current < end:
                c_end = min(current + max_size, end)
                count += 1
                if c_end == end: break
                current = c_end - overlap
            return count

        w1 = count_windows(OFF1_START, OFF1_END, 1000, 16)
        w2 = count_windows(OFF2_START, OFF2_END, 1000, 16)
        
        num_banks = config['GW_search']['num_splits']
        total_jobs = 0
        if args.window in ['both', 'before']: total_jobs += w1 * args.n_slides * num_banks
        if args.window in ['both', 'after']: total_jobs += w2 * args.n_slides * num_banks
        
        num_chunks = max(1, math.ceil(total_jobs / args.chunk_size))
    except Exception as e:
        print(f"Warning: Could not pre-calculate jobs, defaulting to 1 chunk. Error: {e}")
        num_chunks = 1
    # -------------------------------------

    search_nodes = ""
    retry_lines = ""
    parents_search = ""
    for i in range(num_chunks):
        chunk_sub = os.path.join(sub_files_dir, f"sig_search_chunk_{i}.sub")
        search_nodes += f"JOB SIG_SEARCH_{i} {chunk_sub}\n"
        retry_lines += f"RETRY SIG_SEARCH_{i} 3\n"
        parents_search += f"SIG_SEARCH_{i} "

    dag_content = f"""# Define the nodes for Significance Estimation
JOB SIG_PREP {os.path.join(sub_files_dir, "sig_prep.sub")}
{search_nodes}
JOB SIG_POST {os.path.join(sub_files_dir, "sig_post.sub")}

{retry_lines}

# Pass the config file path dynamically
VARS SIG_PREP config="{config_path}"
VARS SIG_POST config="{config_path}"

# Define the workflow dependencies
PARENT SIG_PREP CHILD {parents_search}
PARENT {parents_search} CHILD SIG_POST
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
                    if len(parts) >= 4:
                        slide, bank_num, start_time, end_time = parts[0], parts[1], parts[2], parts[3]
                        # Construction du nom de fichier exact généré par le .sh
                        expected_file = f"{suffix}_bg_bank{bank_num}_{start_time}-{end_time}_slide{slide}.hdf"
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