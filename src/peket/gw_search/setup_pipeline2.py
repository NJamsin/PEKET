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
from gwosc.timeline import get_segments
import math


def get_coincident_segments(detectors, start, end, min_ifos=2, flag_suffix="DATA", margin=0):
    """
    EXACT COPY of the function used in GWsearch_prep2.py. Kept in sync manually
    (duplicated on purpose to avoid an import dependency on the prep script's
    file name/path, which is user-configurable via --prep-script). If you touch
    the coincidence logic in the prep script, mirror the change here too, or the
    DAG's job count will drift from what prep actually generates again.

    Sweep-line over per-detector GWOSC science segments (gwosc.timeline.get_segments)
    to find contiguous segments where at least `min_ifos` detectors are simultaneously
    active. Returns a list of (seg_start, seg_end, (ifo1, ifo2, ...)) sorted by seg_start.
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

    return [(s, e, ifos) for (s, e, ifos) in raw_segments if len(ifos) >= min_ifos]


def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor DAG for PyCBC search.")
    # functionning arg
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--prep-script", default="gw-search-prep2", help="Name of the preparation script")
    parser.add_argument("--post-script", default="gw-search-post2", help="Name of the post-processing script")
    # add arg
    parser.add_argument("--submit", action="store_true", help="Automatically submit the pipeline to HTCondor after generation")
    parser.add_argument("--injection", default=False, action="store_true", help="If true, will inject a fake signal inside the time windows to be searched, for testing purposes. The injection parameters will be read from the config file (under the 'Injection' section).")
    parser.add_argument("--expected-trigger-time", default=None, help="Expected trigger time to be searched, in gps format. Used only in the final trigger distribution plot.")
    parser.add_argument("--skip-search", default=None, action="store_true", help="If true, will skip the search step and directly run the post-processing script. Useful for testing the post-processing independently or if you already have triggers generated from a previous search run.")
    parser.add_argument("--plot-spectrogram", default=None, action="store_true", help="If true, will generate a spectrogram plot for the top trigger in the post-processing step. This can be useful for visually inspecting the trigger.")
    parser.add_argument("--spectrogram-range", default="0,15", help="vmin and vmax for the spectrogram plot. Only used if --plot-spectrogram is set.")
    parser.add_argument("--monitor", default=False, action="store_true", help="If true, will monitor the pipeline execution.")
    parser.add_argument("--template-bank", default=None, help="Path to the template bank file if you want to specify it instead of generating through the resampling posterior. This can be useful if you want to use a custom template bank or if you want to skip the template bank generation. The template bank will still be split for parrallelization. /!\\ Expect an hdf file.")
    parser.add_argument("--detector-threshold", default=0.5, type=float, help="Minimum antenna response required to launch the search. Default is 0.5, can be useful to avoid long search for time windows where the detectors are barely sensitive to the source. Only applied to injections because the merger time is needed for the antenna response.")
    parser.add_argument("--plot-antenna-pattern", default=None, action="store_true", help="If true, will generate an antenna pattern plot for the source location and the injection merger time. Only applied to injections because the merger time is needed for the antenna response. /!\\ The plot is generated at the end of the preparation so if the search is stopped by the threshold it won't be generated.")
    parser.add_argument("--OSW-sigma", default='1', choices=['1','2','3', "full"], help="Size of the time window to be searched around the expected trigger time, in sigmas. Default is 1.")
    parser.add_argument("--tmplt-sigma", default='1', choices=['1','2','3', "full"], help="Size of the the template bank to be used for template bank generation around the expected trigger time, in sigmas. Default is 1. /!\\ If you specify a custom template bank with --template-bank, this argument will be ignored.")
    parser.add_argument("--disk", default="3GB", help="Amount of disk space to request for the prep job. Default is \'3GB\'.")
    parser.add_argument("--chunk-size", default=3000, type=int, help="Number of jobs per chunk")
    # Signifiance related args (deprecated)
    parser.add_argument("--compute-significance", action="store_true", help="If true, runs a significance job after the search to estimate FAR and p-value.")
    parser.add_argument("--significance-method", default="offsource", choices=["offsource", "timeslides"], help="Method to use for background estimation.")
    parser.add_argument("--n-background", default=50, type=int, help="Number of background windows/slides to use.") 
    # LDG tag
    parser.add_argument("--ldg-tag", default=None, help="The \"accounting_group\" tag required for submitting to the LDG cluster. If not specified, the pipeline will be generated without the tag.")
    parser.add_argument("--gwdata-server", default="datafind.ligo.org:443", help="GW data server to use for fetching segments. Default is datafind.ligo.org:443 (LDG datafind only).")
    # Multi-detector / coincidence args -- MUST match the defaults/behavior of GWsearch_prep2.py,
    # since they're used both to pass flags to the prep job AND to pre-compute the exact number
    # of search jobs/chunks for the DAG (see get_coincident_segments above).
    parser.add_argument("--detectors", default="H1,L1,V1", help="Comma-separated list of candidate detectors for the search. Default: H1,L1,V1")
    parser.add_argument("--min-ifos", default=2, type=int, help="Minimum number of detectors simultaneously in science mode required to analyze a time segment. Default is 2.")
    parser.add_argument("--dq-flag", default="DATA", help="GWOSC timeline flag suffix used to build per-detector segment lists (e.g. 'DATA' or 'CBC_CAT1'). Default: DATA")
    parser.add_argument("--segment-margin", default=40, type=int, help="Seconds trimmed off both ends of each raw science segment before computing coincidence. Default: 40")
    parser.add_argument("--min-analysis-length", default=64, type=int, help="Minimum usable duration (s) of a coincident chunk to be submitted as a job. Default: 64")

    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    base_dir = os.path.abspath(config['Directory']['BASE_DIR'])

    # Ensure BASE_DIR and a logs folder exist
    os.makedirs(base_dir, exist_ok=True)
    logs_dir = os.path.join(base_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    sub_files_dir = os.path.join(base_dir, 'sub_files')
    os.makedirs(sub_files_dir, exist_ok=True)

    # check that prep and post .out & .err files exist, if so deletes them to avoid problem with the --monitor flag
    prep_out = os.path.join(logs_dir, "prep.out")
    post_out = os.path.join(logs_dir, "post.out")
    prep_err = os.path.join(logs_dir, "prep.err")
    post_err = os.path.join(logs_dir, "post.err")
    if os.path.exists(prep_out):
        os.remove(prep_out)
    if os.path.exists(post_out):
        os.remove(post_out)
    if os.path.exists(prep_err):
        os.remove(prep_err)
    if os.path.exists(post_err):
        os.remove(post_err)  

    '''
    Create prep.sub and post.sub
    We pass the python interpreter directly as the executable (no more conda dependance) (I wanted to make this usable outside my personnal configuration)
    '''
    bin_dir = os.path.dirname(sys.executable)

    def write_sub_file(sub_name, cmd_name):
        sub_path = os.path.join(sub_files_dir, sub_name)
        cmd_path = os.path.join(bin_dir, cmd_name)
        # Add dynamic arg to be passed to the prep and post scripts, so that they can read the config file path from the DAG variables
        cmd_args = "$(config)"
        if args.injection: # add the --injection flag to the command if the user specified it to both prep and post 
            cmd_args += " --injection"
        if args.template_bank and sub_name == "prep.sub": # add the --template-bank flag to the command if the user specified it to both prep and post
            cmd_args += f" --template-bank {args.template_bank}"
        if args.detector_threshold and sub_name == "prep.sub": # add the --detector-threshold flag to the command if the user specified it to both prep and post
            cmd_args += f" --detector-threshold {args.detector_threshold}"
        if args.plot_antenna_pattern and sub_name == "prep.sub": # add the --plot-antenna-pattern flag to the command if the user specified it to both prep and post
            cmd_args += f" --plot-antenna-pattern"
        if args.expected_trigger_time and sub_name == "post.sub": # only add the --expected-trigger-time flag to the post script, since it's the one that will generate the final trigger distribution plot
            cmd_args += f" --expected-trigger-time {args.expected_trigger_time}"
        if args.plot_spectrogram and sub_name == "post.sub": # only add the --plot-spectrogram flag to the post script, since it's the one that will generate the spectrogram plots
            cmd_args += f" --plot-spectrogram --spectrogram-range {args.spectrogram_range}"
        if args.OSW_sigma: # only add the --OSW-sigma flag tto both prep and post
            cmd_args += f" --OSW-sigma {args.OSW_sigma}"
        if args.tmplt_sigma and sub_name == "prep.sub": # only add the --tmplt-sigma flag to the prep script, since it's the one that will generate the template bank
            cmd_args += f" --tmplt-sigma {args.tmplt_sigma}"
        if args.ldg_tag and sub_name == "prep.sub": 
            cmd_args += f" --ldg-tag {args.ldg_tag}"
        if sub_name == "prep.sub": # detector list & coincidence params only concern the prep step
            cmd_args += f" --detectors {args.detectors}"
            cmd_args += f" --min-ifos {args.min_ifos}"
            cmd_args += f" --dq-flag {args.dq_flag}"
            cmd_args += f" --segment-margin {args.segment_margin}"
            cmd_args += f" --min-analysis-length {args.min_analysis_length}"
        if sub_name == "significance.sub": # for the significance job, we also need to pass the config path as an argument to be able to read the SIG_WINDOW_FILE variable
            cmd_args += f" --method {args.significance_method}"
            cmd_args += f" --n-background {args.n_background}"
            mem = "1GB"
        elif sub_name == "prep.sub": # the prep step can be a bit more memory intensive because of the template bank generation, especially if the user specified a low detector threshold that leads to long time windows. 
            mem = "16GB"
            disk = args.disk
            cmd_args += f" --chunk-size {args.chunk_size}"
        else:
            mem = "512MB"
            disk = "100MB"
        
        if args.ldg_tag:
            ldg_line = f"accounting_group = {args.ldg_tag}\n"
        else:            
            ldg_line = ""

        datafind_cfg = config['GW_search'].get('datafind', {})
        use_datafind = bool(datafind_cfg)
        if use_datafind and sub_name == "prep.sub":
            env_line = f"environment = \"PYTHONUNBUFFERED=1; GW_DATA_SERVER={args.gwdata_server}\"\n"
        else:
            env_line = "environment = \"PYTHONUNBUFFERED=1\"\n"
      

        content = f"""executable     = {cmd_path}
arguments      = {cmd_args}
universe       = vanilla

output         = {logs_dir}/{sub_name.replace('.sub', '.out')}
error          = {logs_dir}/{sub_name.replace('.sub', '.err')}
log            = {logs_dir}/pipeline.log

{env_line}

request_cpus   = 1
request_memory = {mem}
request_disk   = {disk}

{ldg_line}

queue
"""
        with open(sub_path, "w") as f:
            f.write(content)

    # Call the command using argparse
    if args.skip_search:
        print("Skipping the search step as per the --skip-search flag. Only generating the post-processing job.")
        write_sub_file("post.sub", args.post_script if args.post_script else "GWsearch_post2.py")
        # Create a simplified DAG file with only the post-processing step
        dag_path = os.path.join(base_dir, "sub_files", "pipeline_post_only.dag")
        post_sub = os.path.join(base_dir, "sub_files", "post.sub")
        dag_content = f"""# Define the node
JOB POST {post_sub}
# Pass the config file path into the post job dynamically
VARS POST config="{config_path}"
"""
        with open(dag_path, "w") as f:
            f.write(dag_content)
        if args.submit:
            print(f"Post-processing job generated! Automatically submitting to HTCondor...")
            subprocess.run(["condor_submit_dag", "-f", dag_path], check=True)
            print("Submission successful! Check your logs directory for progress or run condor_q.")
        else:
            print(f"Post-processing job generated successfully!")
            print(f"To launch the post-processing job manually, run: condor_submit_dag {dag_path}")

        return 0
    else:
        write_sub_file("prep.sub", args.prep_script if args.prep_script else "GWsearch_prep2.py")
        write_sub_file("post.sub", args.post_script if args.post_script else "GWsearch_post2.py")
        if args.compute_significance:
            write_sub_file("significance.sub", "GWsignifiance.py")

    '''
    Create the DAG file
    '''
    dag_path = os.path.join(base_dir, "sub_files", "pipeline.dag")
    split_search_sub = os.path.join(base_dir, "sub_files", "split_search.sub")
    
    sig_dag_lines = ""
    if args.compute_significance:
        sig_dag_lines = f"""JOB SIG {sub_files_dir}/significance.sub
VARS SIG config="{config_path}"
PARENT SEARCH CHILD SIG
"""

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
        
        window_size = config['GW_search']['window_size']
        overlap = 16

        # Mirror EXACTLY the windowing logic of GWsearch_prep2.py's Step 3: only
        # chunk inside segments where at least --min-ifos detectors were actually
        # on (per GWOSC timeline), otherwise num_chunks (and therefore the number
        # of JOB SEARCH_i nodes declared below) won't match the split_search_chunk_*.sub
        # files that prep will actually generate.
        prep_detectors = [d.strip() for d in args.detectors.split(',') if d.strip()]
        print(f"Querying GWOSC timeline ({args.dq_flag}) for {prep_detectors} to pre-calculate job count...")
        coincident_segments = get_coincident_segments(
            prep_detectors, ON_START, ON_END,
            min_ifos=args.min_ifos, flag_suffix=args.dq_flag, margin=args.segment_margin
        )

        w_count = 0
        for seg_start, seg_end, ifos in coincident_segments:
            current = seg_start
            while current < seg_end:
                c_end = min(current + window_size, seg_end)
                if c_end - current < args.min_analysis_length:
                    break
                w_count += 1
                if c_end == seg_end:
                    break
                current = c_end - overlap

        if w_count == 0:
            raise RuntimeError(f"No segment with at least {args.min_ifos} detectors on was found "
                                f"between {ON_START} and {ON_END}. The prep job would abort immediately.")

        num_splits = config['GW_search']['num_splits']
        total_jobs = w_count * num_splits
        num_chunks = max(1, math.ceil(total_jobs / args.chunk_size))
        print(f" -> {w_count} coincident window(s) x {num_splits} banks = {total_jobs} jobs -> {num_chunks} chunk(s).")
    except Exception as e:
        print(f"Warning: Could not pre-calculate jobs, defaulting to 1 chunk. Error: {e}")
        num_chunks = 1
    # -------------------------------------

    search_nodes = ""
    retry_lines = ""
    parents_search = ""
    for i in range(num_chunks):
        chunk_sub = os.path.join(sub_files_dir, f"split_search_chunk_{i}.sub")
        search_nodes += f"JOB SEARCH_{i} {chunk_sub}\n"
        retry_lines += f"RETRY SEARCH_{i} 3\n"
        parents_search += f"SEARCH_{i} "

    dag_content = f"""# Define the nodes
JOB PREP {os.path.join(sub_files_dir, "prep.sub")}
{search_nodes}
FINAL POST {os.path.join(sub_files_dir, "post.sub")}

{retry_lines}

# Pass the config file path dynamically
VARS PREP config="{config_path}"
VARS POST config="{config_path}"

# Define the workflow dependencies
PARENT PREP CHILD {parents_search}
"""
    with open(dag_path, "w") as f:
        f.write(dag_content)

    if args.submit:
        print(f"Pipeline generated! Automatically submitting to HTCondor...")
        # Run the condor_submit_dag command using subprocess
        subprocess.run(["condor_submit_dag", "-f", dag_path], check=True)
        print("Submission successful! Check your logs directory for progress or run condor_q.")
    else:
        print(f"Pipeline generated successfully!")
        print(f"To launch your pipeline manually, run: condor_submit_dag {dag_path}")

    # --- THE CUSTOM PIPELINE MONITOR ---
    if args.monitor:

        print("\n" + "="*50)
        print("PEKET PIPELINE MONITOR ACTIVE")
        print("Press Ctrl+C at any time to detach and let it run in the background.")
        print("="*50 + "\n")

        # Define log file paths to monitor based on the config
        SUFFIX = config['Directory']['run_name']
        prep_out = f"{base_dir}/logs/prep.out"
        prep_err = f"{base_dir}/logs/prep.err"
        post_out = f"{base_dir}/logs/post.out"
        post_err = f"{base_dir}/logs/post.err"
        trigger_dir = f"{base_dir}/out"
        window_file = f"{base_dir}/{SUFFIX}_windows.txt"
        dag_log = f"{dag_path}.dagman.log"

        def check_for_critical_errors():
                """Checks standard error files and the DAG log for fatal crashes."""
                # Check Condor DAG log for overall job failures
                if os.path.exists(dag_log):
                    with open(dag_log, 'r') as dl:
                        log_content = dl.read()
                        if "Job failed" in log_content or "Abnormal termination" in log_content:
                            print("\n\nCRITICAL ERROR: HTCondor reported a job failure in the DAG log!")
                            sys.exit(1)
                
                # Check specific error files (if they exist and have content)
                if os.path.exists(prep_err) and os.path.getsize(prep_err) > 0:
                    with open(prep_err, 'r') as err_file:
                        err_text = err_file.read()
                        is_critical = False
                        if "Traceback" in err_text:
                            is_critical = True
                        elif "Error" in err_text:
                            # On ignore les erreurs si ce sont juste les warnings de cache
                            if "CacheMissingWarning" not in err_text and "MPLCONFIGDIR" not in err_text:
                                is_critical = True

                        if is_critical:
                            print(f"\n\nCRITICAL ERROR IN {os.path.basename(prep_err)}:")
                            print(err_text)
                            sys.exit(1)
                #  Check specific error files (if they exist and have content)
                if os.path.exists(post_err) and os.path.getsize(post_err) > 0:
                    with open(post_err, 'r') as err_file:
                        err_text = err_file.read()
                        is_critical = False
                        if "Traceback" in err_text:
                            is_critical = True
                        elif "Error" in err_text:
                            # On ignore les erreurs si ce sont juste les warnings de cache
                            if "CacheMissingWarning" not in err_text and "MPLCONFIGDIR" not in err_text:
                                is_critical = True

                        if is_critical:
                            print("\n\nCRITICAL ERROR IN POST STAGE:")
                            print(err_text)
                            sys.exit(1)

        try:
            print(f"--- SEARCH PREPARATION ---")
            while not os.path.exists(prep_out):
                check_for_critical_errors() # Look for instant crashes
                time.sleep(2)
            # Wait for Condor to create the prep log
            while not os.path.exists(prep_out):
                time.sleep(5)
            
            # Stream the file live
            with open(prep_out, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        # check for error 
                        check_for_critical_errors() # Look for crashes that happen after the initial prep log creation
                        # Stop streaming when we see your specific success message!
                        if "Search preparation complete!" in line:
                            break
                    else:
                        check_for_critical_errors() # Look for crashes that happen after the initial prep log creation
                        time.sleep(3) # wait a bit before trying to read new lines to avoid busy waiting
            
            print("\n\n--- PYCBC SEARCH (PARALLEL) ---")
            # Figure out how many search jobs to expect by counting lines in the windows file (each line corresponds to a search job for one time window)
            while not os.path.exists(window_file):
                time.sleep(1)
            with open(window_file, 'r') as wf:
                total_jobs = sum(1 for line in wf)

            # Live Progress Bar
            completed = 0
            while completed < total_jobs:
                # Just count the trigger files generated! If the files already exist the monitoring will directly print 100% BUT the search will be re running !!!!
                completed = len(glob.glob(f"{trigger_dir}/*.hdf")) # assuming only one run in this directory
                
                percent = int((completed / total_jobs) * 100)
                bar = '█' * (percent // 5) + '-' * (20 - (percent // 5))
                sys.stdout.write(f"\r[{bar}] {completed}/{total_jobs} Search Windows Completed ({percent}%)")
                sys.stdout.flush()
                
                if completed < total_jobs:
                    # check for critical errors during the search
                    check_for_critical_errors() # Look for crashes that happen during the search
                    time.sleep(5)

            print("\n\n--- POST-PROCESSING ---")
            while not os.path.exists(post_out):
                time.sleep(3)
                
            with open(post_out, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        sys.stdout.write(line)
                        # check for error
                        check_for_critical_errors() # Look for crashes that happen after the initial post log creation
                        # Replace this with whatever the final line of your post script prints!
                        if "Post-processing completed. Check the output and plots directory." in line: 
                            break
                    else:
                        check_for_critical_errors() # Look for crashes that happen after the initial post log creation
                        time.sleep(3)

            print("\n\n Search pipeline completed successfully! Check the logs for details and outputs and plots for results.")

        except KeyboardInterrupt:
            print("\n\nMonitor detached! Use 'condor_q' to check status later.")

    return 0

if __name__ == '__main__':
    sys.exit(main())