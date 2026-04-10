#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import argparse
import yaml
import subprocess 
import time
import glob

def main():
    parser = argparse.ArgumentParser(description="Generate HTCondor DAG for PyCBC search.")
    # functionning arg
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--prep-script", default="gw-search-prep", help="Name of the preparation script")
    parser.add_argument("--post-script", default="gw-search-post", help="Name of the post-processing script")
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

    # check that prep and post .out file exist, if so deletes them to avoid problem with the --monitor flag
    prep_out = os.path.join(logs_dir, "prep.out")
    post_out = os.path.join(logs_dir, "post.out")
    if os.path.exists(prep_out):
        os.remove(prep_out)
    if os.path.exists(post_out):
        os.remove(post_out)
    
    # Dynamically grab the exact Python interpreter currently running this CLI
    current_python = sys.executable 

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

        content = f"""executable     = {cmd_path}
arguments      = {cmd_args}
universe       = vanilla

output         = {logs_dir}/{sub_name.replace('.sub', '.out')}
error          = {logs_dir}/{sub_name.replace('.sub', '.err')}
log            = {logs_dir}/pipeline.log

environment    = "PYTHONUNBUFFERED=1"

request_cpus   = 1
request_memory = 16GB

queue
"""
        with open(sub_path, "w") as f:
            f.write(content)

    # Call the command using argparse
    if args.skip_search:
        print("Skipping the search step as per the --skip-search flag. Only generating the post-processing job.")
        write_sub_file("post.sub", args.post_script if args.post_script else "GWsearch_post.py")
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
        write_sub_file("prep.sub", args.prep_script if args.prep_script else "GWsearch_prep.py")
        write_sub_file("post.sub", args.post_script if args.post_script else "GWsearch_post.py")

    '''
    Create the DAG file
    '''
    dag_path = os.path.join(base_dir, "sub_files", "pipeline.dag")
    split_search_sub = os.path.join(base_dir, "sub_files", "split_search.sub")
    
    dag_content = f"""# Define the nodes
JOB PREP {os.path.join(base_dir, "sub_files", "prep.sub")}
JOB SEARCH {split_search_sub}
JOB POST {os.path.join(base_dir, "sub_files", "post.sub")}

# Pass the config file path into the prep and post jobs dynamically
VARS PREP config="{config_path}"
VARS POST config="{config_path}"

# Define the workflow dependencies
PARENT PREP CHILD SEARCH
PARENT SEARCH CHILD POST
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
        print("PIPELINE MONITOR ACTIVE")
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
                        if "Traceback" in err_text or "Error" in err_text:
                            print("\n\nCRITICAL ERROR IN PREP STAGE:")
                            print(err_text)
                            sys.exit(1)
                #  Check specific error files (if they exist and have content)
                if os.path.exists(post_err) and os.path.getsize(post_err) > 0:
                    with open(post_err, 'r') as err_file:
                        err_text = err_file.read()
                        if "Traceback" in err_text or "Error" in err_text:
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