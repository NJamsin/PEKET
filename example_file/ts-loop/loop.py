import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import glob
import sys

'''
Python script to show how to use kn-ts-loop on a cluster with condor.
Note: Only kept the bu19 grid for the example, also works with other grids named in the same way (bu26_opt, ka17_opt).
Provides example .prior file for the three models.
'''

# REMARK: as the output of the ts loop for a full grid is quite large, I did not upload the grid.
# To run this script, you first need to create the grid with kn-ts-loop, the command below was the one I used to create the bu19 grid 
'''
kn-make-grid --out-dir peket/example_file/ts-loop/bu19 --model Bu2019lm --filters ps1::r ps1::g ps1::z ps1::i ps1::y --cadence 4 --delay 0.25 --obs-duration 10 --detection-limit ps1::r=26 ps1::g=26 ps1::z=26 ps1::i=26 ps1::y=26 --eos-path peket/example_file/KN_grid/eos.dat --noise-level 0
'''

# AFTER CREATING THE GRID, YOU CAN RUN THIS SCRIPT TO APPLY THE TS LOOP ON THE GRID
# list all directory starting with grids in the base dir
BASE_DIR = "peket/example_file/ts-loop" 
BASE_DIR = os.path.abspath(BASE_DIR)
grids = glob.glob(f"{BASE_DIR}/*")
# suppress logs and prior and sub_files directories
grids = [g for g in grids if not any(x in g for x in ["logs", "prior", "sub_files"])]
#just keep directories
grids = [g for g in grids if os.path.isdir(g)]
print(f"Found grids: {grids}")

# get the env variable for the conda environment
bin_dir = os.path.dirname(sys.executable)
exe_path = os.path.join(bin_dir, "kn-ts-loop")

# create a .submit file to apply ts loop on each lc of each grid
def create_submit_file(grid_dir, model, prior, minus_pts, em_prior, out_dir):
    abs_grid_dir = os.path.join(BASE_DIR, grid_dir)
    file = f"""
getenv = True
initialdir = {BASE_DIR}

executable = {exe_path}
arguments = --idx $(Process) --grid-dir {abs_grid_dir} --model {model} --svd-path /home/stu_jamsin/jamsin/NMMA/svdmodels --prior-file {prior} --minus-pts {minus_pts} --nlive 512 --resampling --EM-prior {em_prior}

output = logs/{grid_dir}_$(Process).out
error = logs/{grid_dir}_$(Process).err
log = logs/{grid_dir}_$(Process).log

request_cpus = 1
request_gpus = 0
request_memory = 2GB

queue 25
"""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/logs", exist_ok=True)
    with open(f'{out_dir}/{grid_dir}_ts_loop.submit', 'w') as f:
        f.write(file)
    return 0

# loop over grids and submit jobs
for grid in grids:
    grid_name = os.path.basename(grid)
    print(f"Processing grid: {grid_name}")
    # extract model name from grid name
    try:
        model0 = grid_name.split("_")[0]
    except IndexError:
        model0 = grid_name
    if model0 == "bu26":
        model = "Bu2026_MLP"
    elif model0 == "bu19":
        model = "Bu2019lm"
    elif model0 == "ka17":
        model = "Ka2017"
    print(f"Model: {model}")
    # def prior file path
    prior = f"{BASE_DIR}/prior/{model0}.prior"
    em_prior = f"{BASE_DIR}/prior/{model0}_GW.prior"
    out_dir = f"{BASE_DIR}/sub_files"
    # create submit file
    create_submit_file(grid_name, model, prior, 8, em_prior, out_dir)
    # submit the file
    subprocess.run(f"condor_submit {out_dir}/{grid_name}_ts_loop.submit", shell=True)