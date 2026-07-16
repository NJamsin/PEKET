# kn-make-grid

Produces a grid of pseudo-randomly sampled (Halton sampling) synthetic kilonovae lightcurves generated via NMMA/FIESTA. 

Supported models: all the models in [models in config.py](https://github.com/NJamsin/peket/blob/main/src/peket/kn_side/config.py).

## Arguments
* `--out-dir`: Base directory for output files.
* `--model`: Model name to use (default: Bu2026_MLP). 
* `--num-lc`: Number of lightcurves to generate (default: 25).
* `--filters`: Filters used for fake observations (e.g., `--filters ps1::r ps1::g`).
* `--eos-path`: Path to the EOS file (`.dat`) containing mass and radius columns.
* `--noise-level`: Standard deviation of Gaussian noise added to magnitude (default: 0.2).
* `--max-error-level`: Maximum error added to the model magnitude (default: 0.4).
* `--trigger-isot`: Trigger time for synthetic lightcurves (default: '2020-01-07T00:00:00').
* `--cadence`: Number of detections per day.
* `--delay` : Delay in days between the trigger and the first observation.
* `--obs-duration`: Observation duration in days (default: 7).
* `--jitter`: Maximum time fluctuation added to sample times in days (default: 0).
* `--detection-limit`: Detection limits format: `filter1:limit1 filter2:limit2`.
* `--param-ranges`: Parameter ranges format: `param1=(min,max) param2=(min,max)`.
* `--save-json`: Saves times and magnitudes for each sample in a JSON file.
* `--svd-path`: Path to the directory containing the svd models (Not needed for `Bu2026_MLP`).

## Example
```bash
kn-make-grid --out-dir peket/example_file/KN_grid \
             --filters ps1::r ps1::g ps1::z ps1::i ps1::y \
             --eos-path peket/example_file/KN_grid/eos.dat \
             --detection-limit ps1::r=24 ps1::g=20 ps1::z=30 \
             --save-json --cadence 3 --delay 1 --jitter 0.25 \
             --svd-path /path/to/your/svd/directory \
```