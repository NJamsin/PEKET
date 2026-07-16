# WARNING: Obsolete CLI (still implemented but not useful)

# kn-ts-loop

Performs iterative inference on synthetic or real kilonova lightcurve data by progressively removing early detection points. Evaluates how the inferred timeshift evolves.

**NOTE**: The `--resampling` flag is not compatible with `Ka2017`.

## Arguments
* `--idx`: Index of the injection/lightcurve to analyze (default: 0).
* `--grid-dir`: Path to the base grid directory containing your data.
* `--model`: EM model to use (default: Bu2019lm).
* `--svd-path`: Path to the SVD models directory.
* `--prior-file`: Path to the prior file for lightcurve analysis.
* `--minus-pts`: Number of early time points to progressively remove (default: 2).
* `--add-ul`: Adds an upper limit instead of completely removing early points (default: False).
* `--true-merger-time`: True merger time for timeshift calculation in ISOT format.
* `--nlive`: Number of live points for nested sampling (default: 1024).
* `--resampling`: Runs the resampling step via `gwem-resampling` after EM inferences.
* `--eos-posterior`: Path to EOS posterior probability file for GW samples. It is recommended to use [this one](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/example_files/tools/gwem_resampling/posterior_probability.txt).
* `--eos-path`: Path to EOS files used for resampling.
* `--GW-prior`: Path to GW prior file for resampling.
* `--EM-prior`: Path to EM prior file for resampling.
* `--restrict-dist-prior`: Restrict the distance prior, will restrict the prior as true_val +/- value.

## Example
Iterative inference removing the first 2 points, adding upper limits, and executing resampling:

```bash
kn-ts-loop --idx 0 \
           --grid-dir /path/to/KN_grid \
           --model Bu2019lm \
           --svd-path /path/to/svd \
           --prior-file /path/to/EM.prior \
           --minus-pts 2 \
           --add-ul \
           --resampling
           --eos-path /path/to/eos.dat \
           --eos-posterior /path/to/posterior.txt \
```
The output will be systematically organized inside your grid directory under `{grid_dir}/{idx}/minusX/`.