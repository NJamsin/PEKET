# gw-setup-sig

This command-line tool estimates the statistical significance of the top trigger identified during the coherent search. 

Unlike the legacy `gw-search-significance` script, `gw-setup-sig` orchestrates a robust **3-node HTCondor DAG** (Preparation -> Parallel Search -> Post-Processing). This prevents cluster overloads, handles "black hole" nodes automatically, and allows accumulating background jobs seamlessly.

The repository also ships `gw-setup-sig3`, which is the newer variant. It adds explicit control over segment length, slide shift, long-slide tiling, and target-significance planning, so it is the preferred reference when you need the current background strategy.

## Usage
`gw-setup-sig` requires the path to the same `.yaml` config file used for the initial search.

```bash
gw-setup-sig path/to/config.yaml [OPTIONS]
```

## Arguments 

**Basic Options:**
- `config` *(Obligatory)*: Path to the configuration file.
- `--submit`: Automatically submits the DAG workflow to HTCondor.
- `--monitor`: Live monitors the execution. The monitor reads Condor logs directly, providing an accurate, fail-proof progress bar.
- `--ldg-tag`: The `accounting_group` tag required for IGWN/LDG clusters (e.g., `ligo.sim.o3.cbc.bns.pycbcoffline`).

**Background Generation Parameters:**
- `--n-slides`: Number of time slides to generate for background estimation (Default: `300`).
- `--window`: Specifies which off-source window(s) to use (`both`, `before`, `after`). Default is `both`. Only the required data is downloaded.
- `--delay`: Delay in slides between the limits of the OSW and the time slides. Extremely useful for chunking massive background estimations into smaller sequential runs without overwriting previous results.
- `--max-timeslides`: Maximum duration (in seconds) of background data to prepare (Default: `4096`).
- `--OSW-sigma`: Size of the On-Source time window (Choices: `1`, `2`, `3`, `full`. Default: `1`).
- `--chunk-size`: Number of jobs per chunk to bypass Schedd limits.
- `--fit-steps`: Number of background points used for the exponential fit in post-processing.
- `--exclude-top-steps`: Number of loudest points excluded from the exponential fit.

## gw-setup-sig3

`gw-setup-sig3` keeps the same high-level job layout but exposes the more advanced background planning logic from `setup_sig_pipeline3.py`.

### Additional Arguments
- `--max-extension`: Extra off-source data to download beyond the nominal window.
- `--segment-length`: `pycbc_multi_inspiral` segment length in seconds.
- `--slide-shift`: `pycbc_multi_inspiral` slide shift in seconds.
- `--n-ifos`: Number of interferometers used in the background planning.
- `--target-sigma`: Automatically expands the background strategy to reach a target FAP.
- `--sigma-safety-factor`: Safety factor applied when targeting a given sigma.
- `--num-longslides`: Number of long-slide offsets per window.
- `--longslide-step`: Spacing between long-slide offsets.
- `--longslide-margin`: Extra headroom for the long-slide downloads.
- `--window-max-size`: Maximum duration of a single off-source analysis window.

## Workflow Structure
The command generates a `significance.dag` file containing three nodes:
1. **SIG_PREP:** Calculates windows, downloads *only* the required data, cleans overlaps, generates PyCBC caches (`.lcf`), and generates the submit files.
2. **SIG_SEARCH:** Massively parallel PyCBC search over the time slides. It incorporates "Late Materialization" (`max_materialize`, `max_idle`) to respect strict cluster limits, and a `periodic_remove` fail-safe to kill hanging jobs.
3. **SIG_POST:** Safely parses the output `.hdf` files (verifying file integrity), accurately computes the total background time (T_bg) without double-counting overlaps, and computes the False Alarm Rate (FAR) and p-value.

## Output & Visualizations
Upon successful execution, the tool outputs:
1. **`out/run_name_significance.txt`**: Text file containing the Top trigger ranking statistic, T_bg, FAR, and the on-source p-value.
2. **`plots/run_name_far_vs_snr.png`**: A plot of the empirical background distribution (FAR vs SNR) with the top candidate highlighted.

## Example
Running a 500-slide background estimation, chunked with a delay of 0, automatically submitted to an IGWN cluster:

```bash
gw-setup-sig /path/to/search_config.yaml --n-slides 500 --submit --monitor --ldg-tag ligo.sim.o3.cbc.bns.pycbcoffline
```

**Console Output:**
```text
Significance Pipeline generated! Automatically submitting to HTCondor...
Submission successful! Check your logs directory for progress.

==================================================
PEKET SIGNIFICANCE MONITOR ACTIVE
Press Ctrl+C at any time to detach and let it run in the background.
==================================================

--- SIGNIFICANCE PREPARATION ---
On-source window (GPS): 1187006504 - 1187008913
Downloading and preparing data for Off-Source Background estimation...
Locating 4kHz data for H1...
Downloading files for first off-source window...
[...]
Significance preparation complete!

--- PYCBC BACKGROUND ESTIMATION (PARALLEL) ---
[████████████████████] 500/500 Background Slides Completed (100%)

--- SIGNIFICANCE POST-PROCESSING ---
Collecting background triggers from /gw_search/significance/out...

──────────────────────────────────────────────────
  Top trigger stat     : 22.2409
  Louder than top      : 0
  T_background         : 12000.0 s  (0.000 yr)
  FAR                  : < 8.33e-05 Hz  (< 2630.0 /yr)
  p-value (on-source)  : < 1.81e-01
──────────────────────────────────────────────────

Generating FAR vs SNR plot...
  FAR vs SNR plot saved to: /plots/run_name_far_vs_snr.png
Significance estimation complete.

 Significance pipeline completed successfully!
```