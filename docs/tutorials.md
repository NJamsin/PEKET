# Tutorials & Utilities

This section provides information on the tutorials available in the repository, as well as a detailed reference for the Python utility functions included in the `peket.kn_side.utils` module.

---

## Available Tutorials
Examples and tutorials are provided in the GitHub repository under the [`example_file`](https://github.com/NJamsin/PEKET/tree/main/example_file) directory:

* **Real KN-GW Data:** Configuration for analyzing AT2017gfo-GW170817. Outputs are located in the [`test_real_data`](https://github.com/NJamsin/PEKET/tree/main/test_real_data) directory.
* **Artificial Injection:** A KN generated with NMMA (`Bu2019lm`) with an injected GW in real H1/L1 strain data. Outputs are located in [`test_injection`](https://github.com/NJamsin/PEKET/tree/main/test_injection).
* **Timeshift Loop Tutorial:** A guide on combining `kn-ts-loop`, `kn-make-grid`, and `plot_param_evolution` is available in the [`example_file/ts-loop`](https://github.com/NJamsin/PEKET/tree/main/example_file/ts-loop) directory. It includes example prior files for all supported models.

The synthetic-lightcurve helpers all use the model registry declared in `peket.kn_side.config`. The current registry includes the `Bu2019lm` and `Ka2017` grids, the newer `Bu2025_*`, `Bu2026_*`, and `Brethauer2025_*` surrogate families, and the GRB-oriented `afgpy_*`, `blastwave_*`, and `pbag_*` model families.

---

## Utilities (`peket.kn_side.utils`)

The package provides several Python functions designed to help you generate synthetic data and plot your results.

### Plotting Utilities

#### `plot_param_evolution`
Generates summary plots (pp plot and so called "evolution plot") for the complete analysis of a grid processed with `kn-ts-loop`. It visualizes how the inferred timeshift and other EM parameters evolve as early data points are removed.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `MODEL` | `str` | *Obligatory* | Name of the EM model (e.g., `'Bu2026_MLP'`). |
| `DIR` | `str` | *Obligatory* | Directory where the timeshift loop results are stored. |
| `UL` | `bool` | `False` | Set to `True` if the loop was run with upper limits substitution. |
| `true_merger` | `str` | `'2020-01-07T00:00:00.000'` | ISOT time of the true merger. |
| `minus_num` | `int` | `4` | Number of early points progressively removed during the loop. |
| `ts_max` | `float` | `-2.5` | Maximum timeshift value (negative) to be plotted. |
| `col_num` | `int` | `5` | Number of columns in the plot grid. |
| `row_num` | `int` | `5` | Number of rows in the plot grid. |

*Returns:* `None`. Automatically saves plots (evolution and injection-recovery P-P plots) in `{DIR}/plots/`.

---

### LSST Synthetic Lightcurve Generation
These functions interact with `rubin_sim` to simulate realistic Vera C. Rubin Observatory (LSST) cadences and limiting magnitudes.

#### `get_lsst_observations`
Retrieves and sorts LSST observations from a SQLite database.
* **Parameters:** `db_path` (str), `full_df` (bool, default `False`), `n_visits` (int, default `20`).
* **Returns:** A pandas DataFrame containing `expMJD`, `filter`, `fiveSigmaDepth`, `_ra`, and `_dec`.

#### `find_valid_sources`
Finds valid random sources in the LSST database that have observations during their expected kilonova lifetime and are located within the LSST field of view.
* **Parameters:** `df_lsst` (DataFrame), `n_sources` (int, default `5`), `duration` (int, default `20`), `min_observations` (int, default `3`).
* **Returns:** A list of dictionaries, each containing the source's `ra`, `dec`, `t0_mjd`, and its relevant `observations`.

#### `generate_synth_lc_lsst`
Generates a full synthetic lightcurve based on LSST observations of a given source and EM model, applying realistic 5-sigma depth noise and upper limits.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `dict` | *Obligatory* | Dictionary containing LSST observations (output from `find_valid_sources`). |
| `model_name` | `str` | *Obligatory* | Name of the EM model (e.g., `'Bu2026_MLP'`, `'Bu2019lm'`). |
| `model_param` | `dict` | *Obligatory* | Physical parameters for the chosen model. |
| `save` | `bool` | `False` | Whether to save the generated lightcurve to a file. |
| `filename` | `str` | `'test_lc_lsst.dat'` | Output filename if `save=True`. |
| `svd_path` | `str` | *Custom* | Path to SVD models (used if model is not `Bu2026_MLP`). |

*Returns:* A pandas DataFrame representing the fully formatted synthetic lightcurve ready for NMMA.

This helper is the main LSST entry point. It accepts any model present in the registry above and dispatches either to the FIESTA surrogate path or to the SVD-based NMMA lightcurve model as needed.

**Example of use of LSST related functions**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # for plotting
from peket.kn_side.utils import get_lsst_observations, generate_synth_lc_lsst, find_valid_sources

df = get_lsst_observations("/path/to/lsst/baseline/simulation/baseline_v5.1.1_10yrs.db", full_df=True) # full df to get the whole 10yrs simulation
valid_sources = find_valid_sources(df, n_sources=3, duration=15, min_observations=2) # 

for source in valid_sources:
    print(f"Source at RA: {source['ra']}, Dec: {source['dec']}")
    model_param = {
        "luminosity_distance": 40, # Mpc
        "inclination_EM": 0.67, # radians
        "log10_mej_dyn": -2,
        "log10_mej_wind": -1,
        "KNphi": 30
        }
    full_lc = generate_synth_lc_lsst(source, save=0, model_name="Bu2019lm", model_param=model_param) # also supports Bu2026_MLP and Ka2017

    # plot the light curve
    
    # Convert ISO time strings to relative days before plotting
    lc_plot = full_lc.copy()
    lc_plot[0] = pd.to_datetime(lc_plot[0], errors="coerce")
    lc_plot = lc_plot.dropna(subset=[0])

    plt.figure(figsize=(10, 6))
    times = (lc_plot[0] - lc_plot[0].min()).dt.total_seconds() / 86400.0  # days since first point
    color_dic = {
        "sdssu": "blue",
        "ps1::g": "green",
        "ps1::r": "red",
        "ps1::i": "orange",
        "ps1::z": "purple",
        "ps1::y": "brown"
    }
    for filt in lc_plot[1].unique():
        mask = lc_plot[1] == filt
        y = lc_plot.loc[mask, 2].astype(float)
        yerr = lc_plot.loc[mask, 3].astype(float)

        # separate UL from the rest
        ul_mask = np.isinf(yerr)
        if ul_mask.any():
            filt_times = times.loc[mask]
            plt.scatter(
                filt_times.loc[ul_mask],
                y.loc[ul_mask],
                label=f"{filt} (UL)",
                marker="v",
                color=color_dic.get(filt, "black"),
            )
        if (~ul_mask).any():
            filt_times = times.loc[mask]
            plt.errorbar(
                filt_times.loc[~ul_mask],
                y.loc[~ul_mask],
                yerr=yerr.loc[~ul_mask],
                fmt="o",
                label=filt,
                color=color_dic.get(filt, "black"),
            )

    plt.gca().invert_yaxis()
    plt.xlabel("Time (days)")
    plt.ylabel("Magnitude")
    plt.title("Synthetic Light Curve from LSST Observations")
    plt.legend()
    plt.grid()
    plt.show()
```

#### Internal Helpers
* **`get_source_observations`**: Extracts specific LSST observations relevant to a source's spatial and temporal position.
* **`get_obs_mag_from_lsst`**: Computes the observed magnitude based on the 5-sigma depth using `rubin_sim.phot_utils`.
* **`build_dic`**: Utility to structure observation arrays and depths per filter.

### Synthetic Lightcurve Utils
#### `regenerate_lc_from_truth`
Regenerates a synthetic lightcurve from an existing truth file and saves it to a specified output directory.

This function acts as a wrapper around the core synthetic lightcurve generation functions (`generate_synth_lc_fiesta` and `generate_synth_lc_v2`). It extracts physical parameters directly from a `true{idx}.csv` file and format them appropriately for the chosen model. It is specifically designed to be compatible with the `kn-ts-loop` workflow, allowing you to easily rebuild a lightcurve if you want to test different observational conditions (cadence, noise, delay) on an existing set of physical parameters.

**Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `idx` | `int` / `str` | *Obligatory* | Index of the lightcurve to regenerate. Used for naming output files and creating subdirectories. |
| `truth_file` | `str` | *Obligatory* | Path to the `.csv` truth file containing the source parameters. |
| `out_dir` | `str` | *Obligatory* | Base directory where the regenerated lightcurve will be saved. A subdirectory `/{idx}/` will be created. |
| `model` | `str` | *Obligatory* | Name of the EM model to use (e.g., `'Bu2019lm'`, `'Ka2017'`, `'Bu2026_MLP'`). |
| `filters` | `list` of `str` | *Obligatory* | List of observation filters (e.g., `["ps1::g", "ps1::r"]`). |
| `cadence` | `float` | *Obligatory* | Number of observation points per day (e.g., `2.0` for one point every 12 hours). |
| `delay` | `float` | *Obligatory* | Delay in days before the first observation. |
| `noise_level` | `float` | *Obligatory* | Standard deviation of the Gaussian noise added to magnitudes. |
| `max_error_level` | `float` | `0.4` | Maximum error level added to the magnitude uncertainties. |
| `obs_duration` | `float` | `15` | Total duration of the synthetic observations in days. |
| `detection_limit_dict` | `dict` | `None` | Dictionary of detection limits per filter (e.g., `{'ps1::g': 24.7, 'ps1::r': 24.2}`). |
| `jitter` | `float` | `0.0` | Maximum time fluctuation (in days) added to sample times. |
| `svd_path` | `str` | *Custom* | Path to SVD models directory (ignored if model is `'Bu2026_MLP'`). |
| `ISOT_trigger` | `str` | `'2020-01-07T00:00:00.000'` | ISOT time of the GW trigger. |

*Returns:* A pandas DataFrame containing the regenerated synthetic lightcurve, formatted for NMMA. It also saves the `.dat` file and a copy of the truth `.csv` file in `{out_dir}/{idx}/`.

The same model registry applies here, which means the helper can be used with the historical `Bu2019lm` / `Ka2017` grids as well as the newer surrogate families.

**Example**
```python
from peket.kn_side.utils import regenerate_lc_from_truth
truth_file = "/path/to/yourgrid/17/true17.csv"
out_dir = "/path/to/your/out/directory" # will create a subdirectory for each idx, so no need to specify it here (to be compatible with ts-loop) (same strucutre as kn-make-grid but for only one lc)
filters = ["ps1::g", "ps1::r", "ps1::i", "ps1::z", "ps1::y"]
cadence = 0.5
delay = 0.3
noise_level = 0.2
detection_limit = {'ps1::g': 24.7, 'ps1::r': 24.2, 'ps1::i': 23.8, 'ps1::z': 23.2, 'ps1::y': 22.3}
ISOT = "2020-01-07T00:00:00.000"
for i, models in enumerate(["Bu2019lm", "Ka2017", "Bu2026_MLP"]): # regenerate a specific lc with the different models but the same input parameters (D_L, inclination, ejected mass, rest are model specific parameters)
    model = models
    lc = regenerate_lc_from_truth(i, truth_file, out_dir, model, filters, cadence, delay, noise_level, max_error_level=0.4, obs_duration=10, detection_limit_dict=detection_limit, jitter=0.0, svd_path="/path/to/your/svdmodels", ISOT_trigger=ISOT)
```
