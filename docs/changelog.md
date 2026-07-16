# Changelog

## Latest Updates 0.1.2

### Documentation and MkDocs refresh
* Reworked the online documentation navigation to expose the current CLI names, including `gw-setup-sig3` and `gw-setup-pipeline2`.
* Aligned the command-line pages with the code in `src/peket/gw_search` and `src/peket/kn_side`.
* Expanded the utilities tutorial so it reflects the live model registry in `peket.kn_side.config`.

### GW pipeline updates
* Added the `gw-setup-sig` HTCondor DAG builder for significance estimation with preparation, background search, and post-processing stages.
* Added the newer `gw-setup-pipeline2` and `gw-setup-sig3` workflows with multi-detector coincidence handling, long-slide planning, and target-significance tuning.
* Refined the pipeline options for injection runs, antenna-pattern inspection, background chunking, and fit-based FAR estimation.

### kn_side utilities and model registry
* Remodeled the lightcurve generation in one function (still uses `nmma` and `fiestEM`).
* Added regeneration helpers for timeshift-loop studies and per-true-event output materialization.
* Expanded the supported model registry to all the `fiestaEM` surrogates.

## Latest Updates 0.1.1

### Documentation Overhaul
* Completely restructured the documentation using **MkDocs** and the **Material for MkDocs** theme. 
* Separated CLI references, installation guides, and tutorials into dedicated, easily navigable pages.

### gw-search-significance (New CLI)
* Added a new command-line tool to estimate the False Alarm Rate (FAR) and p-value for the top trigger.
* Automatically generates time slides across off-source windows to compute the background distribution.
* Features automatic HTCondor job submission, live progress monitoring, and automatic generation of timeline and FAR vs SNR plots.

### kn-ts-loop
* Added a ``--restrict-dist-prior`` argument to somewhat fix the distance for the ts-loop. Reasonnable if we have a redshift estimate for the candidate KN.

### kn_side.utils (LSST Integration)
* Introduced comprehensive utilities to generate synthetic kilonova lightcurves simulating Vera C. Rubin Observatory (LSST) observations.
* Integrated `rubin_sim` to fetch realistic cadences, filters, and 5-sigma depths from LSST databases.
* Added functions to ensure simulated sources fall within the realistic Field of View (FOV) and explode during expected lifetimes.
    * Added a `seed` argument to get reproducible sources
* Automatically calculates observed magnitudes with realistic Gaussian noise and upper limits handling.
    * Corrected the throughputs dir/files to use panStarrs (and sdss for u filter) to be coherent with NMMA inference.

### Others
* Minor correction to other CLIs (cleaning of unused lines).
* Solved non multiple of 5 axis problem with plotting utils.
* Added a function to regenerate a specific lightcurve from a grid.
* Added a function to duplicate a full grid.
* Can't add a `seed` to `kn-make-grid` to to qmc Halton.
---

## Version 0.1.0
Found a name and reorganized the structure. Beta version available on the `package` branch of the old [repo](https://github.com/NJamsin/KN_GW-BNS-master/tree/package).

### gw-setup-pipeline
* Completely changed the pre-treatment of the `.gwf` files. Now uses cache files (`.lcf`) -> reduces the computing time.
* Modified the `--monitor` argument to correctly print errors.
* Added a `--detector-threshold` argument for injection to early-stop the search if the detector response is too low with respect to the sky position.

### kn-make-grid
* Added supports for `Bu2019lm` and `Ka2017`.
* Corrected a small mistake that could lead to non-physical parameter values.

### kn-ts-loop
* Added a new command-line tool to iteratively perform inference by progressively removing early detection points.
* Supports optional upper limits substitutions and automatic GW resampling and corner plotting.
* Enhanced robustness of the timeshift loop to avoid skipping resampling in case of early-stop.