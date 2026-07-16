# gw-setup-pipeline

This command-line tool performs coherent GW searches (using `pycbc_multi_inspiral` as its backend) based on properties inferred from KNe.

The base workflow is still available as `gw-setup-pipeline`, but the current codebase also ships `gw-setup-pipeline2`, which is the more feature-complete variant. It adds multi-detector coincidence control and tighter control over how search jobs are chunked.

## Usage
``gw-setup-pipeline`` requires one **OBLIGATORY** argument: the path to the ``.yaml`` config file.

```bash
gw-setup-pipeline path/to/config.yaml [OPTIONS]
```

## Configuration File Structure (.yaml)
The structure of the config file should be as follows:
```yaml
Directory:
  BASE_DIR: # Directory created for the pipeline. Ideally use one directory per run.
  run_name: # A unique name for this run, used for naming outputs and logs.

KN_data:
  first_detection: # ISOT format (yyyy-mm-ddThh:mm:ss.). Ensure H1/L1 were taking data!
  ra: # Right Ascension in radians
  dec: # Declination in radians
  EM_post_file: # Path to EM posterior samples (requires a timeshift column).
  RESAMP_post_file: # Path to RESAMP posterior samples (requires chirp_mass and mass_ratio).

GW_search:
  num_splits: # Number of splits for the template bank
  window_size: # Size of the max time window in seconds

Injection: # Only read if --injection is passed.
  mass1: # in solar mass
  mass2: # in solar mass
  distance: # Distance in Mpc
  ra: # Right Ascension (radians)
  dec: # Declination (radians)
  polarization: # Polarization angle
  approximant: # Waveform approximant (str)
  time_offset: # Seconds after the middle of the window for the merger.
```

## Optional Arguments
### Execution & Monitoring:
- ``--submit``: Automatically submit the pipeline to HTCondor after generation.
- ``--skip-search``: Skips the search step and runs post-processing (requires existing triggers).
- ``--monitor``: If true, will monitor the pipeline execution. (ignored if ``--skip-search`` is used).
- ``--ldg-tag``: The ``accounting_group`` tag required for submitting to IGWN/LDG clusters (e.g., ``ligo.sim.o3.cbc.bns.pycbcoffline``).
### Search & Template Parameters:
- ``--OSW-sigma``: Size of the On-Source Window around the expected trigger time, in sigmas (Choices: ``1``, ``2``, ``3``, ``full``. Default: ``1``)
- ``--tmplt-sigma``: Size of the posterior bounds used for template bank generation, in sigmas (Choices: ``1``, ``2``, ``3``, ``full``. Default: ``1``)
- ``--template-bank``: Path to the template bank file if you want to specify it instead of generating through the resampling posterior.
- ``--detector-threshold``: Minimum antenna response to launch the search (default: ``0.5``). Only applied to injections.
- ``--disk``: Amount of disk space to request for the prep job (default: ``3GB``)
### Injection & Plotting:
- ``--injection``: Injects a fake signal based on the 'Injection' section of the config.
- ``--expected-trigger-time``: Expected trigger time in gps format. Used in final plots.
- ``--plot-spectrogram``: Generates a spectrogram plot for the top trigger.
  - ``--spectrogram-range``: vmin and vmax for the spectrogram plot. Only used if ``--plot-spectrogram`` is set, default values are ``vmin=0, vmax=15``.
- ``--plot-antenna-pattern``: Generates an antenna pattern plot for the source location (injections only).

## gw-setup-pipeline2

`gw-setup-pipeline2` reuses the same config file structure as the base pipeline, but the prep step now accepts detector-coincidence controls and job-sizing parameters that are wired into the DAG pre-computation.

### Additional Arguments
- ``--detectors``: Comma-separated list of candidate detectors to consider for the search (default: ``H1,L1,V1``).
- ``--min-ifos``: Minimum number of detectors simultaneously in science mode required to analyze a time segment.
- ``--dq-flag``: GWOSC timeline flag suffix used to build per-detector segment lists.
- ``--segment-margin``: Seconds trimmed off both ends of each science segment before coincidence is computed.
- ``--min-analysis-length``: Minimum usable coincident chunk length submitted as a job.
- ``--max-extension``: Extra off-source extension used by the newer coincidence-aware setup.
- ``--segment-length`` / ``--slide-shift`` / ``--n-ifos``: Parameters forwarded to the multi-detector chunking logic.
- ``--target-sigma``: Automatically resolves the segment-length / slide-shift setup to reach a target background significance.
- ``--num-longslides`` / ``--longslide-step`` / ``--longslide-margin``: Controls the long-slide background strategy used by the newer DAG generator.

### Known Issues
It is possible that an OOM (Out Of Memory) error kills the search jobs despite the 4GB of RAM requested for each job. Rerunning the command usually solves the problem. The DAG is configured to retry failed search jobs up to 3 times automatically (`RETRY SEARCH 3`).
