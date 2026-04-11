# PEKET
``peket``: **P**yCBC **E**xtension for **K**ilonova **E**mission **T**argeting

![logo](logo.png)

**NOTE**: ``peket`` is still under development. A basic documentation is available below in the markdown. Feel free to contact me for any questions.

## Add-on Package for PyCBC and NMMA/FIESTA
Contains a command line-tool ``gw-setup-pipeline`` that performs coherent GW search (using ``pycbc_multi_inspiral`` as its backend) based on properties inferred from KNe.

Another command line-tool, ``kn-make-grid`` is also available. This command-line generates a grid of lightcurve using the Bu2026_MLP model using FIESTA/NMMA (see below). 

A third command line-tool is available, ``kn-ts-loop`` to perform inference with NMMA, repeatedly removing the first detection point to see the evolution of the timeshift inference. 

## Changelog (0.1.0)
Found a name and reorganized the structure. ("Beta" version available [here](https://github.com/NJamsin/KN_GW-BNS-master/tree/package))
### ``gw-setup-pipeline``
Completely changed the pre-treatment of the .gwf files. Now uses cache files (.lcf) $\rightarrow$ reduces the computing time.

Modified the ``--monitor`` argument to correctly print errors. 

Added a ``--detector-threshold`` argument for injection to early-stop the search if the detector response is to low with respect to the sky position of the injected signal.

### ``kn-make-grid``
Added supports for ``Bu2019lm`` and ``Ka2017``. /!\ ``Bu2019lm`` requires a small modification to nmma's code (see below).

Corrected a small mistake that could lead to non-physical parameter values.

### ``kn-ts-loop``
Added a new command-line tool to iteratively perform inference on light curve data by progressively removing early detection points. This allows for the tracking of the timeshift evolution. Supports optional upper limits substitutions and automatic GW resampling and corner plotting via NMMA's ``gwem-resampling``. 

Enhance the robustness of the timeshift loop to avoid skipping the resampling in case of early-stop of the first loop.

### IMPORTANT REMARK BEFORE INSTALLATION!
``gw-setup-pipeline`` requires and uses [HTCondor](https://htcondor.readthedocs.io/en/latest/) a lot (creates multiples .sub or .dag files), make sure that you have HTCondor installed on your device. Dynamic slots are recommended as the main job submitted by ``gw-setup-pipeline`` reserves 16 Gb of RAM and each sub job that performs the coherent search reserves 4 Gb of RAM (a bit much, could be manually reduced if needed).

If you want to use ``kn-make-grid`` and ``kn-ts-loop`` make sure to build [FIESTA](https://github.com/nuclear-multimessenger-astronomy/fiestaEM/tree/main) and [NMMA](https://github.com/nuclear-multimessenger-astronomy/nmma) (/!\ NMMA requires python 3.12) from source in the same environment as this package and for this command, HTCondor is not required ! 

Moreover, for ``kn-make-grid``, a slight modification needs to be done to ``nmma/em/model.py`` at line 637.
You must add this line:
```python
def generate_lightcurve(self, sample_times, parameters, filters = 'all'):
        parameters = self.parameter_conversion(parameters) # <-- THIS ONE (allows to generate Bu2019lm lc with inclinationEM instead of KNtheta)
        parameters_list = self.em_parameter_setup(parameters)
```

And for ``kn-ts-loop``, two modification are required to ensure the good functioning of the command.

One to correct a small function of ``pymultinest/analyse.py`` around line 34:
```python
def loadtxt2d(intext): # <-- this function (Modification to avoid Fortran reading issue)
    try:
        return numpy.loadtxt(intext, ndmin=2)
    except ValueError:
        # Catch Fortran formatting missing the 'e'
        with open(intext, 'r') as f:
            text = f.read()
        text = re.sub(r'(?<\=\d)(-[1-3]\d\d)', r'e\1', text) 
        text = re.sub(r'(?<\=\d)(\+[1-3]\d\d)', r'e\1', text)
        try:
            return numpy.loadtxt(StringIO(text), ndmin=2)
        except:
            return numpy.loadtxt(StringIO(text))
    except:
        return numpy.loadtxt(intext)
```
And another one to ``nmma/post_processing/plotting_routines.py``, line 160:
```python
corner_plot(plot_samples.T, labels, limits, outdir=outdir) # <-- Explicitly name the argument
```

All these modifications are minimal and won't alter the functioning of either ``nmma`` nor ``pymultinest``.
***
# Installation
First, build ``FIESTA`` and ``NMMA`` from source.

Then simply clone the repo with
```
git clone -b main git@github.com:NJamsin/PEKET.git
```
Make sure to change your active directory to the cloned repo and execute
```
pip install -e .
```
***
# Documentation
## ``gw-setup-pipeline``
``gw-setup-pipeline`` has one **OBLIGATORY** argument, the path to the ``.yaml`` config file (examples of config files are provided in the [``example_file``](https://github.com/NJamsin/PEKET/tree/main/example_file) directory).
### Additional (optionnal) arguments:
- ``--submit``: Automatically submit the pipeline to HTCondor after generation.
- ``--injection``: If true, will inject a fake signal inside the time windows to be searched, for testing purposes. The injection parameters will be read from the config file (under the 'Injection' section).
- ``--expected-trigger-time``: Expected trigger time to be searched, in gps format (int or float). Used only in the final trigger distribution plot.
- ``--skip-search``: If true, will skip the search step and directly run the post-processing script. Only works if you already have triggers generated from a previous search run.
- ``--plot-spectrogram``: If true, will generate a spectrogram plot for the top trigger in the post-processing step. This can be useful for visually inspecting the trigger.
  - ``--spectrogram-range``: vmin and vmax for the spectrogram plot. Only used if ``--plot-spectrogram`` is set, default values are ``vmin=0, vmax=15``.
- ``--detector-threshold``: Minimum antenna response required to launch the search. Default is 0.5, can be useful to avoid long search for time windows where the detectors are barely sensitive to the source. Only applied to injections because the merger time is needed for the antenna pattern.
- ``--plot-antenna-pattern``: If true, will generate an antenna pattern plot for the source location and the injection merger time. **Only applied to injections** because the merger time is needed for the antenna response. /!\ The plot is generated at the end of the preparation so if the search is stopped by the threshold it won't be generated.
- ``--template-bank``: Path to the template bank file if you want to specify it instead of generating through the resampling posterior. This can be useful if you want to use a custom template bank or if you want to skip the template bank generation step.
- ``--monitor``: If true, will monitor the pipeline execution. /!\ Won't have any effect if you use ``--skip-search``.
### Example utilisation
Two examples are provided:
- One for real KN-GW data (AT2017gfo-GW170817). The output generated by the [config file](https://github.com/NJamsin/PEKET/blob/main/example_file/real_KN/example_170817config.yaml) is located in the [``test_real_data``](https://github.com/NJamsin/PEKET/tree/main/test_real_data) directory.
- One for an artificially generated KN (generated with [NMMA](https://nuclear-multimessenger-astronomy.github.io/nmma/index.html) using the model [``Bu2019lm``](https://arxiv.org/abs/2002.11355)) with an injected GW in real H1 L1 strain data. The output generated by the [config file](https://github.com/NJamsin/PEKET/blob/main/example_file/injection/example_injection.yaml) is located in the [``test_injection``](https://github.com/NJamsin/PEKET/tree/main/test_injection) directory.

The structure of the config file should be as follows:
```yaml
Directory:
  BASE_DIR: # Name of the directory created for the pipeline, all the data/logs/plots/out files will be placed there. Ideally use one directory per run to avoid confusion 
  run_name: # A unique name for this run, used for naming outputs and logs

KN_data:
  first_detection: # Time of the first detection in ISOT format (yyyy-mm-ddThh:mm:ss.) (str) !!!!! Make sure that H1 and/or L1 were actually taking data at this time, otherwise the pipeline won't find any data to analyze and will fail
  ra: # Right Ascension in radians
  dec: # Declination in radians
  EM_post_file: # Path to the EM posterior samples file as given by NMMA's lightcurve-analysis (should work for other inference pipeline as long as the file as a timeshift column)
  RESAMP_post_file: # Path to the RESAMP posterior samples file as given by NMMA's gwem-resampling should work for other fraemwork posterior files as long as it has a 'chirp_mass' and 'mass_ratio' column)

GW_search:
  num_splits: # Number of splits for the template bank
  window_size: # Size of the max time window in seconds

Injection: # only taken if --injection is passed as an argument to the command-line gw-setup-pipeline
  mass1: # in solar mass
  mass2: # in solar mass
  distance: # Distance in Megaparsecs (Mpc)
  ra: # Right Ascension (radians) ideally the same as the one in the KN_data section, but can be different for testing purposes
  dec: # Declination (radians) same as above
  polarization: # Polarization angle
  approximant: # Waveform approximant to use for the injection (str)
  time_offset: # How many seconds after the middle of your (global) search window should the merger happen? Make sure this is smaller than half the window_size defined in the GW_search section, otherwise the injection will be outside of the search window and won't be found by the pipeline
```
### Known issues
It is possible that an ``OOM`` error kills the search jobs despite the 4GB of RAM requested for each job. I am not sure why this happens but simply rerunning the command solve the problem most of the times.
***
## ``kn-make-grid``
``kn-make-grid`` produces a grid of pseudo randomly sampled synthetic kilonovae lightcurves generated via [NMMA](https://nuclear-multimessenger-astronomy.github.io/nmma/index.html)/[FIESTA](https://github.com/nuclear-multimessenger-astronomy/fiestaEM/tree/main). 

Currently supported models are ``Bu2026_MLP``, ``Bu2019lm`` and ``Ka2017``.
### Arguments:
- ``--out-dir``: Base directory for the output files.
- ``--model``: Model name to use for generating synthetic lightcurves. (default: Bu2026_MLP). Currently supports 'Bu2026_MLP', 'Bu2019lm' and 'Ka2017'.
- ``--num-lc``: Number of lightcurves to generate. Default is 25.
- ``--filters``: Filters used for the fake observations. Should be passed as ``--filters filt1 filt2 ...``. Filters supported are those supported by NMMA.
- ``--eos-path``: Path to the EOS file (.dat) for the fitting formulae. The structure of the file should contain ``mass  radius`` columns in solar mass and km respectively.
- ``--noise-level``: Value corresponds to the standard deviation of the Gaussian noise to be added to the model magnitude (noise=0.2 will add a np.random.normal(0, 0.2) to each model point). Default is 0.2.
- ``--max-error-level``: Value corresponds to the maximun error added to the model magnitude (max_error=0.4 will add an error np.random.uniform(0, 0.4) to each detections point). Default is 0.4.
- ``--trigger-isot``: Trigger time for the synthetic lightcurves. Default is '2020-01-07T00:00:00' (date is in O3b) (/!\ it is a str).
- ``--cadence``: Number of detection per day.
- ``--delay`` : Delay in days between the trigger and the first observation (important for inference as some model may not be defined at very early times).
- ``--obs-duration``: Observation duration in days for the synthetic lightcurves. Default is 7.
- ``--jitter``: Value corresponds to the maximum jitter (time fluctuation) to be added to the sample times in days (jitter=0.1 will add a np.random.uniform(-0.1, 0.1) to each time stamp in sample times). Default is 0.
- ``--detection-limit``: Detection limits for the synthetic data in the format: filter1:limit1 filter2:limit2 ... (e.g., --detect_limit ps1::g=24.7 ps1::r=24.2).
- ``--param-ranges``: Parameter ranges for the synthetic lightcurves in the format: param1=(min,max) param2=(min,max) ... (/!\ Params are the parameters of Bu2026_MLP).
-- ``save-json``: Whether to save the times and magnitudes for each sample in a json file. (json files will be saved in the same directory as the .dat files).

### Example utilisation
A grid made with the command is available in the [``example_file/KN_grid``](https://github.com/NJamsin/PEKET/tree/main/example_file/KN_grid). The exact command used was:
```bash
kn-make-grid --out-dir peket/example_file/KN_grid --filters ps1::r ps1::g ps1::z ps1::i ps1::y --eos-path peket/example_file/KN_grid/eos.dat --detection-limit ps1::r=24 ps1::g=20 ps1::z=30 --save-json --cadence 3 --delay 1 --jitter 0.25
```
***
## ``kn-ts-loop``
``kn-ts-loop`` performs iterative inference on synthetic or real kilonova lightcurve data. By progressively removing early detection points (and optionally substituting them with upper limits), it evaluates how the inferred timeshift and other EM parameters evolve as the first detection date increases with respect to the merger time. It can optionally run ``gwem-resampling`` immediately after the inference to extract source parameters (chirp mass, mass ratio, etc.) and automatically plots the resulting corner plots.

**NOTE**: The ``--resampling`` flag is not compatible with ``Ka2017``.

### Arguments:
- ``--idx``: Index of the injection/lightcurve to analyze (following the same structure as the lc generated with ``kn-make-grid``). Default is 0.
- ``--grid-dir``: Path to the base grid directory containing your data.
- ``--model``: EM model to use for the inference. Currently supports 'Bu2019lm', 'Ka2017' and 'Bu2026_MLP'. Default is 'Bu2019lm'.
- ``--svd-path``: Path to the SVD models directory (required if not using 'Bu2026_MLP').
- ``--prior-file``: Path to the prior file for the lightcurve analysis.
- ``--minus-pts``: Number of early time points to progressively remove for the inference iterations. Default is 2 (update as needed, up to the total number of time points - 1).
- ``--add-ul``: If passed, adds an upper limit instead of completely removing the early points. Default is False.
- ``--true-merger-time``: True merger time to use for the timeshift calculation in ISOT format. Default is '2020-01-07T00:00:00' (keep the same trigger time for all analyses to see how the timeshift evolves).
- ``--nlive``: Number of live points for the nested sampling (pymultinest). Default is 1024. /!\ Update prior bounds if needed.
- ``--resampling``: If passed, runs the resampling step via ``gwem-resampling`` after the EM inferences are done. Default is False.
- ``--eos-posterior``: Path to the EOS posterior probability file for the GW samples generation. Default is ``/home/stu_jamsin/jamsin/add_files/posterior_probability.txt`` (which is the same as this [file](https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/example_files/tools/gwem_resampling/posterior_probability.txt)).
- ``--eos-path``: Path to the EOS files used for the resampling.
- ``--GW-prior``: Path to the GW prior file for the resampling.
- ``--EM-prior``: Path to the EM prior file for the resampling.

### Example utilisation
To run an iterative inference removing the first 2 points on the injection index `0`, adding upper limits, and executing the resampling step at the end:
```bash
kn-ts-loop --idx 0 --grid-dir /path/to/KN_grid --model Bu2019lm --svd-path /path/to/svd --prior-file /path/to/EM.prior --minus-pts 2 --add-ul --resampling
```
The output (posteriors ``.dat`` files, config ``.yaml`` files, and ``.png`` corner plots) will be systematically organized inside your grid directory under ``{grid_dir}/{idx}/minusX/``, where ``X`` goes from 0 to the value of ``--minus-pts``.
***
## Plotting utils
``peket.kn_side.utils`` includes a function, ``plot_param_evolution`` that generates summary plots for the complete analysis of a grid with ``kn-ts-loop``/``lightcurve-analysis``. 
**NOTE**: A small tutorial on the combined use of ``kn-ts-loop``, ``kn-make-grid`` and ``plot_param_evolution`` is available in the [``ts-loop``](https://github.com/NJamsin/PEKET/tree/main/example_file/ts-loop) directory in the example files. It also contains example prior files for all supported models.
***
### Acknowledgements
I would like to acknowledge the use of [Gemini](https://gemini.google.com/app) for fixing code errors and assisting with the logo design.