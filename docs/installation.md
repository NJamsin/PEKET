# Installation Guide

## 1. Prerequisites
Before installing PEKET, ensure you have the following dependencies correctly set up in your environment:

!!! warning "HTCondor Requirement"
    The GW DAG builders (`gw-setup-pipeline`, `gw-setup-pipeline2`, `gw-setup-sig`, and `gw-setup-sig3`) heavily rely on [HTCondor](https://htcondor.readthedocs.io/en/latest/) to create multiples `.sub` or `.dag` files. Make sure HTCondor is installed on your cluster/device.
    
    *(Note: HTCondor is not required if you only plan to use `kn-make-grid` and `kn-ts-loop`).*

!!! warning "Important: Python Version"
    Build [FIESTA](https://github.com/nuclear-multimessenger-astronomy/fiestaEM/tree/main) and [NMMA](https://github.com/nuclear-multimessenger-astronomy/nmma) from source in the same environment. 
    **NMMA explicitly requires Python 3.12.**

!!! tip "LSST Utilities (Required for the `kn_side`)"
    If you plane to use the KN related cli or utils or the LSST synthetic lightcurve generation utilities (`generate_synth_lc_lsst`), you will need the Rubin Observatory simulation framework. It is highly recommended to install it via `conda-forge` to gracefully handle its complex dependencies (`rubin_sim` is not in the requirements **BUT** is necessary):
    ```bash
    conda install -c conda-forge rubin_sim
    ```
    You also need to run
    ```bash
    rs_download_data -d throughputs
    ```
    To get the throughputs files of the filters needed for the LSST related functions.
---

## 2. Source Code Modifications
!!! info "Ensure compatibility"
    To ensure full compatibility with PEKET, minor modifications to NMMA and PyMultiNest are required. These modifications are minimal and will not alter the standard functioning of either package.

**NMMA Modification 1 (`nmma/em/model.py`, line 637):**
Allows generating Bu2019lm lightcurves with inclinationEM instead of KNtheta.
```python
def generate_lightcurve(self, sample_times, parameters, filters = 'all'):
        parameters = self.parameter_conversion(parameters) # <-- THIS ONE 
        parameters_list = self.em_parameter_setup(parameters)
```

**NMMA Modification 2 (`nmma/post_processing/plotting_routines.py`, line 160):**
Explicitly names the argument for corner plots.
```python
corner_plot(plot_samples.T, labels, limits, outdir=outdir) # <-- Explicitly name the argument
```

**PyMultiNest Modification (`pymultinest/analyse.py`, line 34):**
Prevents Fortran reading issues.
```python
def loadtxt2d(intext): 
    try:
        return numpy.loadtxt(intext, ndmin=2)
    except ValueError:
        # Catch Fortran formatting missing the 'e'
        with open(intext, 'r') as f:
            text = f.read()
        text = re.sub(r'(?<=\d)(-[1-3]\d\d)', r'e\1', text) 
        text = re.sub(r'(?<=\d)(\+[1-3]\d\d)', r'e\1', text)
        try:
            return numpy.loadtxt(StringIO(text), ndmin=2)
        except:
            return numpy.loadtxt(StringIO(text))
    except:
        return numpy.loadtxt(intext)
```

## 3. Install PEKET
Once prerequisites and modifications are done, clone and install the package:
```bash
git clone -b main git@github.com:NJamsin/peket.git
cd peket
pip install -e .
```