This readme file describes Python code associated with the manuscript 'Causal networks for climate model evaluation and constrained projections' by Nowack et al.

FILE OVERVIEW - FILES CAN BE USED FOR TESTING PURPOSES/DEMO
---
generate_varimax_ncar_3dm.py

Python file to carry out the PCA Varimax analysis on NCAR/NCEP reanalysis data as described in the main paper.

geo_field_jakob.py

File containing an additional helper function needed in generate_varimax_ncar_3dm.py

generate_varimax_3dm_EC-EARTH-hist_ensemble_WEIGHTS-FROM-NCAR_jja.py

A file giving an example for how the components learned on NCAR/NCEP data can then be applied to climate model data, e.g. EC-EARTH. Here the file is set up for June, July, August components from NCAR/NCEP. 

run_pcmci_example.py

Example file for running PCMCI on the resulting PCA Varimax components.

We add four csv files with indices for the selected seasonal PCA Varimax components (DJF, MAM, JJA, SON) to the folder. Accordingly, file names need to be adapted in run_pcmci_example.py depending on the season of interest. These indices match Supplementary Table 3 (p. 27) in Supplementary file: 

https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15195-y/MediaObjects/41467_2020_15195_MOESM1_ESM.pdf

---
SOFTWARE DEPENDENCIES:
---
Python needs to be installed as well as the corresponding Python packages. Installation instructions for Tigramite for Python2.7 and Python3.6 can be found here: https://github.com/jakobrunge/tigramite

The calculations for the manuscript were run using Python2.7 but the Tigramite software is also compatible with Python3.6 

Tigramite is an implementation of the PCMCI causal discovery method, which has been published in: 

J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic,
           Detecting and quantifying causal associations in large nonlinear time
           series datasets. Sci. Adv. 5, eaau4996 (2019)
           https://advances.sciencemag.org/content/5/11/eaau4996

The software has been tested on Ubuntu 14.04, Ubtunu 16.04, Mac, Windows 10, and several UNIX-based computing clusters.

The typical installation time for Python and Tigramite should not exceed a couple of hours but depends on the system and internet connection, for example.
---
EXPECTED RUNTIME: The PCA Varimax analysis should not take longer than 1-2 hours to run for the full record of the NCAR/NCEP reanalysis data on a standard scientific laptop. The PCMCI analysis can take several hours per dataset equivalent to the length of the historic reanalysis record at a 3-day-average time resolution. If many datasets are analysed we recommend use of a high-throughput computing cluster.

TUTORIALS & INSTRUCTIONS FOR USE: tutorials for how to run the PCMCI methods using functions provided through Tigramite can be found here: https://github.com/jakobrunge/tigramite/tree/master/tutorials

Python files provided in this folder can be run from the command line using 'python2.7 filename.py'.

REPRODUCTION INSTRUCTIONS:

After installing the relevant software:

1) Extract sea level pressure data for the reanalyses and CMIP5 model data from the publicly available archives using the same time periods, ensemble members etc. Links to the data are provided in the main manuscript in the section 'Data availablility'.

2) Detrend the data using cdo operators and the command 'cdo detrend'.

3) Run the PCA Varimax code on the detrended NCAR/NCEP reanalysis data for the period 01/01/1948 to 31/12/2017. Replace variable 'load_filename' in function 'main'. Currently components are calculated for Northern Hemisphere winter months (December, January, February). For other months, change the variable 'month_mask'. Note the vriable 'varname' might also need to be changed depending on the variable name in the sea level pressure dataset.

4) Apply the learned transformation using the additonally provided code on all climate model data/other reanalysis datasets. For this the climate model data needs to be first interpolated to the same grid as the NCAR/NCEP reanalysis data. Apply the same pre-processing steps (detrending) on all datasets. See e.g. file generate_varimax_3dm_EC-EARTH-hist_ensemble_WEIGHTS-FROM-NCAR_jja.py

5) Use the resulting .bin files with the PCA Varimax components in the run_pcmci_example.py file one at a time. Either run individually on dataset (ensemble member), or adapt the time-slicing in the run_pcmci.py file. The results provide other .bin output files with the causal network data.

If question arise contact Peer Nowack.










