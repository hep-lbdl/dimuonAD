## [paper title]

This repository contains all the scripts needed to generate the plots in the papers [link to upsilon paper] and [link to general paper]

## Data preparation
*Running the scripts in this section can be somewhat time-intensive. If using data at [this Zenodo], you can jump to the next section "Analysis preparation".*

Before analyzing CMS Open Data, it is necessary to *skim* events such that only those coming from validated luminosity blocks are analyzed. Instructions for how to skim the CMS NanoAOD Open Data are given on [this CMS Open Data site](https://opendata.cern.ch/docs/cms-getting-started-nanoaod). 

If using CMS Simulation, this skimming procedure does not need to be carried out. 


### Pull relevant features
Once the NanoAOD files have been skimmed, the script `01_process_skimmed_root_files.py` can be run to extract relevant features from the ROOT files and save them into smaller pickle arrays. The script currently extracts a small selection of muon, electron, and jet variables. Note that this script outputs a separate pickles for every analysis object and for every skimmed ROOT file. 

If using CMS Simulation, the script `01_process_unskimmed_root_files.py` should be used instead.

### Calculate additional features and compile
The notebook `02_process_nanoAOD_skimmed.ipynb` compiles all analysis objects and root files into a single pickle array. Loose event filters and selection cuts should be applied at this level (e.g. PFC ID criteria, number of PFCs, number of jets). This notebook can also be used to quickly visualize analysis object distributions.

## Analysis Preparation
