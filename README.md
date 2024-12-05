## [paper title]

This repository contains all the scripts needed to generate the plots in the papers [link to upsilon paper] and [link to general paper]

## Data preparation
*Running the scripts in this section can be somewhat time-intensive. If using data at [this Zenodo], you can jump to the next section "Analysis preparation".*

Before analyzing CMS Open Data, it is necessary to *skim* events such that only those coming from validated luminosity blocks are analyzed. Instructions for how to skim the CMS NanoAOD Open Data are given on [this CMS Open Data site](https://opendata.cern.ch/docs/cms-getting-started-nanoaod). 

If using CMS Simulation, this skimming procedure does not need to be carried out. 


### Pull relevant features
Once the NanoAOD files have been skimmed, the script `01_process_skimmed_root_files.py` can be run to extract relevant features from the ROOT files and save them into smaller pickle arrays. The script currently extracts a small selection of muon, electron, and jet variables. Note that this script outputs a separate pickle for every analysis object and for every skimmed ROOT file. 

If using CMS Simulation, the script `01_process_unskimmed_root_files.py` should be used instead.

### Calculate additional features and compile
The notebook `02_process_nanoAOD_skimmed.ipynb` compiles all analysis objects and root files into a single pickle array. Loose event filters and selection cuts should be applied at this level (e.g. PFC ID criteria, number of PFCs, number of jets, triggering).

 `03_visualize_data` is a sample notebook to quickly visualize features with and without cuts.


## Analysis preparation
Next, analysis-dependent cuts and modifications need to be applied to the data. These may involve:

- choosing signal region(s) (SR) and sideband region(s) (SB) (and therefore choosing a specific resonance to analyze)
- applying specific observable cuts (such as the anti-isolation cut for the $\Upsilon$ study (cite))
- applying additional event filters (e.g. triggering)
- injecting a BSM signal for setting limits (cite).

In addition, a smaller set of analysis features (e.g. the ``auxiliary features") can be specified. 

For the ML study in particular, the data must be further preprocessed before being fed into the CATHODE-inspired normalizing flow architecture. We first logit-transform all the features (except the dimuon invariant mass, which is standard-scaled), then min-max scale them to the range (0, 1). This transformation was found to be effective for the normalizing flow training. 

In `04_preprocess_data_lowmass`, we provide a notebook that applies the cuts for a single choice of signal region (SR) and sidebands (SB).

At this point, it is helpful to specify a few keywords to identify the specific project / features of interest. **(should these be added to the workflow??)**
- `run_id`:
- `project_id`:
- `particle_id`:
- `analysis_test_id`:
- `feature_id`:
- `feature_list`:

## ML study: network training

Once some version of notebook `04` has been run, use the script `05_trueCATHODE.py` to train the normalizing flow on the auxiliary features, conditioned on the invariant mass.

Helpful keywords:
- `train_samesign`
- `train_jet`
- `bkg_fit_type`
- `num_bins_SR`
- `feats`

You can specify `epochs` and `batch_size` as an argument to the script. For larger flow architecture changes, create a new config file in the `configs/<your_config.yml>` folder and pass the file with `-c your_config.yml`.

To regenerate flow samples with a different choice of SR binnins or background polynomial fit, use the `-no_train` flag.

Check the flow performance in the SB with `06_eval_CATHODE.py`

`07_bump_hunt_boostrap.py`

## Compilation
08

## Plotting
09
