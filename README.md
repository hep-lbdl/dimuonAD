## [paper title]

This repository contains all the scripts needed to generate the plots in the papers [link to upsilon paper] and [link to general paper]

The analysis is based on the [DoubleMuon primary dataset from RunH of 2016](https://opendata.cern.ch/record/30555).

## Data preparation
At **this zenodo link**, we provide a selection of muon and jet variables from the DoubleMuon Primary Dataset . All events in the Zenodo come from validated luminosity runs. 

Muon variables: `Muon_pt`, `Muon_eta`, `Muon_phi`, `Muon_charge`, `Muon_pfRelIso03_all`, `Muon_pfRelIso04_all`, `Muon_tightId`, `Muon_jetIdx`, `Muon_ip3d`, `Muon_jetRelIso`, `Muon_dxy`, `Muon_dz`
 
Jet variables: `Jet_pt`, `Jet_eta`, `Jet_phi`, `Jet_mass`, `Jet_nConstituents`, `Jet_btagCSVV2`, `Jet_btagDeepB`, `Jet_btagDeepFlavB`, `MET_pt`, `MET_sumEt`, `PV_npvsGood`, `Jet_nMuons`, `Jet_qgl`, `Jet_muEF`, `Jet_chHEF`, `Jet_chEmEF`, `Jet_neEmEF`, `Jet_neHEF`

In addition, we store the triggering information for all 40 triggers listed on the Open Data record page. 
    
The events are split into (28*2) files -- for both the muon and the analysis objects, there are 28 files corresponding to the 28 `ROOT` files in the CMS Open Data record for the dataset. 

To acquire the data, run the following commands
```
mkdir <data_storage_dir>/precompiled_data/skimmed_data_2016H_30555
cd <data_storage_dir>/precompiled_data/skimmed_data_2016H_30555
TODO
```
`<data_storage_dir>` should be added into the relevant line of `workflow.yaml`.

### Manual download and skimming
    
If you would like to use other event variables, you will need to skim the root files from the Open Data record yourself. Instructions for how to skim the CMS NanoAOD Open Data are given in [this tutorial](https://opendata.cern.ch/docs/cms-getting-started-nanoaod). If you decide to manually skim the files, you may find the script `00_process_skimmed_root_files.py` helpful as a starting point to extract relevant features from the skimmed `ROOT` files and save them into smaller pickle arrays, identical to the format on the official Zenodo.

### Calculate additional features and compile
The script `01_concatenate_and_filter_data.py` compiles all analysis objects and root files into a single pickle array, after filtering events with at least 2 muons that pass the `Muon_tightId` criteria. In addition, a number of dimuon observables are calculated and saved out.

Other loose event filters and selection cuts can be applied at this level (e.g. PFC ID criteria, number of PFCs, number of jets, triggering).

 `02_visualize_data.ipynb` is a sample notebook to quickly visualize features with and without cuts.

## Analysis preparation
Next, analysis-dependent cuts and modifications need to be applied to the data. These may involve:

- choosing signal region(s) (SR) and sideband region(s) (SB) (and therefore choosing a specific resonance to analyze)
- applying specific observable cuts (such as the anti-isolation cut for the $\Upsilon$ study (cite))
- applying additional event filters (e.g. triggering)
- injecting a BSM signal for setting limits (cite).

In addition, a specific set of analysis features for the classical and ML studies can be specified. 

For the ML study in particular, the data must be further preprocessed before being fed into the CATHODE-inspired normalizing flow architecture. We first logit-transform all the features (except the dimuon invariant mass, which is standard-scaled), then min-max scale them to the range (0, 1). This transformation was found to be effective for the normalizing flow training. 

`03_preprocess_data_lowmass.ipynb` applies the cuts for a single choice of signal region (SR) and sidebands (SB).

At this point, it is helpful to specify a few `analysis_keywords` to identify the specific project / features of interest. Example keywords for the upsilon analysis can be found in `workflow.yaml.` 
- `name`: a high-level name for the analysis
- `particle`: used to define SR and SB definitions in the `workflow` file (see `window_definitions.particle`)
- `analysis_cuts`: various lower and / or upper bounds for variables pulled or calculated in `01_concatenate_and_filter_data.py`
- `dataset_id`: high-level name for the CMS Open Dataset

## ML study: network training


Once some version of notebook `03` has been run, use the script `04_train_cathode.py` to train the normalizing flow on the auxiliary features, conditioned on the invariant mass.

Helpful flags:
- `train_samesign`: if you want to train on samesign muon pairs, instead of opposite-sign pairs
- `bkg_fit_degree`: pick an odd number
- `num_bins_SR`: `int` for the number of bin *boundaries* in the SR

You can specify `epochs` and `batch_size` as an argument to the script. For larger flow architecture changes, create a new config file in the `configs/<your_config.yaml>` folder and pass the file with `-c your_config.yaml`.

To regenerate flow samples with a different choice of SR binning or or background polynomial fit, use the `-no_train` flag.

Once flow samples have been generated, check the flow performance in the SB with `05_eval_cathode.py`. This code trains a BDT to discriminate SB samples from SB data. The ROC AUC should be close to that of a random classifier (~0.5).

Finally, carry out the bump hunt with `06_run_bump_hunt.py`. 

Helpful flags:
- `train_samesign`: if you want to train on samesign muon pairs, instead of opposite-sign pairs
- `num_to_ensemble`: how many BDTs to train for a single pseudoexperiment
To change the BDT architecture, edit the `bdt_hyperparameters` sections of `workflow.yaml`.

Batch scripts for `bash` and `slurm` jobs can be make with `make_scripts.ipynb`

## Compilation
08

## Plotting
09
