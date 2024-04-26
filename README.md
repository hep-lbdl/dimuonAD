# To process nanoAOD

1. Download the CMS ROOT docker container: https://opendata.cern.ch/docs/cms-guide-docker#nanoaod. (For NERSC users: `conda activate cernroot` is sufficient.)

2. Install this repo (standalone version): https://github.com/cms-opendata-analyses/nanoAOD-tools/. if installed, run `source standalone/env_standalone.sh`

3. Skim for the valid luminosity runs. 

In the `NanoAODTools` repository, run: `python run_skim.py -i 2` with `i` being the number of root files to process (takes a long time).