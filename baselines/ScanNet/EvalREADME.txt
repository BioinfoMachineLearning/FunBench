# Evaluation of ScanNet

## Installation
```bash
conda create -n py_scannet python=3.6.12
conda activate py_scannet
pip install -r requirements.txt
pip install tensorflow-gpu==1.14

conda install -c conda-forge -c bioconda hhsuite 
mkdir UniRef30_2023_02
wget https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz -P UniRef30_2023_02
tar -xvf UniRef30_2023_02/UniRef30_2023_02_hhsuite.tar.gz -C UniRef30_2023_02

```

## Setup
set up the following paths in `utilities/paths.py`:
- `library_folder`
- path2hhblits
- path2sequence_database

## Evaluation 

```bash
python eval.py 
```
This script will first run ScanNet on given test set and then compute the scores against true labels.

Predicted binding results are saved in `predictions'.


Create a new conda environment for computing the scores:
```bash
conda create -n eval_env python=3.9
pip install torch torcheval pandas numpy 
```
Computing the scores and save them in csv files:
```bash
py compute_metrics.py --predictions_folder ./predictions --output_csv ./scan_net_noMSA_metrics.csv
py compute_metrics.py --predictions_folder ./predictions_msa --output_csv ./scan_net_MSA_metrics.csv
```