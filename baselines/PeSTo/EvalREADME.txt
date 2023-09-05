# Evaluation of PeSTo

## Installation

```
conda create -n pestocus python=3.9 -y # python 3.9 to be supported by gemmi
conda activate pestocus
conda install pytorch=2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y # pytorch with CUDA support
conda install conda-forge::gemmi -y 
conda install conda-forge::tensorboard -y 
conda install numpy scipy pandas matplotlib scikit-learn conda-forge::tqdm -y
conda install anaconda::h5py -y 
```

## Predicting binding sites and computing scores.
```bash
python eval.py
```

## Analyzing and plotting results  

Open the notebook `plot.ipynb` in Jupyter Notebook and run all cells.
