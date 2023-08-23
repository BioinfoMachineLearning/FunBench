# Evaluation of GraphBind

## Installation

Running the install script to install conda env, required packages, and download binary tools blast and hhsuite.
```
bash install.sh
```
### Download Database

GraphBind requires a database *nr* for blast, please download it by running the following command:

```
ncbi-blast-2.14.0+/bin/update_blastdb.pl --decompress nr
```

And then download Uniref 
```
wget https://gwdu111.gwdg.de/~compbiol/uniclust/2023_02/UniRef30_2023_02_hhsuite.tar.gz
tar xzf UniRef30_2023_02_hhsuite.tar.gz

```

## Setup 

Set the absolute paths of HHblits and uniclust databases in the script "scripts/prediction.py".

## Evaluation

```
cd scripts
python prediction.py --querypath ../output/example --filename 6ama.pdb \
--chainid A --ligands DNA,RNA,CA,MG,MN,ATP,HEME --cpu 10
```