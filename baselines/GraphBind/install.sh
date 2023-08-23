# Please following the order to install all packages! 

conda create -n GraphBind_env python=3.8 -y
source activate GraphBind_env


conda install pytorch-scatter pytorch-sparse pytorch-cluster  pyg -c pyg -y

conda install pytorch==1.12.1  cudatoolkit=11.3 -c pytorch -y
# pip install --no-cache-dir torch-scatter==1.3.1
# pip install --no-cache-dir torch-sparse==0.4.0
# pip install --no-cache-dir torch-cluster==1.4.4


pip install biopython torchnet==0.0.4 tqdm prettytable pandas==1.5.3

chmod +x scripts/dssp

wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.14.0+-x64-linux.tar.gz;\
tar -xzf ncbi-blast-2.14.0+-x64-linux.tar.gz

mkdir hhsuite
wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz;\
tar -xzf hhsuite-3.3.0-AVX2-Linux.tar.gz -C hhsuite
