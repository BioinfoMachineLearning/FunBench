

#### PDB sequences
wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -O data/pdb_seqres.txt.gz
# extract pdb_seqres
gunzip -f data/pdb_seqres.txt.gz > data/pdb_seqres.txt

##### PDB data

# parameters
MIRRORDIR=data/all_biounits
LOGFILE=pdb_logs
SERVER=rsync.ebi.ac.uk::pub/databases/rcsb/pdb-remediated
PORT=873
FTPPATH=/data/biounit/PDB/divided/

# download
rsync -rlpt -v -z --delete --port=$PORT ${SERVER}${FTPPATH} $MIRRORDIR > $LOGFILE 2>/dev/null
