## Test Set Construction

```bash
# install python packages
bash install.sh
# download pdb data
bash download_data.sh
# generate test data by date cutoff and sequence identity 
python process_seq.py
# generate binding site labels
python data_handler.py
```