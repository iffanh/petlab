python3 src/create_ensemble.py data/Egg_CCS/cobyla_config_const_7.json
python3 src/run_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/cobyla_Egg_CCS_const_7.json
python3 src/extract_ensemble.py simulations/studies/cobyla_Egg_CCS_const_7.json 
python3 src/optimize_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/cobyla_Egg_CCS_const_7.json