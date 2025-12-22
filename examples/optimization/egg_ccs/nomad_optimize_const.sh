python3 src/create_ensemble.py data/Egg_CCS/nomad_config_const.json
python3 src/run_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/nomad_Egg_CCS_const.json
python3 src/extract_ensemble.py simulations/studies/nomad_Egg_CCS_const.json 
python3 src/optimize_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/nomad_Egg_CCS_const.json