python3 src/create_ensemble.py data/Egg_CCS/config.json
python3 src/run_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/Egg_CCS.json
python3 src/extract_ensemble.py simulations/studies/Egg_CCS.json 
python3 src/optimize_ensemble.py /mnt/c/ecl/2022.3/bin/pc_x86_64/e300.exe simulations/studies/Egg_CCS.json

# python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_Egg_4Controls.json
# python3 src/extract_ensemble.py simulations/studies/IE_Egg_4Controls.json 
# python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_Egg_4Controls.json