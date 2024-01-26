python3 src/create_ensemble.py data/Egg/Egg_4Controls.json
python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_Egg_4Controls.json
python3 src/extract_ensemble.py simulations/studies/IE_Egg_4Controls.json 
python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_Egg_4Controls.json