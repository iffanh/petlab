
python3 src/create_ensemble.py data/SPE1_data/SPE1_PoroPerm2_SV.json
python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_PoroPerm2_SV.json
python3 src/extract_ensemble.py simulations/studies/IE_PoroPerm2_SV.json 