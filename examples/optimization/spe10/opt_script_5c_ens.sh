python3 src/create_ensemble.py data/SPE10/spe10_layer21_ensemble/SPE10_5C.json
python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_SPE10_L21_5C_ENS.json
python3 src/extract_ensemble.py simulations/studies/IE_SPE10_L21_5C_ENS.json
python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_SPE10_L21_5C_ENS.json