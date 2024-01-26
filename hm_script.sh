
# python3 src/create_ensemble.py data/SPE1_data/SPE1_PG8607.json
# python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_PG8607.json
# python3 src/extract_ensemble.py simulations/studies/IE_PG8607.json

python3 src/hm_ensemble.py simulations/studies/IE_PG8607.json
# python3 src/hm_ensemble.py simulations/studies/HM_PoroPerm2_RF.json

python3 src/create_ensemble.py data/SPE1_data/SPE1_PG8607_POST.json
python3 src/run_ensemble.py /usr/bin/flow simulations/studies/HM_PG8607.json
python3 src/extract_ensemble.py simulations/studies/HM_PG8607.json
