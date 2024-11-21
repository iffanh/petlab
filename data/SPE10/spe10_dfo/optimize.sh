METHOD="DFTR"
CONFIG_FILE="data/SPE10/spe10_dfo/${METHOD}_SPE10_DFO.json" 
STUDY_PATH=simulations/studies/${METHOD}_IE_SPE10.json

SIMULATOR_PATH="/usr/bin/flow"


python3 src/create_ensemble.py ${CONFIG_FILE}
python3 src/run_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
python3 src/extract_ensemble.py ${STUDY_PATH}
python3 src/optimize_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}