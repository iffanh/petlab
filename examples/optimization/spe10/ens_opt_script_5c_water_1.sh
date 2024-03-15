METHOD="DFTR"

CONFIG_FILE="data/SPE10/spe10_layer21_ensemble/SPE10_5C_WATER_${METHOD}.json"
STUDY_PATH="simulations/studies/IE_SPE10_L21_5C_WATER_ENS_${METHOD}.json"

SIMULATOR_PATH="/usr/bin/flow"

python3 src/create_ensemble.py ${CONFIG_FILE}
python3 src/run_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
python3 src/extract_ensemble.py ${STUDY_PATH}
python3 src/optimize_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}