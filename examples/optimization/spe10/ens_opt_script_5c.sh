METHOD="COBYQA"

for i in $(seq 1 5)
do
    echo "STUDY ${i}"
    CONFIG_FILE="data/SPE10/spe10_layer21_ensemble/SPE10_5C_${METHOD}_$i.json"
    STUDY_PATH="simulations/studies/IE_SPE10_L21_5C_ENS_${METHOD}_$i.json"

    SIMULATOR_PATH="/usr/bin/flow"

    python3 src/create_ensemble.py ${CONFIG_FILE}
    python3 src/run_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
    python3 src/extract_ensemble.py ${STUDY_PATH}
    python3 src/optimize_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
done

for i in $(seq 1 5)
# for i in 1 {3...10}
do
    echo "STUDY ${i}"
    CONFIG_FILE="data/SPE10/spe10_layer21_ensemble/SPE10_5C_WATER_${METHOD}_$i.json"
    STUDY_PATH="simulations/studies/IE_SPE10_L21_5C_WATER_ENS_${METHOD}_$i.json"

    SIMULATOR_PATH="/usr/bin/flow"

    python3 src/create_ensemble.py ${CONFIG_FILE}
    python3 src/run_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
    python3 src/extract_ensemble.py ${STUDY_PATH}
    python3 src/optimize_ensemble.py ${SIMULATOR_PATH} ${STUDY_PATH}
done