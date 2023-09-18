import json
import sys
import os
import subprocess
import utils.utilities as u
from datetime import datetime

STORAGE_DIR = './simulations/storage/'
STUDIES_DIR = './simulations/studies/'

def run_case(simulator_path, real_name, real_path):

    command = [simulator_path, real_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)

    out = process.stdout

    result = process.wait()

    if result:
        raise RuntimeError("%s failed to run" %real_path)
    else: 
        print("%s ran successfully" %real_name)

def run_cases(simulator_path, study_path):

    with open(study_path, 'r') as f:
        data = json.load(f)

    realizations = data['creation']['realizations']

    for real_name in realizations.keys():
        run_case(simulator_path, real_name, realizations[real_name])

def main(argv):

    simulator_path = argv[0]
    study_path = argv[1]

    if not os.path.isfile(study_path):
        raise ValueError("%s cannot be found" %study_path)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))
    run_cases(simulator_path, study_path)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    
    
    studies = u.read_json(study_path)
    realizations = studies["creation"]["realizations"]
    
    studies = u.read_json(study_path)
    studies["status"] = "simulated"
    studies["simulation"] = {}
    studies["simulation"]["start"] = dt_start
    studies["simulation"]["end"] = dt_end
    
    studies["extraction"] = {} # make sure the data extraction is up-to-date
    
    u.save_to_json(study_path, studies)
    

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex: "python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_Poro.json"
    """
    main(sys.argv[1:])