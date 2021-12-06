import json
import sys
import os
import subprocess

STORAGE_DIR = '/mnt/c/Users/iffan/Documents/Github/petlab/storage/'
STUDIES_DIR = '/mnt/c/Users/iffan/Documents/Github/petlab/studies/'

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

    realizations = data['realizations']

    for real_name in realizations.keys():
        run_case(simulator_path, real_name, realizations[real_name])

def main(argv):

    simulator_path = argv[0]
    study_path = argv[1]

    if not os.path.isfile(study_path):
        raise ValueError("%s cannot be found" %study_path)

    run_cases(simulator_path, study_path)

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    """
    main(sys.argv[1:])