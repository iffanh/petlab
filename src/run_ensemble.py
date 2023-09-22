import json
import sys
import os
import subprocess
import utils.utilities as u
from datetime import datetime
from pathlib import Path

STORAGE_DIR = './simulations/storage/'
STUDIES_DIR = './simulations/studies/'

def change_control(base_datafile_path, real_datafile_path, controls):
    
    with open(base_datafile_path, 'r') as file :
        filedata = file.read()
        
    # optimization
    for control in controls:
        Name = control["Name"]
        Default = control["Default"]
        if control['type'] == "float":
            replaced_value = '%.3f '%Default
        elif control['type'] == "int":
            replaced_value = '%s '%int(Default)
            
        filedata = filedata.replace(Name, replaced_value)
        
    # Write the file out again
    with open(real_datafile_path, 'w') as file:
        file.write(filedata)
    
    return

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
        
    study = u.read_json(study_path)
    
    config = u.read_json(study['creation']['json'])
    
    _, tail = os.path.split(study['creation']['root']) # dir_path = /path/to/data
    root_name = os.path.splitext(tail)[0] #root_name = SPE1
    
    # create actual realization folders
    actual_path = os.path.join(STORAGE_DIR, study['Name'])
    Path(actual_path).mkdir(parents=True, exist_ok=True)
    
    base_realizations = study['creation']['base_realizations']

    realizations = {}
    for i, real_name in enumerate(base_realizations.keys()):
        
        real_name = root_name + '_%s'%(i+1) # SPE1_i
        
        real_path = os.path.join(actual_path, real_name) # /path/to/data/SPE1_i
        Path(real_path).mkdir(parents=True, exist_ok=True)

        real_datafile_path = os.path.join(real_path, real_name + '.DATA') # /path/to/data/SPE1_i/SPE1_i.DATA
    
        change_control(base_realizations[real_name], real_datafile_path, config['controls'])
        run_case(simulator_path, real_name, real_datafile_path)
        
        realizations[real_name] = real_datafile_path
        
    return realizations
        
def main(argv):

    simulator_path = argv[0]
    study_path = argv[1]

    if not os.path.isfile(study_path):
        raise ValueError("%s cannot be found" %study_path)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))
    realizations = run_cases(simulator_path, study_path)
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    
    
    studies = u.read_json(study_path)
    studies["status"] = "simulated"
    studies["simulation"] = {}
    studies["simulation"]["start"] = dt_start
    studies["simulation"]["end"] = dt_end
    studies["simulation"]["realizations"] = realizations
    
    
    studies["extraction"] = {} # make sure the data extraction is up-to-date
    
    u.save_to_json(study_path, studies)
    

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex: "python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_Poro.json"
    """
    main(sys.argv[1:])