from tqdm import tqdm
import sys
import os
import subprocess
try:
    from .utils import utilities as u
except ImportError:
    import utils.utilities as u 
from datetime import datetime
from pathlib import Path

import ecl.eclfile 

import time
from wrapt_timeout_decorator import *

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

def simulate_case(simulator_path, real_name, real_path):
    if "flow" in simulator_path:
        command = [simulator_path, '--enable-terminal-output=false', real_path]
    elif "ecl" in simulator_path:
        command = [simulator_path, real_path[:-5]] #does not need the .DATA format
    
    return command
    
def run_case(base_datafile_path, real_datafile_path, controls, simulator_path, real_name):
    change_control(base_datafile_path, real_datafile_path, controls)
    command = simulate_case(simulator_path, real_name, real_datafile_path)
    return command

@timeout(3000) #10 min timeout # 1 hour timeout
def run_cases(simulator_path, study, simfolder_path, controls, n_parallel):
    
    _, tail = os.path.split(study['creation']['root']) # dir_path = /path/to/data
    root_name = os.path.splitext(tail)[0] #root_name = SPE1
    
    base_realizations = study['creation']['base_realizations']

    commands = []
    realizations = {}
    l = len(base_realizations)
    for i, real_name in tqdm(enumerate(base_realizations.keys()), total=l, desc="Preparing: ", leave=False):
        
        real_name = root_name + '_%s'%(i+1) # SPE1_i
        
        real_path = os.path.join(simfolder_path, real_name) # /path/to/data/SPE1_i
        Path(real_path).mkdir(parents=True, exist_ok=True)

        real_datafile_path = os.path.join(real_path, real_name + '.DATA') # /path/to/data/SPE1_i/SPE1_i.DATA
        base_datafile_path = base_realizations[real_name]
        command = run_case(base_datafile_path, real_datafile_path, controls, simulator_path, real_name)
        commands.append(command)
        realizations[real_name] = real_datafile_path
    
    is_success = u.run_bash_commands_in_parallel(commands, max_tries=1, n_parallel=n_parallel)
    
    # post process to know whether the simulation ends successfully or not
    # When a simulation fails, flow reports error, while E300 might not
    # So we check the timesteps generated instead from .UNRST
    for i, real_name in tqdm(enumerate(base_realizations.keys()), total=l, desc="Preparing: ", leave=False):
        real_name = root_name + '_%s'%(i+1) # SPE1_i
        
        real_path = os.path.join(simfolder_path, real_name) # /path/to/data/SPE1_i
        real_unrst_path = os.path.join(real_path, real_name + '.UNRST') # /path/to/data/SPE1_i/SPE1_i.DATA
        
        file = ecl.eclfile.EclFile(real_unrst_path)
        
        if len(file.report_steps) == 1:
            is_success[i] = False
            
    print(is_success)
    
    return realizations, is_success
        
def main(argv):

    simulator_path = argv[0]
    study_path = argv[1]

    if not os.path.isfile(study_path):
        raise ValueError("%s cannot be found" %study_path)

    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))
    
    study = u.read_json(study_path)
    
    # create actual realization folders
    simfolder_path = os.path.join(STORAGE_DIR, study['Name'])
    Path(simfolder_path).mkdir(parents=True, exist_ok=True)
    
    config = u.read_json(study['creation']['json'])
    realizations, is_success = run_cases(simulator_path, study, simfolder_path, config['controls'], n_parallel=config['n_parallel'])
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    
    
    studies = u.read_json(study_path)
    studies["status"] = "simulated"
    studies["simulation"] = {}
    studies["simulation"]["start"] = dt_start
    studies["simulation"]["end"] = dt_end
    studies["simulation"]["realizations"] = realizations
    
    ens_path = os.path.join(STORAGE_DIR, studies['Name'])
    studies["simulation"]["storage"] = ens_path
    
    studies["extraction"] = {} # make sure the data extraction is up-to-date
    
    u.save_to_json(study_path, studies)
    

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex: "python3 src/run_ensemble.py /usr/bin/flow simulations/studies/IE_Poro.json"
    """
    main(sys.argv[1:])