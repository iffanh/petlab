import numpy as np
import sys
import run_ensemble
import extract_ensemble
import optimize_ensemble

import os
import utils.utilities as u

from datetime import datetime
from pathlib import Path
import csv


STORAGE_DIR = './simulations/storage/'

def read_csv(path:str) -> list:
    
    with open(path) as f:
        reader = csv.reader(f)
        data = list(reader)
    
    
    # cleaning
    data = [float(d[0]) for d in data]
    
    return data

def main(args):
    
    simulator_path = args[0]
    study_path = args[1]
    controls_path = args[2] # csv text file

    study = u.read_json(study_path)
    config = u.read_json(study['creation']['json'])

    controls = config['controls']
    control_vals = read_csv(controls_path) # must be a list
    
    new_controls = []
    for c, cv in zip(controls, control_vals):
        Name = c['Name']
        # Default = c['Default']
        if c['type'] == "float":
            val = cv
        elif c['type'] == "int":
            val = int(cv)
            
        c['Default'] = val
        
        new_controls.append(c)
    
    print(new_controls)
    # change the controls
    
    
    simfolder_path = os.path.join(STORAGE_DIR, study['Name'])
    Path(simfolder_path).mkdir(parents=True, exist_ok=True)
    
    
    ### SIMULATION
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))
    realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, new_controls, config['n_parallel'])
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
    
    
    ### Extraction
    study = u.read_json(study_path)
    realizations = study["simulation"]["realizations"]
    storage = study["simulation"]["storage"]
    
    # fetch list of summaries to load
    sum_keys = study["creation"]["config"]["vectors"]["summary"]
    
    # check summary
    summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
    
    # fetch list of 3d props to load
    static3d_keys = study["creation"]["config"]["vectors"]["static3d"]
    dynamic3d_keys = study["creation"]["config"]["vectors"]["dynamic3d"]
    static3d, dynamic3d = extract_ensemble.get_3dprops(realizations, storage, static3d_keys, dynamic3d_keys)
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_ext = str(datetime.fromtimestamp(timestamp))
    
    # save results 
    studies["status"] = "extracted"
    studies["extraction"] = {}
    studies["extraction"]["timestamp"] = dt_ext
    studies["extraction"]["summary"] = summary
    studies["extraction"]["static3d"] = static3d
    studies["extraction"]["dynamic3d"] = dynamic3d
    u.save_to_json(study_path, studies)
    
    
    # print objective and constraints based on the parameters in the "study", 
    # make sure to use the same function as the one in the "optimize_ensemble.py" code.
    ### CALCULATING OBJECTIVES
    study = u.read_json(study_path)
    ext_dict = optimize_ensemble.create_extension_folders(study)
    study['extension'] = ext_dict
    u.save_to_json(study_path, study)
    
    unit = optimize_ensemble.get_unit(study)
    summary_folder = study['extraction']['summary']
    study = optimize_ensemble.calculate_npv(study, unit, summary_folder) 
    
    u.save_to_json(study_path, study)
    
    ### PRINTING OBJECTIVE FUNCTIOn
    
    of_path = study['extension']['optimization']['OF']
    
    ofs = np.load(of_path)
    ofs = np.cumsum(ofs, axis=0)
    of = np.mean(ofs, axis=1)[-1]
    
    print(f"OBJECTIVE FUNCTION = {of}")
    
    ## PRINTING CONSTRAINTS
    
    # check constraints
    eqs = []
    ineqs = []
    constraints = config['optimization']['parameters']['constraints']
    for c, d in constraints.items():
        if not d['is_active']:
            continue 
        
        def pick_timestep(vector:np.ndarray, d:dict):
            if d['timestep'] == 'all':
                return np.max(vector)
            elif d['timestep'] == 'last':
                return vector[-1]
            else:
                raise Exception(f"'timestep' must be 'all' or 'last'. Found {d['timestep']}.")
        
        def compute_normalized_constraint(vector:list, d:dict):
            vector = np.array(vector)[is_success]
            val = optimize_ensemble.calc_stats(vector, d['robustness']['type'], d['robustness']['value'])
            # val = (d['value'] - val)/abs(d['value']) 
            return val
        
        if c == "FWPT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FWPT'])
                val = pick_timestep(vector, d)
                v_list.append(val)
           
            val = compute_normalized_constraint(v_list, d)     
            
        elif "WWPT" in c:
            well = d["wellname"]
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename][f'WWPT:{well}'])
                val = pick_timestep(vector, d)
                v_list.append(val)

            val = compute_normalized_constraint(v_list, d)    
        
        elif c == "FWCT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FWCT'])
                val = pick_timestep(vector, d)
                v_list.append(val)
           
            val = compute_normalized_constraint(v_list, d)     
            
        elif "WWCT" in c:
            well = d["wellname"]
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename][f'WWCT:{well}'])
                val = pick_timestep(vector, d)
                v_list.append(val)

            val = compute_normalized_constraint(v_list, d) 

        elif c == "FGPT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FGPT'])
                val = pick_timestep(vector, d)
                v_list.append(val)
           
            val = compute_normalized_constraint(v_list, d)
        
        else:
            raise NotImplementedError(f"Constraint of type {c} has not been implemented yet.")
        
        if d['type'] == "inequality":
            ineqs.append(val)
        elif d['type'] == "equality":
            eqs.append(val)
            
    print(f"EQUALITY : {eqs}")
    print(f"INEQUALITY : {ineqs}")
    
    return

if __name__ == '__main__':
    main(sys.argv[1:])