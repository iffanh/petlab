import numpy as np
import sys
import utils.utilities as u
import utils.deck_parser as dp
import run_ensemble
import extract_ensemble
import py_trsqp.trsqp as trsqp

import os
import scipy.stats

from pathlib import Path

STORAGE_DIR = './simulations/storage/'
STUDIES_DIR = './simulations/studies/'

def replace_single_value(d):
    
    p = d["parameters"]
    if d["name"] == 'Normal':
            
        a = (p['min'] - p['mean']) / p['std']
        b = (p['max'] - p['mean']) / p['std']
        sample = scipy.stats.truncnorm.rvs(a, b)
        sample = sample*p['std'] + p['mean']

    elif d["name"] == 'LogNormal':
        
        a = (np.log(p['min']) - np.log(p['mean'])) / np.log(p['std'])
        b = (np.log(p['max']) - np.log(p['mean'])) / np.log(p['std'])
        sample = scipy.stats.truncnorm.rvs(a, b)
        sample = np.exp(sample*np.log(p['std']) + np.log(p['mean']))

    elif d["name"] == 'Constant':
        sample = p['value']

    else:
        raise ValueError("%s distribution not implemented yet" %d["name"])

    if p['type'] == "float":
        replaced_value = '%.3f'%sample
    elif p['type'] == "int":
        replaced_value = '%s'%int(sample)
            
    return replaced_value

def mutate_case(root_datafile_path, real_datafile_path, parameters, optimization):

    with open(root_datafile_path, 'r') as file :
        filedata = file.read()

    # parameters
    for param in parameters:
        
        Name = param["Name"]
        Type = param["Type"]
        d = param["Distribution"]
        
        if Type == "SingleValue":
            replaced_value = replace_single_value(d)
            
        filedata = filedata.replace(Name, replaced_value)

    # optimization
    for optim in optimization:
        Name = optim["Name"]
        Default = optim["Default"]
        if optim['type'] == "float":
            replaced_value = '%.3f '%Default
        elif optim['type'] == "int":
            replaced_value = '%s '%int(Default)
            
        filedata = filedata.replace(Name, replaced_value)

    # Write the file out again
    with open(real_datafile_path, 'w') as file:
        file.write(filedata)

    return 

def mutate_cases(data, root_datafile_path):

    _, tail = os.path.split(root_datafile_path) # dir_path = /path/to/data
    root_name = os.path.splitext(tail)[0] #root_name = SPE1

    ens_path = os.path.join(STORAGE_DIR, data['Name'])
    Path(ens_path).mkdir(parents=True, exist_ok=True)

    real_files = {}
    for i in range(1, data['Ne']+1):
        real_name = root_name + '_%s'%i # SPE1_i
        
        real_path = os.path.join(ens_path, real_name) # /path/to/data/SPE1_i
        Path(real_path).mkdir(parents=True, exist_ok=True)

        real_datafile_path = os.path.join(real_path, real_name + '.DATA') # /path/to/data/SPE1_i/SPE1_i.DATA
        
        mutate_case(root_datafile_path, real_datafile_path, data['parameters'], data['optimization'])

        real_files[real_name] = real_datafile_path

    return real_files

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

def create_extension_folders(study):
    """
    Copy the ensemble, but re-replace the optimization keywords
    """
    
    root_path = study["creation"]["root"]
    
    storage = study["simulation"]["storage"]
    
    study_folder, tail = os.path.split(root_path) # dir_path = /path/to/data
    root_name = os.path.splitext(tail)[0] #root_name = SPE1
    opt_root_name = root_name + '_Opt' # SPE1_Opt
    
    opt_storage = os.path.join(STORAGE_DIR, 'OPT_' + study["Name"])
    Path(opt_storage).mkdir(parents=True, exist_ok=True)
    print(f"Optimization storage folder {opt_storage} created")
    
    # create results folder
    res_storage = os.path.join(opt_storage, 'results')
    Path(res_storage).mkdir(parents=True, exist_ok=True)
    
    base_realizations = study['creation']['base_realizations']
    
    extension_dict = {} 
    extension_dict['storage'] = opt_storage
    extension_dict['realizations'] = {}
    extension_dict['optimization'] = {}
    extension_dict['historymatching'] = {}
    
    # copy the ensemble
    for i, real_name in enumerate(study['simulation']['realizations']):
        real_data_path = study['simulation']['realizations'][real_name]
        
        #create subfolder
        opt_realpath = os.path.join(opt_storage, real_name)
        Path(opt_realpath).mkdir(parents=True, exist_ok=True)
        
        real_name = root_name + '_%s'%(i+1) # SPE1_i
        
        real_path = os.path.join(opt_storage, real_name) # /path/to/data/SPE1_i
        Path(real_path).mkdir(parents=True, exist_ok=True)

        real_datafile_path = os.path.join(real_path, real_name + '.DATA') # /path/to/data/SPE1_i/SPE1_i.DATA
        extension_dict['realizations'][real_name] = real_datafile_path
        
    return extension_dict

def calculate_npv(study, unit, summary_folder):
    
    """
    Calculate NPV based on the production data. The following
    keywords has to be imported from the deck.
    
        FOPR : sm3/day [METRIC], STB/day [FIELD]
        FWPR : sm3/day [METRIC], STB/day [FIELD]
        FGPR : sm3/day [METRIC], Mscf/day [FIELD]
        
        FWIR : sm3/day [METRIC], STB/day [FIELD]
        FGIR : sm3/day [METRIC], Mscf/day [FIELD]
        
    NPV is stored in study['extension']['optimization']
    """
    
    ## Define npv parameters
    if unit == "METRIC":
        ropr = 90.5 # oil price -- $/stb 
        rgpr = 1.5 # gas price -- $/Mscf
        
        rwpr = 2.0 # water production cost -- $/stb
        rwir = 2.0 # water injection cost -- $/stb
        rgir = 10.0 # gas injection cost -- $/Mscf
        
    elif unit == "FIELD":
        
        ropr = 565.7 # oil price 90.5/0.16 -- $/sm3 
        rgpr = 0.0529 # gas price 1.5/28.316 -- $/sm3
        
        rwpr = 12.5 # water production cost 2.0/0.16 -- $/sm3
        rwir = 12.5 # water injection cost 2.0/0.16 -- $/sm3
        rgir = 0.353 # gas injection cost 10.0/28.316 -- $/sm3
    
    
    d = 0.05
    realizations = study['simulation']['realizations']
    
    tN = np.inf
    
    for i, real in enumerate(realizations):
        years = np.load(summary_folder[real]["YEARS"]) 
        
        tM = len(years)
        if tM < tN:
            tN = tM*1
            filename = os.path.join(study['extension']['storage'], 'results', 'years.npy')
            np.save(filename, years)
            study['extension']['optimization']['YEARS'] = filename
    
    npv_arr = []
    for i, real in enumerate(realizations):
        years = np.load(summary_folder[real]["YEARS"]) 
        fopr = np.load(summary_folder[real]["FOPR"])
        fwpr = np.load(summary_folder[real]["FWPR"])
        fgpr = np.load(summary_folder[real]["FGPR"])
        
        fgir = np.load(summary_folder[real]["FGIR"])
        fwir = np.load(summary_folder[real]["FWIR"])
            
        tM = len(years)
        
        dy = np.diff(years, prepend=0)
        cashflow = (fopr*ropr + fgpr*rgpr - fwpr*rwpr - fgir*rgir - fwir*rwir)*dy*365
        npv = []

        for tn in range(tN):
            tm = tM*tn/tN
            w1 = tm - np.floor(tm)
            w2 = np.ceil(tm) - tm
            npv_t = (w1*cashflow[int(np.floor(tm))] + w2*cashflow[int(np.ceil(tm))])/(1+d)**tm
            npv.append(npv_t)
            
        npv_arr.append(npv)
        
    npv_arr = np.array(npv_arr).T
    filename = os.path.join(study['extension']['storage'], 'results', 'NPV.npy')
    np.save(filename, npv_arr)

    study['extension']['optimization']['NPV'] = filename
    
    return study

def get_unit(study):
    
    reals = list(study['simulation']['realizations'].keys())
    real = study['simulation']['realizations'][reals[0]]
    
    DeckParser = dp.DeckParser()
    is_metric, _ = DeckParser.keyword_search(real, keyword="METRIC")
    if is_metric:
        unit = "METRIC"
    else:
        unit = "FIELD"
        
    return unit

def formulate_problem(study):
    
    unit = get_unit(study)
    study = calculate_npv(study, unit, summary_folder=study['extraction']['summary'])
    
    return study

def cost_function(x, study_path, simulator_path):
    
    study = u.read_json(study_path)
    
    config = u.read_json(study['creation']['json'])
    controls = config['controls']
    
    
    for i, control in enumerate(controls):
        control["Default"] = x[i]
    
    simfolder_path = study['extension']['storage']
    try:
        run_ensemble.run_cases(simulator_path, study, simfolder_path, controls)
    except RuntimeError:
        return -np.nan
    
    realizations = study['extension']['realizations'] 
    storage = study['extension']['storage']
    summary = extract_ensemble.get_summary(realizations, storage)
    study['extension']['optimization']['summary'] = summary
    
    u.save_to_json(study_path, study)
    
    # get npv
    unit = get_unit(study)
    study = calculate_npv(study, unit, summary)
    npv_path = study['extension']['optimization']['NPV']
    npv_arr = np.load(npv_path)
    npv_T = np.cumsum(npv_arr, axis=1)[:,-1]
    npv_cf = np.mean(npv_T, axis=0)

    return -npv_cf

def run_optimization(study_path, simulator_path):
    
    study = u.read_json(study_path)
    # Fetch initial value
    config_path = study['creation']['json']
    config = u.read_json(config_path)
    controls = config["controls"]
    Nc = len(controls)
    x0 = [c["Default"] for c in controls]
    
    # define cost function
    cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function(x, study_path, simulator_path)
    
    
    CONSTANTS = {}
    CONSTANTS['L_threshold'] = 1.00
    tr = trsqp.TrustRegionSQPFilter(x0, 
                                    k=2*Nc+1,
                                    cf=cf,
                                    constants=CONSTANTS)
    
    tr.optimize(max_iter=5)
    
    return

def main(args):
    
    simulator_path = args[0]
    study_path = args[1]
    study = u.read_json(study_path)   
    
    ext_dict = create_extension_folders(study)
    study['extension'] = ext_dict
    u.save_to_json(study_path, study)
    
    study = formulate_problem(study)
    u.save_to_json(study_path, study)
    
    
    run_optimization(study_path, simulator_path)
    
    
    return

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex. python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_PoroPerm_Opt_RandomField.json 
    """
    main(sys.argv[1:])