#!usr/bin/python


from utils import deck_parser as dp
from datetime import datetime
import utils.utilities as u
import sys
import os

from pathlib import Path

STORAGE_DIR = './simulations/storage/'
STUDIES_DIR = './simulations/studies/'
Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(STUDIES_DIR).mkdir(parents=True, exist_ok=True)

def mutate_case(root_datafile_path, real_datafile_path, parameters):

    with open(root_datafile_path, 'r') as file :
        filedata = file.read()

    # parameters
    for param in parameters:
        
        Name = param["Name"]
        Type = param["Type"]
        d = param["Distribution"]
        
        if Type == "SingleValue":
            replaced_value = u.replace_single_value(d)
        if Type == "RandomField":
            replaced_value = u.replace_random_field(d)
            
        filedata = filedata.replace(Name, replaced_value)

    # # optimization
    # for optim in optimization:
    #     Name = optim["Name"]
    #     Default = optim["Default"]
    #     if optim['type'] == "float":
    #         replaced_value = '%.3f '%Default
    #     elif optim['type'] == "int":
    #         replaced_value = '%s '%int(Default)
            
    #     filedata = filedata.replace(Name, replaced_value)

    # Write the file out again
    with open(real_datafile_path, 'w') as file:
        file.write(filedata)


def mutate_cases(data, root_datafile_path):

    _, tail = os.path.split(root_datafile_path) # dir_path = /path/to/data
    root_name = os.path.splitext(tail)[0] #root_name = SPE1

    base_ens_path = os.path.join(STORAGE_DIR, 'BASE_' + data['Name'])
    Path(base_ens_path).mkdir(parents=True, exist_ok=True)

    real_files = {}
    for i in range(1, data['Ne']+1):
        real_name = root_name + '_%s'%i # SPE1_i
        
        real_path = os.path.join(base_ens_path, real_name) # /path/to/data/SPE1_i
        Path(real_path).mkdir(parents=True, exist_ok=True)

        real_datafile_path = os.path.join(real_path, real_name + '.DATA') # /path/to/data/SPE1_i/SPE1_i.DATA
        
        mutate_case(root_datafile_path, real_datafile_path, data['parameters'])

        real_files[real_name] = real_datafile_path

    return real_files

def dump_ensemble(data, real_files, root_datafile_path, json_path):

    ensemble_study = os.path.join(STUDIES_DIR, '%s.json' %data['Name'])

    # current date and time
    now = datetime.now()

    timestamp = datetime.timestamp(now)
    dt_object = str(datetime.fromtimestamp(timestamp))

    ens_path = os.path.join(STORAGE_DIR, data['Name'])
    
    study = {'status':"created",
             'Name': data['Name'],
             'creation': {
                'root': root_datafile_path,
                'json': json_path,
                'timestamp': dt_object,
                'base_realizations': real_files,
                'storage': ens_path}    
             }
             
    u.save_to_json(ensemble_study, study)

def main(argv):

    # root_datafile_path = argv[0] # root_path to the data file e.g. /path/to/data/SPE1.DATA
    json_path = argv[0] # path to the json file e.g. /path/to/json/SPE1.json
    
    if not os.path.isfile(json_path):
        raise ValueError("%s cannot be found" %json_path)
    
    config = u.read_json(json_path)
    
    root_datafile_path = config["root"]
    if not os.path.isfile(root_datafile_path):
        raise ValueError("%s cannot be found" %root_datafile_path)

    data = u.read_json(json_path)
    real_files = mutate_cases(data, root_datafile_path)

    dump_ensemble(data, real_files, root_datafile_path, json_path)
    
if __name__ == "__main__":
    """The arguments are the following:
    1. json path (str): path to the .json file that explains the uncertainties in the model 
    
    Ex: "python3 src/create_ensemble.py data/SPE1_Ensemble/SPE1_Poro.json"
    """

    main(sys.argv[1:])