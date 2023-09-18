import sys
from ecl.summary import EclSum
import utils.utilities as u
import numpy as np
import os
from pathlib import Path

class SummaryKeys():
    
    def __init__(self) -> None:
        self.keys = ["FWPT", 
                     "FGPT",
                     "FOPT",
                     "WWPT",
                     "WGPT",
                     "WOPT",
                     "FWPR",
                     "FGPR",
                     "FOPR",
                     "WWPR",
                     "WGPR",
                     "WOPR",
                     "FGOR",
                     "WFOR",
                     "WBHP"]

def get_summary(study_path):
    
    studies = u.read_json(study_path)
    realizations = studies["creation"]["realizations"]
    
    casename = list(realizations.keys())[0]
    # for casename in realizations:
    real_path = realizations[casename]
    summary = EclSum(real_path)
    
    summary_keys = summary.keys()
    available_keys = ["YEARS"]
    for k in summary_keys:
        for _k in SummaryKeys().keys:
            if _k in k:
                available_keys.append(k)
                
    data_dir = os.path.join(studies["creation"]["storage"], "results", "summary")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    summary_dict = {}
    for k in available_keys:
        for casename in realizations:
            
            try:
                tmp = summary_dict[casename]
            except:
                summary_dict[casename] = {}
            
            real_path = realizations[casename]
            summary = EclSum(real_path)
            vector = summary.numpy_vector(k)
        
            real_dir = os.path.join(data_dir, casename)
            Path(real_dir).mkdir(parents=True, exist_ok=True)
            
            filename = os.path.join(real_dir, f"{k}.npy")
            np.save(filename, vector)
            
            summary_dict[casename][f"{k}"] = filename
    
    studies["status"] = "extracted"
    studies["extraction"] = {}
    studies["extraction"]["summary"] = summary_dict
    u.save_to_json(study_path, studies)

    

def main(argv):

    study_path = argv[0]
    
    # check summary
    get_summary(study_path)
    
    # check 
    
    pass

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex. python3 src/extract_ensemble.py simulations/studies/IE_Poro.json 
    """
    main(sys.argv[1:])

