import sys
from ecl.summary import EclSum
# from ecl.grid import EclGrid
# from ecl.eclfile import Ecl3DFile, EclRestartFile
import ecl.eclfile
try:
    from .utils import utilities as u
except ImportError:
    import utils.utilities as u 
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
                     "FWIT", 
                     "FGIT",
                     "FOIT",
                     "WWIT",
                     "WGIT",
                     "WOIT",
                     "FWIR",
                     "FGIR",
                     "FOIR",
                     "WWIR",
                     "WGIR",
                     "WOIR",
                     "FGOR",
                     "FWCT",
                     "WFOR",
                     "WBHP",
                     "WWCT"]
        
class Static3DPropertyKeys():
    def __init__(self) -> None:
        self.keys = ["PORO",
                     "PERMX",
                     "PERMY",
                     "PERMZ",
                     "NTG"]

class Dynamic3DPropertyKeys():
    def __init__(self) -> None:
        # self.keys = ["PRESSURE",
        #              "SGAS",
        #              "SWAT",
        #              "RS"]
        self.keys = []
        
def get_summary(realizations, storage, sum_keys:list):
    
    casename = list(realizations.keys())[0]
    # for casename in realizations:
    real_path = realizations[casename]
    
    # dirname = os.path.dirname(real_path)
    # summary_path = os.path.join(dirname, casename + ".SMSPEC")
    
    summary = EclSum(real_path)
    
    summary_keys = summary.keys()
    available_keys = ["YEARS"]
    for k in summary_keys:
        # for _k in SummaryKeys().keys:
        for _k in sum_keys:
            if _k in k:
                available_keys.append(k)
                
    data_dir = os.path.join(storage, "results", "summary")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    summary_dict = {}
    pbar = tqdm(available_keys, total=len(available_keys), desc="Summary: ", disable=True)
    for k in pbar:
        pbar.set_description(f"Extracting {k}")
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
    
    return summary_dict

def get_3dprops(realizations, storage, static3d_keys:list, dynamic3d_keys:list):
    
    cs = list(realizations.keys())[0]
    # for casename in realizations:
    real_path = realizations[cs]
    dirname = os.path.dirname(real_path)
    
    # Save static
    static3d_path = os.path.join(dirname, cs + ".INIT")
    static3d = ecl.eclfile.EclFile(static3d_path)
    
    static_available_keys = []
    for k in static3d.keys():
        # for _k in Static3DPropertyKeys().keys:
        for _k in static3d_keys:
            if _k in k:
                static_available_keys.append(k)
        
    data_dir = os.path.join(storage, "results", "static3d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    static3d_dict = {}
    pbar = tqdm(static_available_keys, total=len(static_available_keys), desc="Static3D: ", disable=True)
    for k in pbar:
        pbar.set_description(f"Extracting {k}")
        for casename in realizations:
            
            try:
                tmp = static3d_dict[casename]
            except:
                static3d_dict[casename] = {}
            
            real_path = realizations[casename]
            dirname = os.path.dirname(real_path)
            static3d_path = os.path.join(dirname, casename + ".INIT")
            static3d = ecl.eclfile.EclFile(static3d_path)
    
            vector = static3d[k][0].numpy_view()
        
            real_dir = os.path.join(data_dir, casename)
            Path(real_dir).mkdir(parents=True, exist_ok=True)
            
            filename = os.path.join(real_dir, f"{k}.npy")
            np.save(filename, vector)
            
            static3d_dict[casename][f"{k}"] = filename
    
    
    cs = list(realizations.keys())[0]
    # for casename in realizations:
    real_path = realizations[cs]
    dirname = os.path.dirname(real_path)
    # Save dynamic
    dynamic3d_path = os.path.join(dirname, cs + ".UNRST")
    dynamic3d = ecl.eclfile.EclFile(dynamic3d_path)
    
    dynamic_available_keys = []
    for k in dynamic3d.keys():
        # for _k in Dynamic3DPropertyKeys().keys:
        for _k in dynamic3d_keys:
            if _k in k:
                dynamic_available_keys.append(k)
        
    data_dir = os.path.join(storage, "results", "dynamic3d")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    dynamic3d_dict = {}
    pbar = tqdm(dynamic_available_keys, total=len(dynamic_available_keys), desc="Dynamic3D: ", disable=True)
    for k in pbar:
        pbar.set_description(f"Extracting {k}")
        for casename in realizations:
            real_path = realizations[casename]
            dirname = os.path.dirname(real_path)
            real_dir = os.path.join(data_dir, casename)
            Path(real_dir).mkdir(parents=True, exist_ok=True)
            
            dynamic3d_path = os.path.join(dirname, casename + ".UNRST")
            dynamic3d = ecl.eclfile.EclFile(dynamic3d_path)
    
            try:
                tmp = dynamic3d_dict[casename]
            except:
                dynamic3d_dict[casename] = {}
                filename = os.path.join(real_dir, "DATES.npy")
                np.save(filename, dynamic3d.report_dates)
                dynamic3d_dict[casename][f"{k}"] = filename
                
            matrix = []
            for t in range(len(dynamic3d.report_dates)):
                matrix.append(dynamic3d[k][0].numpy_view())
            
            matrix = np.array(matrix)
            filename = os.path.join(real_dir, f"{k}.npy")
            np.save(filename, matrix)
            
            dynamic3d_dict[casename][f"{k}"] = filename
    
    return static3d_dict, dynamic3d_dict

def main(argv):

    study_path = argv[0]
    studies = u.read_json(study_path)
    
    realizations = studies["simulation"]["realizations"]
    storage = studies["simulation"]["storage"]
    
    # fetch list of summaries to load
    sum_keys = studies["creation"]["config"]["vectors"]["summary"]
    
    # check summary
    summary = get_summary(realizations, storage, sum_keys)
    
    # fetch list of 3d props to load
    static3d_keys = studies["creation"]["config"]["vectors"]["static3d"]
    dynamic3d_keys = studies["creation"]["config"]["vectors"]["dynamic3d"]
    static3d, dynamic3d = get_3dprops(realizations, storage, static3d_keys, dynamic3d_keys)
    
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
    
    pass

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex. python3 src/extract_ensemble.py simulations/studies/IE_Poro.json 
    """
    main(sys.argv[1:])

