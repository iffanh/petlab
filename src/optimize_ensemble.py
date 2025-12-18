import numpy as np
import sys

try:
    from .utils import utilities as u
    from .utils import deck_parser as dp
    from . import run_ensemble
    from . import extract_ensemble
except ImportError:
    import utils.utilities as u
    import utils.deck_parser as dp
    import run_ensemble
    import extract_ensemble
    
# import py_trsqp.trsqp as trsqp

import os
from datetime import datetime

from pathlib import Path


STORAGE_DIR = './simulations/storage/'
STUDIES_DIR = './simulations/studies/'

def mutate_case(root_datafile_path, real_datafile_path, parameters, optimization):

    with open(root_datafile_path, 'r') as file :
        filedata = file.read()

    # parameters
    for param in parameters:
        
        Name = param["Name"]
        Type = param["Type"]
        d = param["Distribution"]
        
        if Type == "SingleValue":
            replaced_value = u.replace_single_value(d)
            
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

def calculate_cost_function(study):
    config = u.read_json(study['creation']['json'])
    cf_type = config['optimization']['parameters']['costFunction'] 
    if cf_type == "NPV":
        return calculate_npv
    elif cf_type == "NetCashFlow-Last":
        return calculate_net_cash_flow
    
def calculate_net_cash_flow(study, unit, summary_folder):
    
    # From Jinjie
    FCMIT = 732000000 # m3
    FCMIT = FCMIT * 1.98 / 1000 # ton
    FWIT = 3600000 # m3
    coil = 85 #$/bbl
    coil = coil * 6.2898 #$/m3
    cco2tax = 86*44/1000 #$/ton
    cwp = 8 #$/bbl
    cwp = cwp * 6.2898 #$/m3
    cwi = 8 #$/bbl
    cwi = cwi * 6.2898 #$/m3
    cco2i = 50 #$/ton
    
    
    cashflow_arr = []
    realizations = study['simulation']['realizations']
    for i, real in enumerate(realizations):
        FWPT = np.load(summary_folder[real]["FWPT"])[-1]
        FOPT = np.load(summary_folder[real]["FOPT"])[-1]
        FCO2PT = np.load(summary_folder[real]["FCMPT_1"])[-1]

        CashFlow = coil*FOPT + cco2tax*FCO2PT - cwp*FWPT - cwi*FWIT - cco2i*FCMIT

        cashflow_arr.append(CashFlow)
        
    # cashflow = np.nanmean(cashflow_arr)
    cashflow = np.array(cashflow_arr)
    filename = os.path.join(study['extension']['storage'], 'results', 'cashflow.npy')
    np.save(filename, cashflow)

    study['extension']['optimization']['OF'] = filename
    
    return study
    
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
    if unit == "FIELD":
        ropr = 80.0 # oil price -- $/stb 
        rgpr = 1.5 # gas price -- $/Mscf
        
        rwpr = 10.0 # 20.0 # water production cost -- $/stb
        rwir = 5.0 # 20.0 # water injection cost -- $/stb
        rgir = 2.0 # gas injection cost -- $/Mscf
        
    elif unit == "METRIC":
        
        ropr = 503.185 # oil price 90.5/0.16 -- $/sm3 
        rgpr = 0.0529 # gas price 1.5/28.316 -- $/sm3
        
        rwpr = 62.9 # 125.796 # water production cost 2.0/0.16 -- $/sm3
        rwir = 31.5 # water injection cost 2.0/0.16 -- $/sm3
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
        cashflow = 0
        years = np.load(summary_folder[real]["YEARS"]) 
        
        try:
            fopr = np.load(summary_folder[real]["FOPR"])
            cashflow += fopr*ropr
        except KeyError:
            pass
        
        try:
            fwpr = np.load(summary_folder[real]["FWPR"])
            cashflow += -fwpr*rwpr
        except KeyError:
            pass
        
        try:
            fgpr = np.load(summary_folder[real]["FGPR"])
            cashflow += fgpr*rgpr
        except KeyError:
            pass
        
        try:
            fgir = np.load(summary_folder[real]["FGIR"])
            cashflow += -fgir*rgir
        except KeyError:
            pass
        
        try:
            fwir = np.load(summary_folder[real]["FWIR"])
            cashflow += -fwir*rwir
        except KeyError:
            pass
        
        tM = len(years)
        
        dy = np.diff(years, prepend=0)
        # cashflow = (fopr*ropr + fgpr*rgpr - fwpr*rwpr - fgir*rgir - fwir*rwir)*dy*365
        cashflow = cashflow*dy*365
        npv = []

        for tn in range(tN):
            tm = tM*tn/tN
            w1 = tm - np.floor(tm)
            w2 = np.ceil(tm) - tm
            if w1 == 0.0:
                npv_t = cashflow[int(np.floor(tm))]/(1+d)**tm
            else:
                npv_t = (w1*cashflow[int(np.floor(tm))] + w2*cashflow[int(np.ceil(tm))])/(1+d)**tm
            npv.append(npv_t)
        npv_arr.append(npv)
        
    npv_arr = np.array(npv_arr).T
    filename = os.path.join(study['extension']['storage'], 'results', 'NPV.npy')
    np.save(filename, npv_arr)

    study['extension']['optimization']['OF'] = filename
    
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
    study = calculate_cost_function(study)(study, unit, summary_folder=study['extraction']['summary'])
    # study = calculate_npv(study, unit, summary_folder=study['extraction']['summary'])
    
    return study

def calc_stats(vec:np.ndarray, stats_type:str, *args):
    
    if stats_type == "average":
        r = np.mean(vec)
    elif stats_type == "percentile":
        r = np.percentile(vec, args[0])
    
    return r

def cost_function_stosag(x, study_path, simulator_path):
    
    study = u.read_json(study_path)
    config = u.read_json(study['creation']['json'])
    
    # if config['optimization']['parameters']['optimizer'] == "STOSAG":
    if x.ndim == 2 and config['optimization']['parameters']['optimizer'] == "STOSAG": #
        # In STOSAG, each control correspond to one ensemble member. Thus, we must set the controls one by one.
        # Here, x must be an array of controls with length equal to the number of ensemble members.  
        # The size of x is N ensemble x N controls
        
        Ne = x.shape[0]
        controls = [ config['controls'] for i in range(Ne)] # initialize controls for each ensemble member
         
        for i, controls_i in enumerate(controls):
            for j, control in enumerate(controls_i):
                control["Default"] = x[i, j]
            
        simfolder_path = study['extension']['storage']
        
    else:
        
        controls = config['controls']
        for i, control in enumerate(controls):
            try:
                control["Default"] = x[i]
            except TypeError: # for NOMAD
                control["Default"] = x.get_coord(i)
    
        simfolder_path = study['extension']['storage']
        
    try:
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
        
    except RuntimeError:
        return (-np.nan, np.nan, np.nan)
    
    except TimeoutError:
        return (-np.nan, np.nan, np.nan)
    
    # print(is_success)
    if not any(is_success): #every realization is 'false' (failed)
        return (-np.nan, np.nan, np.nan)
    
    realizations = study['extension']['realizations'] 
    storage = study['extension']['storage']
    # fetch list of summaries to load
    sum_keys = study["creation"]["config"]["vectors"]["summary"]
    summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
    study['extension']['optimization']['summary'] = summary
    
    u.save_to_json(study_path, study)
    
    
    ### In STOSAG, we want to return all the values, not just the robustness measure
    
    # cost functions
    cf_type = config['optimization']['parameters']['costFunction'] 
    if cf_type == "NPV":
        # get npv
        unit = get_unit(study)
        study = calculate_cost_function(study)(study, unit, summary)
        # study = calculate_npv(study, unit, summary)
        npv_path = study['extension']['optimization']['OF']
        npv_arr = np.load(npv_path)[:, is_success]
        # npv_T = np.cumsum(npv_arr, axis=1)[:,-1]
        npv_T = np.cumsum (npv_arr, axis=0)
        cf = npv_T[-1,:]
        
    elif cf_type == "NetCashFlow-Last":
        
        unit = get_unit(study)
        study = calculate_cost_function(study)(study, unit, summary)
        
        cashflow_path = study['extension']['optimization']['OF']
        cf = np.load(cashflow_path)
    
    else:
        raise NotImplementedError(f"Cost function of type {cf_type} has not been implemented yet.")
        
    eqs = []
    ineqs = []
    
    # constraints
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
            val = calc_stats(vector, d['robustness']['type'], d['robustness']['value'])
            val = (d['value'] - val)/abs(d['value']) 
            return val
        
        if c == "FWPT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FWPT'])
                val = pick_timestep(vector, d)
                v_list.append(val)  
            
        elif "WWPT" in c:
            well = d["wellname"]
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename][f'WWPT:{well}'])
                val = pick_timestep(vector, d)
                v_list.append(val)    
        
        elif c == "FWCT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FWCT'])
                val = pick_timestep(vector, d)
                v_list.append(val)     
            
        elif "WWCT" in c:
            well = d["wellname"]
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename][f'WWCT:{well}'])
                val = pick_timestep(vector, d)
                v_list.append(val) 

        elif c == "FGPT":
            v_list = []
            for casename in summary.keys():  
                vector = np.load(summary[casename]['FGPT'])
                val = pick_timestep(vector, d)
                v_list.append(val)
        
        else:
            raise NotImplementedError(f"Constraint of type {c} has not been implemented yet.")
        
        if d['type'] == "inequality":
            ineqs.append(v_list)
        elif d['type'] == "equality":
            eqs.append(v_list)
    
    results = (-cf, eqs, ineqs)
    return results


@u.np_cache
def cost_function(x, study_path, simulator_path):
    
    study = u.read_json(study_path)
    config = u.read_json(study['creation']['json'])
    
    controls = config['controls']
    for i, control in enumerate(controls):
        try:
            control["Default"] = x[i]
        except TypeError: # for NOMAD
            control["Default"] = x.get_coord(i)

    simfolder_path = study['extension']['storage']
        
    try:
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
    except RuntimeError:
        return (-np.nan, np.nan, np.nan)
    
    except TimeoutError:
        return (-np.nan, np.nan, np.nan)
    
    # print(is_success)
    if not any(is_success): #every realization is 'false' (failed)
        return (-np.nan, np.nan, np.nan)
    
    realizations = study['extension']['realizations'] 
    storage = study['extension']['storage']
    # fetch list of summaries to load
    sum_keys = study["creation"]["config"]["vectors"]["summary"]
    summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
    study['extension']['optimization']['summary'] = summary
    
    u.save_to_json(study_path, study)
    
    # cost functions
    cf_type = config['optimization']['parameters']['costFunction'] 
    if cf_type == "NPV":
        # get npv
        unit = get_unit(study)
        study = calculate_cost_function(study)(study, unit, summary)
        # study = calculate_npv(study, unit, summary)
        npv_path = study['extension']['optimization']['OF']
        npv_arr = np.load(npv_path)[:, is_success]
        # npv_T = np.cumsum(npv_arr, axis=1)[:,-1]
        npv_T = np.cumsum (npv_arr, axis=0)
        cf = np.mean(npv_T, axis=1)[-1]
        
    elif cf_type == "NetCashFlow-Last":
        
        unit = get_unit(study)
        study = calculate_cost_function(study)(study, unit, summary)
        
        cashflow_path = study['extension']['optimization']['OF']
        cashflow = np.load(cashflow_path)
        cf = np.mean(cashflow)
    
    else:
        raise NotImplementedError(f"Cost function of type {cf_type} has not been implemented yet.")
        
    eqs = []
    ineqs = []
    
    # constraints
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
            val = calc_stats(vector, d['robustness']['type'], d['robustness']['value'])
            val = (d['value'] - val)/abs(d['value']) 
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
    
    results = (-cf, eqs, ineqs)
    return results

def get_n_constraints(constraints:dict):
    
    n_eq = 0
    n_ineq = 0
    for c, d in constraints.items():
        if d['is_active']:
            if d['type'] == 'inequality':
                n_ineq += 1
            elif d['type'] == 'equality':
                n_eq += 1
    
    return n_eq, n_ineq

def run_optimization(study_path, simulator_path):
    
    study = u.read_json(study_path)
    # Fetch initial value
    config_path = study['creation']['json']
    config = u.read_json(config_path)
    controls = config["controls"]
    Nc = len(controls)
    x0 = [c["Default"] for c in controls]
    
    constraints = config['optimization']['parameters']['constraints']
    n_eq, n_ineq = get_n_constraints(constraints)
    
    optimizer = config['optimization']['parameters']['optimizer']

    if optimizer == "DFTR":
        # define cost function
        import py_trsqp.trsqp as trsqp
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function(x, study_path, simulator_path)[0]
        eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
        # define bounds
        lb = []
        ub = []
        for c in controls:
            try:
                lb.append(c['lb'])
            except:
                lb.append(-np.inf)
                
            try:
                ub.append(c['ub'])
            except:
                ub.append(np.inf)
        
        # ub = [c["ub"] for c in controls]
        
        # redefine constants
        opt_constants = config['optimization']['parameters']['constants']
        opts = config['optimization']['parameters']['options']
        
        tr = trsqp.TrustRegionSQPFilter(x0, 
                                        cf=cf,
                                        lb=lb,
                                        ub=ub,
                                        eqcs=[*eqs],
                                        ineqcs=[*ineqs],
                                        constants=opt_constants,
                                        opts=opts)
        
        
        tr.optimize(max_iter=config['optimization']['parameters']['maxIter'])
        
        cost_function.cache_clear()
        
        # run best points
        print("run the latest ...")
        # cost_function(tr.iterates[-1]["y_curr"], study_path, simulator_path)

        # xsol = tr.iterates[-1]["y_curr"]
        xsol = tr.iterates[-1]['best_point']["y"]
        study = u.read_json(study_path)
        config = u.read_json(study['creation']['json'])
        controls = config['controls']
        
        for i, control in enumerate(controls):
            control["Default"] = xsol[i]
        
        simfolder_path = study['extension']['storage']
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
        # print(is_success)
        
        storage = study['extension']['storage']
        sum_keys = study["creation"]["config"]["vectors"]["summary"]
        summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
        study = calculate_cost_function(study)(study, get_unit(study), summary)
        # study = calculate_npv(study, get_unit(study), summary)

        out = save_iterations(tr)
        
        return out
    
    elif optimizer == 'COBYLA':
        # define cost function
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function(x, study_path, simulator_path)[0]
        _eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        _ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
        from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, SR1
        eqs = [NonlinearConstraint(eq, 0, 0) for eq in _eqs]
        ineqs = [NonlinearConstraint(eq, 0, np.inf) for eq in _ineqs]
        
        cons = []
        
        for eq in _eqs:
            cons.append({
                'type': 'eq',
                'func': eq
            })
            
        for ineq in _ineqs:
            cons.append({
                'type': 'ineq',
                'func': ineq
            })
        
        
        # define bounds
        bounds = []
        for c in controls:
            try:
                lb = c['lb']
            except:
                lb = -np.inf
                
            try:
                ub = c['ub']
            except:
                ub = np.inf
                
            bounds.append((lb, ub))
            
        def callback(x):
    
            result = cost_function(x, study_path, simulator_path)
            
            global OUT
            OUT['f'].append(result[0])
            OUT['x'].append(x.tolist())
            OUT['v'].append(result[1])
            OUT['v'].append(result[2])
            
            return
            
        global OUT
        OUT = {}
        OUT['f'] = []
        OUT['x'] = []
        OUT['v'] = []
        
        result = minimize(fun = cf, 
                            x0 = x0,
                            method = 'COBYLA', 
                            constraints = eqs + ineqs,
                            bounds=bounds,
                            options={'maxiter': config['optimization']['parameters']['options']['budget']},
                            callback=callback)
    
        
        OUT['nfev'] = result.nfev
        OUT['status'] = result.status
        OUT['message'] = result.message
        OUT['maxcv'] = result.maxcv
        
        return OUT
    
    elif optimizer == 'COBYQA':
        # define cost function
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function(x, study_path, simulator_path)[0]
        _eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        _ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
        from scipy.optimize import LinearConstraint, NonlinearConstraint, SR1
        from cobyqa import minimize
        eqs = [NonlinearConstraint(eq, 0, 0) for eq in _eqs]
        ineqs = [NonlinearConstraint(eq, 0, np.inf) for eq in _ineqs]
        
        cons = []
        
        for eq in _eqs:
            cons.append({
                'type': 'eq',
                'func': eq
            })
            
        for ineq in _ineqs:
            cons.append({
                'type': 'ineq',
                'func': ineq
            })
        
        
        # define bounds
        bounds = []
        for c in controls:
            try:
                lb = c['lb']
            except:
                lb = -np.inf
                
            try:
                ub = c['ub']
            except:
                ub = np.inf
                
            bounds.append((lb, ub))
            
        def callback(x):
    
            result = cost_function(x, study_path, simulator_path)
            
            global OUT_COBYQA
            OUT_COBYQA['f'].append(result[0])
            OUT_COBYQA['x'].append(x.tolist())
            OUT_COBYQA['v'].append(result[1])
            OUT_COBYQA['v'].append(result[2])
            
            return
            
        global OUT_COBYQA
        OUT_COBYQA = {}
        OUT_COBYQA['f'] = []
        OUT_COBYQA['x'] = []
        OUT_COBYQA['v'] = []
        
        opt_constants = config['optimization']['parameters']['constants']
        
        result = minimize(fun = cf, 
                            x0 = x0,
                            constraints = eqs + ineqs,
                            bounds=bounds,
                            options={'maxfev': config['optimization']['parameters']['options']['budget'],
                                     'low_ratio': opt_constants['eta_1'],
                                     'high_ratio': opt_constants['eta_2'],
                                     'very_low_ratio': 1E-16,
                                     'decrease_radius_factor': opt_constants['gamma_0'],
                                     'increase_radius_factor': opt_constants['gamma_2'],
                                     'store_history': True,
                                     'radius_init': opt_constants['init_radius'],
                                     'radius_final': opt_constants['stopping_radius'],
                                     'nb_points': 2*len(x0) + 1, 
                                     'scale': True,
                                     'disp': True},
                            callback=callback)
    
        
        OUT_COBYQA['nfev'] = result.nfev
        OUT_COBYQA['status'] = result.status
        OUT_COBYQA['message'] = result.message
        OUT_COBYQA['maxcv'] = result.maxcv
        OUT_COBYQA['fun_history'] = list(result.fun_history)
        OUT_COBYQA['maxcv_history'] = list(result.maxcv_history)
        
        return OUT_COBYQA
    
    elif optimizer == "NOMAD":
        
        import PyNomad #must install PyNomadBBO package
        
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function(x, study_path, simulator_path)[0]
        _eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        _ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
        # define bb that is compatible with the package
        def bb(x):
            
            f = cf(x)
            
            rawBBO = str(f) + " "
            
            for _ineq in _ineqs:
                rawBBO += str(-_ineq(x)) + " "
            
            x.setBBO(rawBBO.encode("UTF-8"))
            
            return 1
        
        INEQS = ""
        for _ in _ineqs:
            INEQS += "EB "
        
        # define bounds
        lb = [c['lb'] for c in controls]
        ub = [c['ub'] for c in controls]
            
        X0 = x0
        params = [f"DIMENSION {len(x0)}", 
                  f"BB_OUTPUT_TYPE OBJ {INEQS}", 
                  f"MAX_BB_EVAL {config['optimization']['parameters']['options']['budget']}",
                  f"DISPLAY_DEGREE 2",
                  f"DISPLAY_ALL_EVAL true",
                  f"DISPLAY_STATS BBE BBO"]
        
        result = PyNomad.optimize(bb, X0, lb, ub, params)
        
        fmt = ["{} = {}".format(n,v) for (n,v) in result.items()]
        output = "\n".join(fmt)
        print("\nNOMAD results \n" + output + " \n")
        
        # run best points
        print("run the latest ...")
        # cost_function(tr.iterates[-1]["y_curr"], study_path, simulator_path)

        study = u.read_json(study_path)
        config = u.read_json(study['creation']['json'])
        controls = config['controls']
        
        for i, control in enumerate(controls):
            control["Default"] = result['x_best'][i]
        
        simfolder_path = study['extension']['storage']
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
        print(is_success)
        
        storage = study['extension']['storage']
        sum_keys = study["creation"]["config"]["vectors"]["summary"]
        summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
        study = calculate_cost_function(study)(study, get_unit(study), summary)
        # study = calculate_npv(study, get_unit(study), summary)

        OUT = {}
        OUT['x_best'] = result['x_best']
        OUT['f_best'] = result['f_best']
        OUT['nb_evals'] = result['nb_evals']
        OUT['stop_reason'] = result['stop_reason']
        
        return OUT
    
    elif optimizer == "BO":
        
        from bayes_opt import BayesianOptimization
        from scipy.optimize import NonlinearConstraint
                
        pbounds = {f'x_{i}': (c['lb'], c['ub']) for i,c in enumerate(controls)}
        
        # define cost function
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: - (cost_function(x, study_path, simulator_path)[0])
        eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
        
        def constraint_function(**kwargs):
            return np.array([
                ineq(np.array([_x for _,_x in kwargs.items()])) for ineq in ineqs
            ])
        
        constraint = NonlinearConstraint(constraint_function, np.array([0]*len(ineqs)), np.array([np.inf]*len(ineqs)))


        def _cf(**kwargs):
            return cf(np.array([_x for _,_x in kwargs.items()]))


        optimizer = BayesianOptimization(
                            f=_cf,
                            constraint=constraint,
                            pbounds=pbounds,
                            verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                            random_state=1,
                        )
        
        optimizer.maximize(
                    init_points=len(controls),
                    n_iter=config['optimization']['parameters']['options']['budget'],
                )
        
         # run best points
        print("run the latest ...")
        # cost_function(tr.iterates[-1]["y_curr"], study_path, simulator_path)

        study = u.read_json(study_path)
        config = u.read_json(study['creation']['json'])
        controls = config['controls']
        
        for i, control in enumerate(controls):
            control["Default"] = optimizer.max['params'][f"x_{i}"]
        
        simfolder_path = study['extension']['storage']
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
        print(is_success)
        
        storage = study['extension']['storage']
        sum_keys = study["creation"]["config"]["vectors"]["summary"]
        summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
        study = calculate_cost_function(study)(study, get_unit(study), summary)
        # study = calculate_npv(study, get_unit(study), summary)

        OUT = {}

        OUT['best'] = optimizer.max
        OUT['best']['constraint'] = OUT['best']['constraint'].tolist()

        OUT['xs'] = optimizer.res

        for i, xs in enumerate(optimizer.res):
            OUT['xs'][i]['constraint'] = optimizer.res[i]['constraint'].tolist()
            OUT['xs'][i]['allowed'] = bool(optimizer.res[i]['allowed'])
        
        return OUT
    
    elif optimizer == "STOSAG":
        from stosag.stosag import stosag
        from stosag import utilities as stosag_utilities
        
        cf = lambda x, study_path=study_path, simulator_path=simulator_path: cost_function_stosag(x, study_path, simulator_path)[0]
        _eqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function_stosag(x, study_path, simulator_path)[1][i] for i in range(n_eq)]
        _ineqs = [lambda x, study_path=study_path, simulator_path=simulator_path, i=i: cost_function_stosag(x, study_path, simulator_path)[2][i] for i in range(n_ineq)]
        
         # define bounds
        lb = []
        ub = []
        for c in controls:
            try:
                lb.append(c['lb'])
            except:
                lb.append(-np.inf)
                
            try:
                ub.append(c['ub'])
            except:
                ub.append(np.inf)
        
        # ub = [c["ub"] for c in controls]
        
        Ct_parameters = config['optimization']['parameters']['well_covariance_parameters']
        # redefine constants
        Ct = stosag_utilities.create_spherical_covariance_function(
            Ct_parameters['Nw'],
            Ct_parameters['std'],
            Ct_parameters['Ns'],
            Ct_parameters['Nt'],
        )
        
        st = stosag(
            x0=np.array(x0), 
            functions=cf, 
            lb=lb, 
            ub=ub,
            ineqs=[*_ineqs],
            Nens=config["Ne"],
            Ct=Ct,
            constants=config['optimization']['parameters']['constants'],
        )
        
        st.run()
        
        # run best points
        print("run the latest ...")
        cost_function(st.x_list[-1], study_path, simulator_path)

        # xsol = tr.iterates[-1]["y_curr"]
        # xsol = tr.iterates[-1]['best_point']["y"]
        xsol = st.x_list[-1]
        study = u.read_json(study_path)
        config = u.read_json(study['creation']['json'])
        controls = config['controls']
        
        for i, control in enumerate(controls):
            control["Default"] = xsol[i]
        
        simfolder_path = study['extension']['storage']
        realizations, is_success = run_ensemble.run_cases(simulator_path, study, simfolder_path, controls, n_parallel=config['n_parallel'])
        # print(is_success)
        
        storage = study['extension']['storage']
        sum_keys = study["creation"]["config"]["vectors"]["summary"]
        summary = extract_ensemble.get_summary(realizations, storage, sum_keys)
        study = calculate_cost_function(study)(study, get_unit(study), summary)
        # study = calculate_npv(study, get_unit(study), summary)

        OUT = {}
        OUT['x_best'] = st.x_list
        OUT['f_best'] = st.j_list
        OUT['nb_evals'] = st.N_EVAL
        
        return OUT
        
    else:
        raise(f"Optimizer {optimizer} is not supported. Try 'DFTR', 'NOMAD', 'BO' or 'COBYLA'.")
    
    
    

def save_iterations(tr):
                        
    # save iterations
    out = {}
    f = []
    for t in tr.iterates:
        # f.append(t["fY"][0])
        f.append(t["best_point"]["f"])
    out["f"] = f
    
    x = []
    for t in tr.iterates:
        # xs = t["Y"][:,0].tolist()
        # x.append(tr.denorm(np.array(xs)).tolist())
        xs = t["best_point"]["y"].tolist()
        x.append(xs)
    out["x"] = x
    
    neval = []
    for t in tr.iterates:
        neval.append(t["number_of_function_calls"])
    out["neval"] = neval
    
    total_neval = []
    for t in tr.iterates:
        total_neval.append(t["total_number_of_function_calls"])
    out["total_neval"] = total_neval

    # v = []
    # for t in tr.iterates:
    #     v.append(t['all_violations'])
    # out["all_violations"] = v
    
    v = []
    for t in tr.iterates:
        # v.append(list(t['v']))
        v.append(t["best_point"]["v"])
    out["violations"] = v
    
    return out
    
def main(args):
    
    simulator_path = args[0]
    study_path = args[1]
    study = u.read_json(study_path)   
        
    ext_dict = create_extension_folders(study)
    study['extension'] = ext_dict
    u.save_to_json(study_path, study)
    
    study = formulate_problem(study)
    u.save_to_json(study_path, study)
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))
    study['extension']['start'] = dt_start
    out = run_optimization(study_path, simulator_path)
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    study['extension']['end'] = dt_end
    u.save_to_json(study_path, study)
    
    study['extension']['iterations'] = out
    
    study['status'] = "optimized"
    u.save_to_json(study_path, study)
    
    return

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex. python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_PoroPerm_Opt_RandomField.json 
    """
    main(sys.argv[1:])
