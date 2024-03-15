import json
import numpy as np
import scipy.stats
import gstools as gs
from functools import lru_cache, wraps
from subprocess import DEVNULL

import scipy.interpolate as interp

import subprocess
import time
from collections import deque
from tqdm import tqdm 

import csv


def run_bash_commands_in_parallel(commands, max_tries, n_parallel):
    """
    Run a list of bash commands in parallel with maximum number of processes
    https://stackoverflow.com/questions/69009892/python-run-multiple-subprocesses-in-parallel-but-allow-to-retry-when-a-bash-co
    """
    # we use a tuple of (command, tries) to store the information
    # how often a command was already tried.
    
    waiting = deque([(command, 1, i) for i, command in enumerate(commands)]) #(command, #of tries, realizationindex)
    running = deque()
    is_success = [False]*len(waiting) # list of whether the simulation is a success or not
    
    pbar = tqdm(total=len(waiting), disable=True)
    while len(waiting) > 0 or len(running) > 0:

        # if less than n_parallel jobs are running and we have waiting jobs,
        # start new jobs
        while len(waiting) > 0 and len(running) < n_parallel:
            command, tries, index = waiting.popleft()
            try:
                running.append((subprocess.Popen(command, stdout=DEVNULL), command, tries, index))
                # print(f"Started task {command}. Running: {len(running)}, Waiting: {len(waiting)}")
                pbar.set_description(f"Running: {len(running)}, Waiting: {len(waiting)}")
            except OSError:
                print(f'Failed to start command {command}')

        # poll running commands 
        for _ in range(len(running)):
            process, command, tries, index = running.popleft()
            ret = process.poll()
            if ret is None:
                running.append((process, command, tries, index))
            # retry errored jobs
            elif ret != 0:
                if tries < max_tries:
                    waiting.append((command, tries  + 1))
                else:
                    print(f'Command: {command} errored after {max_tries} tries')
                    pbar.update(1)
            else:
                # print(f'Command {command} finished successfully')
                is_success[index] = True
                pbar.update(1)
                
        # sleep a bit to reduce CPU usage
        time.sleep(0.5)
    pbar.set_description(f'All simulation done')
    pbar.close()
    return is_success

def hashable_lru(func):
    cache = lru_cache(maxsize=1024)

    def deserialise(value):
        try:
            return json.loads(value)
        except Exception:
            return value

    def func_with_serialized_params(*args, **kwargs):
        _args = tuple([deserialise(arg) for arg in args])
        _kwargs = {k: deserialise(v) for k, v in kwargs.items()}
        return func(*_args, **_kwargs)

    cached_function = cache(func_with_serialized_params)

    @wraps(func)
    def lru_decorator(*args, **kwargs):
        _args = tuple([json.dumps(arg, sort_keys=True) if type(arg) in (list, dict) else arg for arg in args])
        _kwargs = {k: json.dumps(v, sort_keys=True) if type(v) in (list, dict) else v for k, v in kwargs.items()}
        return cached_function(*_args, **_kwargs)
    lru_decorator.cache_info = cached_function.cache_info
    lru_decorator.cache_clear = cached_function.cache_clear
    return lru_decorator

def np_cache(function):
    @lru_cache
    def cached_wrapper(*args, **kwargs):

        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
    
def resample(base_date, custom_date, data):
    
    f = interp.interp1d(custom_date, data, fill_value="extrapolate")
    new_data = f(base_date)
    return new_data

def read_json(jsonfilename):
    with open(jsonfilename, 'r') as j:
        jsondata = json.loads(j.read())
        
    return jsondata

def save_to_json(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f, indent=4)
        
def replace_single_value(d:dict):
    
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
        replaced_value = '%.5f'%sample
    elif p['type'] == "int":
        replaced_value = '%s'%int(sample)
            
    return replaced_value

def replace_random_field(d:dict):
    
    p = d["parameters"]
    size = d["size"]
    x = np.arange(-0.5, size[0], 1)
    y = np.arange(-0.5, size[1], 1)
    z = np.arange(-0.5, size[2], 1)
    
    scale = d["scale"]
    angles = d["angles"]
    if d["name"] == "Normal":
        model = gs.Gaussian(dim=3, var=1, len_scale=scale, angles=angles)
        srf = gs.SRF(model)
        srf((x, y, z), mesh_type='structured')
        
        fieldcdf = scipy.stats.norm.cdf(srf.field[:-1, :-1, :-1], 0, 1)
        
        a = (p['min'] - p['mean']) / p['std']
        b = (p['max'] - p['mean']) / p['std']
        var = scipy.stats.truncnorm.ppf(fieldcdf, a, b)
        
        var = var*p["std"] + p["mean"]
        var = np.reshape(var, (size[0]*size[1]*size[2]), order="F")
    
    elif d["name"] == "LogNormal":
        model = gs.Gaussian(dim=3, var=1, len_scale=scale, angles=angles)
        srf = gs.SRF(model)
        srf((x, y, z), mesh_type='structured')
        # print(srf.field)
        fieldcdf = scipy.stats.norm.cdf(srf.field[:-1, :-1, :-1], 0, 1)
        
        a = (np.log(p['min']) - np.log(p['mean'])) / np.log(p['std'])
        b = (np.log(p['max']) - np.log(p['mean'])) / np.log(p['std'])
        var = scipy.stats.truncnorm.ppf(fieldcdf, a, b)
        
        var = np.exp(var*np.log(p["std"]) + np.log(p["mean"]))
        var = np.reshape(var, (size[0]*size[1]*size[2]), order="F")
        
    elif d["name"] == "LogNormal-ArcSin":
        from gstools import transform as tf
        model = gs.Gaussian(dim=3, var=1, len_scale=scale, angles=angles)
        srf = gs.SRF(model)
        srf((x, y, z), mesh_type='structured')
        # print(srf.field)
        tf.normal_to_arcsin(srf)
        fieldcdf = scipy.stats.norm.cdf(srf.field[:-1, :-1, :-1], 0, 1)
        
        a = (np.log(p['min']) - np.log(p['mean'])) / np.log(p['std'])
        b = (np.log(p['max']) - np.log(p['mean'])) / np.log(p['std'])
        var = scipy.stats.truncnorm.ppf(fieldcdf, a, b)
        
        var = np.exp(var*np.log(p["std"]) + np.log(p["mean"]))
        var = np.reshape(var, (size[0]*size[1]*size[2]), order="F")
    
    elif d["name"] == 'Constant':
        var = [p['value']]*(size[0]*size[1]*size[2])
        var = np.array(var)

    else:
        raise ValueError("%s distribution not implemented yet" %d["name"])
    
    replaced_value = ''
    if p['type'] == "float":
        for v in var:
            replaced_value += '%.5f '%v
    elif p['type'] == "int":
        for v in var:
            replaced_value += '%s '%int(v)
    
    return replaced_value

def replace_incremental_text(d:dict, case_number:int):
    p = d["parameters"]
    prefix = p['prefix'] 
    suffix = p['suffix']
    replaced_value = "'"+str(prefix + str(case_number) + suffix)+"'"
    return replaced_value

def replace_incremental_value(d:dict, case_number:int):
    
    p = d["parameters"]
    prefix = p['prefix'] 
    suffix = p['suffix']
    
    # csv_file_path = str(prefix + str(case_number) + suffix)
    
    # with open(csv_file_path, 'r') as f:
    #     reader = csv.reader(f)
    #     rows = list(reader)
    
    # row = rows[0]
    
    file_path = str(prefix + str(case_number) + suffix)
    
    row = np.load(file_path)
    replaced_value = ''
    for v in row:
        replaced_value += '%.5f '%float(v)
    
    return replaced_value