import numpy as np
import utils.utilities as u
import sys, os
from datetime import datetime
import math

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
import chaospy
import scipy.optimize
from scipy.optimize import NonlinearConstraint

from utils.es_mda import ESMDA

from typing import Union

def get_static3d(study_path:str) -> np.ndarray:
    study = u.read_json(study_path)
    
    config = u.read_json(study['creation']['json'])
    hm = config['historymatching']
    models = hm['model3d'] #['PORO', 'PERMX', ...]
    
    static3d = []
    for model in models:
        dummy = []
        for casename, path in study['extraction']['static3d'].items():
            path = study['extraction']['static3d'][casename][model]
            p = np.load(path)
            dummy.append(p)
        dummy = np.array(dummy)
        if 'PERM' in model:
            dummy = np.log(dummy)
        
        static3d.append(dummy)
        
    static3d = np.concatenate([*static3d], axis=1)
    
    Ncell = dummy.shape[1]
    
    return static3d, Ncell

def get_summary(study_path:str) -> np.ndarray:
    study = u.read_json(study_path)
    
    config = u.read_json(study['creation']['json'])
    hm = config['historymatching']
    objectives = hm['objectives']
    
    # get history 
    first_realname = list(study['extraction']['summary'].keys())[0] 
    history_years = np.load(config['historymatching']['timestep'])
    
    darray = [] # array of difference of simulation and history (Ne x Nd)
    sim_array = [] # simulation array (Ne x Nd)
    hist_vector = [] # historic vector
    ofs = []
    for kw in objectives.keys():
        base_years = np.load(study['extraction']['summary'][first_realname]['YEARS'])
        
        sum_history = np.load(config['historymatching']['objectives'][kw])
        sum_history = u.resample(base_years, history_years, sum_history)
        
        hist_vector.extend(sum_history)
        
        _dsums = []
        _sim_array = []
        for casename, _ in study['extraction']['summary'].items():
                        
            sum_simulation = np.load(study['extraction']['summary'][casename][kw])
            sum_years = np.load(study['extraction']['summary'][casename]['YEARS'])
            sum_simulation = u.resample(base_years, sum_years, sum_simulation)
            # sum_simulation = u.resample(base_years, sum_years, sum_simulation)

            _sim_array.append(sum_simulation)
            
            dsum = sum_history - sum_simulation
            _dsums.append(dsum)
            
        sim_array.extend(np.array(_sim_array).T)
        
        _dsums = np.array(_dsums)
        darray.append(_dsums)
        
        var = np.var(_dsums, axis=0)
        ind_feasible = var > 0.0 #avoid zero variance
        dsums = np.nansum((_dsums**2)[:,ind_feasible]/var[ind_feasible], axis=1)
        
        ofs.append(dsums)
    darray = np.array(darray)
    sim_array = np.array(sim_array).T
    hist_vector = np.array(hist_vector)
    ofs = np.array(ofs).T
    
    return darray, sim_array, hist_vector

def get_sim_property(study_path):
    
    data = {}
    data['static3d'], data['Ncell'] = get_static3d(study_path)
    data['darray'], data['sim_array'], data['hist_vector'] = get_summary(study_path)
    return data

def run_history_matching(data:dict, params:dict, dim_red_method:Union[PLSRegression, FastICA, PCA]):
    """This part consists of several steps:
        1. Using PLSR, calculate the scores in the projected space
        2. Get the first N_comp = 5 components. These components will be modeled by Data Driven-PCE
        3. Minimize the model
    """

    static3d = data['static3d']
    darray = data['darray']

    ofs = np.hstack(darray)
    std = np.std(ofs, axis=0)
    
    _ofs = ofs.T[std > 0]/std[std > 0][:,np.newaxis]
    
    _ofs = _ofs.T
    
    n_component = params['ncomponent']
    polynomial_order = params['polynomial_order']
    
    method = dim_red_method(n_component)
   
    try:
        scoresX, scoresY = method.fit_transform(static3d, _ofs)
        static3d_hat = method.inverse_transform(scoresX)
    except:
        scoresX = method.fit_transform(static3d, _ofs)
        static3d_hat = method.inverse_transform(scoresX)
    # static3d_hat = np.matmul(scoresX, plsr.x_loadings_.T)
    # static3d_hat *= plsr._x_std
    # static3d_hat += plsr._x_mean
    
    
    diff = static3d - static3d_hat
    
    scoresX_post = scoresX*1
    
    nodes = scoresX[:,:n_component].T
    weights = np.ones(len(ofs))
    J = []
    for j in range(n_component):
        d = chaospy.GaussianKDE(scoresX[:,j], h_mat=1E+1)
        J.append(d)
    joint = chaospy.J(*J)
    
    expansion = chaospy.generate_expansion(polynomial_order, joint, rule="three_terms_recurrence")
    models = chaospy.fit_quadrature(expansion, nodes, weights, _ofs)
            
    min_bounds = np.min(nodes, axis=1)
    max_bounds = np.max(nodes, axis=1)
    bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    x0 = np.mean(scoresX[:,:n_component], axis=0)
    
    def fun(x):
        ultimate_of = 0
        for i, model in enumerate(models):            
            ultimate_of += (model(*x)**2)
            
        return np.sqrt(ultimate_of)
        
    def dummy(x, ii):
        _var = scoresX*1
        _var[:,:n_component] = x
        result = method.inverse_transform(_var)
        return result[ii,:]
    
    cons = []
    for ii in range(static3d.shape[0]):
        # one constraint for each ensemble member
        cons.append(NonlinearConstraint(lambda x, ii=ii : dummy(x, ii), 
                                        np.min(static3d_hat, axis=0), 
                                        np.max(static3d_hat, axis=0)))
    
    print(fun(x0))
    # asdf
    print("Minimizing ...")
    res = scipy.optimize.minimize(fun, 
                                x0, 
                                args=(), 
                                method=None, 
                                jac=None, 
                                hess=None, 
                                hessp=None, 
                                bounds=bounds, 
                                constraints=cons,
                                tol=None, 
                                callback=None, 
                                options=None)
    print("Done!")
    
    scoresX_post[:,:n_component] = res.x

    # Transform to the original space
    static3d_post = method.inverse_transform(scoresX_post)
    static3d_post = diff + static3d_post
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
    
    return np.array(static3d_post) 

def run_esmda(data, params):
    
    static3d = data['static3d']
    sim_array = data['sim_array']
    hist_vector = data['hist_vector']
    
    esmda = ESMDA(m=static3d,
                  g_func=None,
                  g_obs=hist_vector,
                  alphas=[1],
                  cd=np.ones(hist_vector.shape[0]))
    
    # plsr = PLSRegression(static3d.shape[1])
    plsr = PLSRegression(params['ncomponent'])
    scoresX, scoresY = plsr.fit_transform(static3d, sim_array)
    diff = static3d - plsr.inverse_transform(scoresX)

    _, hist_scoresY = plsr.transform(static3d, np.array([hist_vector]))
    
    var = np.var(scoresY, axis=0)
    ind_feasible = var > 0.0
    
    var = var[ind_feasible]
    g = scoresY[:,ind_feasible]
    g_obs = hist_scoresY[0,ind_feasible]
    
    scoresX_post = esmda.update(m = scoresX,
                 g = g, 
                 g_obs = g_obs,
                 alpha = params['alpha'],
                 cd = 1/var) #cd = np.ones(len(hist_vector))*100)
    
    static3d_post = plsr.inverse_transform(scoresX_post)
    static3d_post = static3d_post + diff
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
    
    
    return static3d_post

def run_pcesmda(data, params):
    
    """
    Steps: 
    1. Generate a surrogate model g based on the training data of M
    2. Create a helper function that reduce the order model M to X, and D to Y
    3. Call the ESMDA class and update in the full-order space in M but reduced-order space in Y
    """
    static3d = data['static3d']
    sim_array = data['sim_array']
    hist_vector = data['hist_vector']
    darray = data['darray']
    n_component = params['ncomponent']
    polynomial_order = params['polynomial_order']

    ofs = np.hstack(darray)
    std = np.std(ofs, axis=0)
    
    _ofs = ofs.T[std > 0]/std[std > 0][:,np.newaxis]
    _ofs = _ofs.T
    
    
    def pce_model(m_train):
        plsr = PLSRegression(params['ncomponent'])
        scoresX, scoresY = plsr.fit_transform(m_train, sim_array)
        
        nodes = scoresX[:,:n_component].T
        weights = np.ones(len(ofs))
        J = []
        for j in range(n_component):
            d = chaospy.GaussianKDE(scoresX[:,j], h_mat=1E+1)
            J.append(d)
        joint = chaospy.J(*J)
        
        expansion = chaospy.generate_expansion(polynomial_order, joint, rule="three_terms_recurrence")
        models = chaospy.fit_quadrature(expansion, nodes, weights, scoresY)
        
        return models, plsr
    
    models, plsr = pce_model(static3d)
    _, hist_scoresY = plsr.transform(static3d, np.array([hist_vector]))
    
    def g_func(m):
        
        x = plsr.transform(np.array([m]))[0]
        y = []
        for model in models:
            evaluation = model(*x)
            y.append(evaluation)
        return np.array(y)

    esmda = ESMDA(m=static3d,
                  g_func=g_func,
                  g_obs=hist_scoresY[0],
                  alphas=[9.333, 7.0, 4.0, 2.0],
                  cd=np.ones(hist_scoresY.shape[1]))
    
    static3d_post, _ = esmda.run()
    
    # static3d_post = plsr.inverse_transform(scoresX_post)
    # static3d_post = static3d_post + diff
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
    
    
    return static3d_post


def save_posterior(static3d_post, study_path):
    
    study = u.read_json(study_path)
    
    config = study['creation']['config']
    hm = config['historymatching']
    updatepath = hm['updatepath']
    
    if os.path.isdir(updatepath):
        pass
    else:
        os.mkdir(updatepath)
    
    models = hm['model3d']
    
    Ncell = int(static3d_post.shape[1]/len(models))
    
    posterior_paths = {}
    for i in range(config['Ne']):
        posterior_paths[f'Realization_{int(i)}'] = {}
        for j, model in enumerate(models):
            
            prop = static3d_post[:, j*Ncell:(j+1)*Ncell]
            if 'PERM' in model:
                prop = np.exp(prop)
            
            filename = os.path.join(updatepath, f'{model}_{i+1}.npy')
            np.save(filename, prop[i,:])
            
            posterior_paths[f'Realization_{int(i)}'][model] = filename
    
    return posterior_paths
    
def main(argv):
    
    study_path = argv[0]
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_start = str(datetime.fromtimestamp(timestamp))

    study = u.read_json(study_path)
    data = get_sim_property(study_path)
    
    try:
        method = argv[1]
    except IndexError:
        method = study['creation']['config']['historymatching']['method']
        pass
    
    # Number of principal components
    try: 
        ncomponent = int(argv[2])
    except IndexError:
        ncomponent = 15
        
    try:
        hyperparameter = float(argv[3])
    except IndexError:
        if method == 'PLSR':
            hyperparameter = 3
        elif method == 'FICA':
            hyperparameter = 3 
        elif method == 'ESMDA':
            hyperparameter = 4.0
        elif method == 'PCESMDA':
            hyperparameter = 3
    
    if method == "PLSR":
        params = {
                  'ncomponent': ncomponent,
                  'polynomial_order': int(hyperparameter)
                  }
        static3d_post = run_history_matching(data, params, PLSRegression)
    
    elif method == "FICA":
        params = {
                  'ncomponent': ncomponent,
                  'polynomial_order': int(hyperparameter)
                  }
        static3d_post = run_history_matching(data, params, FastICA)
        
    elif method == "PCA":
        params = {
                  'ncomponent': ncomponent,
                  'polynomial_order': int(hyperparameter)
                  }
        static3d_post = run_history_matching(data, params, PCA)
        
    elif method == "ESMDA":
        params = {'ncomponent': ncomponent, 
                  'alpha': hyperparameter}
        static3d_post = run_esmda(data, params)
        
    elif method == 'PCESMDA':
        params = {'ncomponent': ncomponent, 
                  'polynomial_order': int(hyperparameter)}
        static3d_post = run_pcesmda(data, params)
        
    posterior_paths = save_posterior(static3d_post, study_path)
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    
    
    study['status'] = 'history matched'
    study['historymatched'] = {}
    study["historymatched"]["start"] = dt_start
    study["historymatched"]["end"] = dt_end
    study['historymatched']['paths'] = posterior_paths
    u.save_to_json(study_path, study)
    
    return

if __name__ == "__main__":
    """The arguments are the following:
    1. study path (str): path to a study json file
    
    Ex. python3 src/hm_ensemble.py simulations/studies/IE_PoroPerm2_RF.json 
    """
    print("HISTORY MATCHING")
    main(sys.argv[1:])
