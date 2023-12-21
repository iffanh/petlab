import numpy as np
import utils.utilities as u
import sys, os
from datetime import datetime
import math

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import chaospy
import scipy.optimize
from scipy.optimize import NonlinearConstraint

from utils.es_mda import ESMDA

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

def run_history_matching(data:dict, params:dict):
    """This part consists of several steps:
        1. Using PLSR, calculate the scores in the projected space
        2. Get the first N_comp = 5 components. These components will be modeled by Data Driven-PCE
        3. Minimize the model
    """

    static3d = data['static3d']
    darray = data['darray']
    
    # For each ensemble, calculate the sum of all the mismatch given a vector of objective function
    ofs = []
    for _dsums in darray:
        var = np.var(_dsums, axis=0)
        ind_feasible = var > 0.0
        dsums = np.nansum((_dsums**2)[:,ind_feasible]/var[ind_feasible], axis=1)
        ofs.append(dsums)
        
    ofs = np.array(ofs).T
    
    scaler = StandardScaler()
    n_ofs = scaler.fit_transform(ofs)

    n_component = params['ncomponent']
    plsr = PLSRegression(n_component)
   
    scoresX, scoresY = plsr.fit_transform(static3d, ofs)
    static3d_hat = np.matmul(scoresX, plsr.x_loadings_.T)
    static3d_hat *= plsr._x_std
    static3d_hat += plsr._x_mean

    diff = static3d - static3d_hat

    
    total_variance_in_x = np.sum(np.var(static3d, axis=0))
    variance_in_x = np.var(plsr.x_scores_, axis=0)
    fractions_of_explained_variance = variance_in_x/total_variance_in_x
    fractions_of_explained_variance = fractions_of_explained_variance/np.sum(fractions_of_explained_variance)

    # n_component = [i for i, v in enumerate(np.cumsum(fractions_of_explained_variance)) if v > 0.75][0]

    models = []
    nodes = scoresX[:,:n_component].T
    weights = np.ones(len(ofs))
    for j in range(n_component):
        d = chaospy.GaussianKDE(scoresX[:,j])
        expansion = chaospy.generate_expansion(3, d, rule="three_terms_recurrence")
        # model = chaospy.fit_quadrature(expansion, nodes, weights, ofs)
        model = chaospy.fit_quadrature(expansion, nodes, weights, n_ofs)
        models.append(model)
        
    model = lambda x: np.sum([(fractions_of_explained_variance[i])*model(x[i]) for i, model in enumerate(models)])
        # model = lambda x: np.sum([model(x[i]) for i, model in enumerate(models)])
         
    min_bounds = np.min(nodes, axis=1)
    max_bounds = np.max(nodes, axis=1)
    bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    x0 = np.mean(scoresX[:,:n_component], axis=0)
    def fun(x):
        return model(x)
        
    def dummy(x, ii):
        _var = scoresX*1
        _var[:,:n_component] = x
        result = plsr.inverse_transform(_var)
        return result[ii,:]
    
    cons = []
    for ii in range(static3d.shape[0]):
        # one constraint for each ensemble member
        cons.append(NonlinearConstraint(lambda x, ii=ii : dummy(x, ii), 
                                        np.min(static3d_hat, axis=0), 
                                        np.max(static3d_hat, axis=0)))
    
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
    scoresX_post = scoresX*1
    
    scoresX_post[:,:n_component] = res.x

    # Transform to the original space
    static3d_post = plsr.inverse_transform(scoresX_post)
    static3d_post = diff + static3d_post
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]

    return static3d_post 


def run_plsr(data:dict, params:dict):
    """This part consists of several steps:
        1. Using PLSR, calculate the scores in the projected space
        2. Get the first N_comp = 5 components. These components will be modeled by Data Driven-PCE
        3. Minimize the model
    """

    static3d = data['static3d']
    darray = data['darray']
    
    # For each ensemble, calculate the sum of all the mismatch given a vector of objective function
    ofs = []
    for _dsums in darray:
        var = np.var(_dsums, axis=0)
        ind_feasible = var > 0.0
        dsums = np.nansum((_dsums**2)[:,ind_feasible]/var[ind_feasible], axis=1)
        ofs.append(dsums)
        
    ofs = np.array(ofs).T
    
    scaler = StandardScaler()
    n_ofs = scaler.fit_transform(ofs)

    n_component = params['ncomponent']
    plsr = PLSRegression(n_component)
   
    scoresX, scoresY = plsr.fit_transform(static3d, ofs)
    static3d_hat = np.matmul(scoresX, plsr.x_loadings_.T)
    static3d_hat *= plsr._x_std
    static3d_hat += plsr._x_mean

    diff = static3d - static3d_hat

    
    total_variance_in_x = np.sum(np.var(static3d, axis=0))
    variance_in_x = np.var(plsr.x_scores_, axis=0)
    fractions_of_explained_variance = variance_in_x/total_variance_in_x
    fractions_of_explained_variance = fractions_of_explained_variance/np.sum(fractions_of_explained_variance)

    # n_component = [i for i, v in enumerate(np.cumsum(fractions_of_explained_variance)) if v > 0.75][0]

    models = []
    priors = []
    nodes = scoresX[:,:n_component].T
    weights = np.ones(len(ofs))
    
    for j in range(n_component):
        d = chaospy.GaussianKDE(scoresX[:,j])
        expansion = chaospy.generate_expansion(3, d, rule="three_terms_recurrence")
        # print(ofs.shape)
        model = chaospy.fit_quadrature(expansion, nodes[j,:], weights, n_ofs)

        priors.append(d)
        models.append(model)
         
    # print(models)
    min_bounds = np.min(nodes, axis=1)
    max_bounds = np.max(nodes, axis=1)
    bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    
    scoresX_post = scoresX*1
    for jj, (model, prior) in enumerate(zip(models, priors)):
        
        def fun(x):
            # calculate posterior pdf
            
            # post_model = lambda x : np.matmul((scoresY[:,jj] - model(x)), (scoresY[:,jj] - model(x)).T)/np.std(scoresY[:,jj]) + \
            #                         np.matmul((scoresX[:,jj] - x), (scoresX[:,jj] - x).T)/np.std(scoresX[:,jj])
                    
            # return post_model(x)
            return np.mean(model(x))
            
        def dummy(x, ii):
            _var = scoresX*1
            _var[:,jj] = x
            result = plsr.inverse_transform(_var)
            return result[ii,:]
        
        cons = []
        for ii in range(static3d.shape[0]):
            # one constraint for each ensemble member
            cons.append(NonlinearConstraint(lambda x, ii=ii : dummy(x, ii), 
                                            np.min(static3d_hat, axis=0), 
                                            np.max(static3d_hat, axis=0)))

        x0 = np.mean(scoresX[:,jj])
        print(x0)
        print("Minimizing ...")
        res = scipy.optimize.minimize(fun, 
                                    x0, 
                                    args=(), 
                                    method=None, 
                                    jac=None, 
                                    hess=None, 
                                    hessp=None, 
                                    bounds=None, 
                                    constraints=cons,
                                    tol=None, 
                                    callback=None, 
                                    options=None)
        print("Done!")
        scoresX_post[:,jj] = scoresX_post[:,jj] + res.x - x0

    print('YEAYYY')
    # Transform to the original space
    static3d_post = plsr.inverse_transform(scoresX_post)
    static3d_post = diff + static3d_post
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
        
    return static3d_post 

def run_fica(data:dict, params:dict):
    """This part consists of several steps:
        1. Using FICA, calculate the scores in the projected space
        2. Get the first N_comp = 5 components. These components will be modeled by Data Driven-PCE
        3. Minimize the model
    """

    static3d = data['static3d']
    darray = data['darray']
    
    # For each ensemble, calculate the sum of all the mismatch given a vector of objective function
    ofs = []
    for _dsums in darray:
        var = np.var(_dsums, axis=0)
        ind_feasible = var > 0.0
        dsums = np.nansum((_dsums**2)[:,ind_feasible]/var[ind_feasible], axis=1)
        ofs.append(dsums)
        
    ofs = np.array(ofs).T
    
    scaler = StandardScaler()
    n_ofs = scaler.fit_transform(ofs)

    n_component = params['ncomponent']
    fica = FastICA(n_component)
   
    scoresX = fica.fit_transform(static3d)
    static3d_hat = fica.inverse_transform(scoresX)
    # static3d_hat = np.matmul(scoresX, plsr.x_loadings_.T)
    # static3d_hat *= plsr._x_std
    # static3d_hat += plsr._x_mean

    
    
    diff = static3d - static3d_hat
    print(diff)
    
    total_variance_in_x = np.sum(np.var(static3d, axis=0))
    variance_in_x = np.var(scoresX, axis=0)
    fractions_of_explained_variance = variance_in_x/total_variance_in_x
    fractions_of_explained_variance = fractions_of_explained_variance/np.sum(fractions_of_explained_variance)

    # n_component = [i for i, v in enumerate(np.cumsum(fractions_of_explained_variance)) if v > 0.75][0]

    #### ORIGINAL ####
    # models = []
    # nodes = scoresX[:,:n_component].T
    # weights = np.ones(len(ofs))

    # for j in range(n_component):
    #     d = chaospy.GaussianKDE(scoresX[:,j])
    #     expansion = chaospy.generate_expansion(3, d, rule="three_terms_recurrence")
    #     model = chaospy.fit_quadrature(expansion, nodes, weights, ofs)
    #     # model = chaospy.fit_quadrature(expansion, nodes, weights, n_ofs)
    #     models.append(model)
    #     # model = lambda x: np.sum([(fractions_of_explained_variance[i])*model(x[i]) for i, model in enumerate(models)])
    #     model = lambda x: np.sum([(fractions_of_explained_variance[i])*model(x[i])*d.pdf(x[i]) for i, model in enumerate(models)])
    #     # model = lambda x: np.sum([model(x[i]) for i, model in enumerate(models)])
         
    # min_bounds = np.min(nodes, axis=1)
    # max_bounds = np.max(nodes, axis=1)
    # bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    # x0 = np.mean(scoresX[:,:n_component], axis=0)
    # def fun(x):
    #     return np.log(model(x))
        
    # # res = scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=bounds, constraints=(), tol=None, callback=None, options=None)
    
    # def dummy(x, ii):
    #     _var = scoresX*1
    #     _var[:,:n_component] = x
    #     result = fica.inverse_transform(_var)
    #     return result[ii,:]
    
    # cons = []
    # for ii in range(static3d.shape[0]):
    #     # one constraint for each ensemble member
    #     cons.append(NonlinearConstraint(lambda x, ii=ii : dummy(x, ii), 
    #                                     np.min(static3d_hat, axis=0), 
    #                                     np.max(static3d_hat, axis=0)))
    
    # print("Minimizing ...")
    # res = scipy.optimize.minimize(fun, 
    #                               x0, 
    #                               args=(), 
    #                               method=None, 
    #                               jac=None, 
    #                               hess=None, 
    #                               hessp=None, 
    #                             #   bounds=bounds,
    #                               bounds=None, 
    #                               constraints=cons,
    #                               tol=None, 
    #                               callback=None, 
    #                               options=None)
    # print("Done!")
    # scoresX_post = scoresX*1
    
    # scoresX_post[:,:n_component] = res.x

    # # Transform to the original space
    # static3d_post = fica.inverse_transform(scoresX_post)
    # static3d_post = diff + static3d_post
    
    # for j in range(static3d_post.shape[0]):
    #     violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
    #     static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
    #     violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
    #     static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
        
    # ### END ####
    
    models = []
    nodes = scoresX[:,:n_component].T
    weights = np.ones(len(ofs))

    for j in range(n_component):
        d = chaospy.GaussianKDE(scoresX[:,j])
        expansion = chaospy.generate_expansion(3, d, rule="three_terms_recurrence")
        model = chaospy.fit_quadrature(expansion, nodes, weights, ofs)
        # model = chaospy.fit_quadrature(expansion, nodes, weights, n_ofs)
        models.append(model)
        # model = lambda x: np.sum([(fractions_of_explained_variance[i])*model(x[i]) for i, model in enumerate(models)])
        model = lambda x: np.sum([(fractions_of_explained_variance[i])*model(x[i]) for i, model in enumerate(models)])
        # model = lambda x: np.sum([model(x[i]) for i, model in enumerate(models)])
         
    min_bounds = np.min(nodes, axis=1)
    max_bounds = np.max(nodes, axis=1)
    bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    x0 = np.mean(scoresX[:,:n_component], axis=0)
    def fun(x):
        return np.log(model(x))
        
    # res = scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=bounds, constraints=(), tol=None, callback=None, options=None)
    
    def dummy(x, ii):
        _var = scoresX*1
        _var[:,:n_component] = x
        result = fica.inverse_transform(_var)
        return result[ii,:]
    
    cons = []
    for ii in range(static3d.shape[0]):
        # one constraint for each ensemble member
        cons.append(NonlinearConstraint(lambda x, ii=ii : dummy(x, ii), 
                                        np.min(static3d_hat, axis=0), 
                                        np.max(static3d_hat, axis=0)))
    
    print("Minimizing ...")
    res = scipy.optimize.minimize(fun, 
                                  x0, 
                                  args=(), 
                                  method=None, 
                                  jac=None, 
                                  hess=None, 
                                  hessp=None, 
                                #   bounds=bounds,
                                  bounds=None, 
                                  constraints=cons,
                                  tol=None, 
                                  callback=None, 
                                  options=None)
    print("Done!")
    scoresX_post = scoresX*1
    
    scoresX_post[:,:n_component] = res.x

    # Transform to the original space
    static3d_post = fica.inverse_transform(scoresX_post)
    static3d_post = diff + static3d_post
    
    for j in range(static3d_post.shape[0]):
        violated_cell_index = static3d_post[j,:] < np.min(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.min(static3d, axis=0)[violated_cell_index]
        
        violated_cell_index = static3d_post[j,:] > np.max(static3d, axis=0)
        static3d_post[j,:][violated_cell_index] = np.max(static3d, axis=0)[violated_cell_index]
        
    
    
    return static3d_post 

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
    
    
    print(g.shape, g_obs.shape, scoresX.shape)
    
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
        alpha = float(argv[3])
    except IndexError:
        if method == 'PLSR':
            alpha = None
        elif method == 'FICA':
            alpha = None 
        elif method == 'ESMDA':
            alpha = 4.0
    
    if method == "PLSR":
        params = {'ncomponent': ncomponent}
        static3d_post = run_history_matching(data, params)
        # static3d_post = run_plsr(data, params)
    
    elif method == "FICA":
        params = {'ncomponent': ncomponent, 
                  'alpha': alpha}
        static3d_post = run_fica(data, params)
        
    elif method == "ESMDA":
        params = {'ncomponent': ncomponent, 
                  'alpha': alpha}
        static3d_post = run_esmda(data, params)
        
        
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
