import numpy as np
import utils.utilities as u
import sys, os
from datetime import datetime

from sklearn.cross_decomposition import PLSRegression
import chaospy
import scipy.optimize

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
    
    dsums = {}
    for casename, _ in study['extraction']['summary'].items():
        base_years = np.load(study['extraction']['summary'][first_realname]['YEARS'])
        dsums[casename] = []
        
        for kw in objectives.keys():
            sum_history = np.load(config['historymatching']['objectives'][kw])
            sum_history = u.resample(base_years, history_years, sum_history)
                        
            sum_simulation = np.load(study['extraction']['summary'][casename][kw])
            sum_years = np.load(study['extraction']['summary'][casename]['YEARS'])
            sum_simulation = u.resample(base_years, sum_years, sum_simulation)
        
            dsum = sum_history - sum_simulation
            dsums[casename].extend(dsum)
        
        dsums[casename] = np.array(dsums[casename])
            
    # calculate objective function
    ofs = []
    for i, (casename, _) in enumerate(study['extraction']['summary'].items()):
        of = np.matmul(dsums[casename], dsums[casename])/len(dsums[casename])
        ofs.append(of)
        
    ofs = np.log(np.array(ofs))
    
    return ofs

def get_sim_property(study_path):
    
    data = {}
    data['static3d'], data['Ncell'] = get_static3d(study_path)
    data['ofs'] = get_summary(study_path)
    return data

def run_history_matching(data:dict):
    """This part consists of several steps:
        1. Using PLSR, calculate the scores in the projected space
        2. Get the first N_comp = 5 components. These components will be modeled by Data Driven-PCE
        3. Minimize the model
    """

    static3d = data['static3d']
    ofs = data['ofs']
    plsr = PLSRegression(static3d.shape[1])
    scoresX, scoresY = plsr.fit_transform(static3d, ofs)
    
    N_comp = 5
    distributions = []
    for j in range(N_comp):
        d = chaospy.GaussianKDE(scoresX[:,j])
        distributions.append(d)
        
    J = chaospy.J(*distributions)
    expansion = chaospy.generate_expansion(3, J, rule="cholesky")
    nodes = scoresX[:,:N_comp].T
    weights = np.ones(len(ofs))
    model = chaospy.fit_quadrature(expansion, nodes, weights, ofs)
    
    min_bounds = np.min(nodes, axis=1)
    max_bounds = np.max(nodes, axis=1)
    bounds = [(a, b) for a, b in zip(min_bounds, max_bounds)]
    
    x0 = np.zeros(N_comp)
    def fun(x):
        return model(*x)

    res = scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=bounds, constraints=(), tol=None, callback=None, options=None)
    scoresX_post = scoresX*1
    
    scoresX_post[:,:N_comp] = res.x

    # Transform to the original space
    static3d_post, _ = plsr.inverse_transform(scoresX_post, scoresY)
    return static3d_post 

def save_posterior(static3d_post, study_path):
    
    study = u.read_json(study_path)
    
    config = study['creation']['config']
    hm = config['historymatching']
    updatepath = hm['updatepath']
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

    data = get_sim_property(study_path)
    static3d_post = run_history_matching(data)
    posterior_paths = save_posterior(static3d_post, study_path)
    
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_end = str(datetime.fromtimestamp(timestamp))
    
    study = u.read_json(study_path)
    
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
    main(sys.argv[1:])
