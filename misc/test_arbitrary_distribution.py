import chaospy
import numpy as np
import os, json
import scipy.interpolate as interp
import sklearn.cross_decomposition

def read_json(jsonfilename):
    with open(jsonfilename, 'r') as j:
        jsondata = json.loads(j.read())
        
    return jsondata

def resample(base_date, custom_date, data):
    
    f = interp.interp1d(custom_date, data)
    new_data = f(base_date)
    return new_data

# def f(x1, x2):
#     return x1**2 + x2**2

# samples1 = np.random.random(100)
# samples2 = np.random.random(100)

# d1 = chaospy.GaussianKDE(samples1)
# d2 = chaospy.GaussianKDE(samples2)

# J = chaospy.J(d1, d2)

# expansion = chaospy.generate_expansion(3, J, rule="cholesky")

# gauss_quads = chaospy.generate_quadrature(3, J, rule="gaussian") #(order of quadrature, joint prob, rule)
# nodes, weights = gauss_quads


# evals = f(nodes[0,:], nodes[1,:])
# print(evals.shape)
# model = chaospy.fit_quadrature(expansion, nodes, weights, evals)


study_path = f"./simulations/studies/IE_PoroPerm2_RF.json"
study = read_json(study_path)

ensemble_storage = study['simulation']['storage']

# get porosity

poro = []
for casename, path in study['extraction']['static3d'].items():
    poro_path = study['extraction']['static3d'][casename]['PORO']
    p = np.load(poro_path)
    poro.append(p)
poro = np.array(poro)

fakehistory_path = f'./misc/fakehistory/SPE1_RF_3/'
# history_kws = ['WBHP:INJ', 'WBHP:PROD']
history_kws = ['WBHP:INJ']


history_years = np.load(os.path.join(fakehistory_path, 'summary', 'YEARS.npy'))

# get error terms
for kw in history_kws:
    sum_history = np.load(os.path.join(fakehistory_path, 'summary', kw + '.npy'))
    
    first_realname = list(study['extraction']['summary'].keys())[0] 
    base_years = np.load(study['extraction']['summary'][first_realname]['YEARS'])
        
    sum_history = resample(base_years, history_years, sum_history)
    
    dsums = []
    for casename, path in study['extraction']['summary'].items():
        sum_simulation = np.load(study['extraction']['summary'][casename][kw])
        sum_years = np.load(study['extraction']['summary'][casename]['YEARS'])
        sum_simulation = resample(base_years, sum_years, sum_simulation)
        
        dsum = sum_history - sum_simulation
        dsums.append(dsum)
        
    dsums = np.array(dsums)
    
objective_function = np.matmul(dsums[0,:].T, dsums[0,:])/len(dsums[0,:])
