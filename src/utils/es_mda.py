import numpy as np

def calculate_covariance(matrix1:np.ndarray, matrix2:np.ndarray):

    def shifted_mean(matrix):
        meanM = np.mean(matrix, axis=0)
        dmatrix = matrix - meanM
        return dmatrix
    
    """
    Function responsible for calculating covariance between two matrices of ensembles. 

    matrix1: Ne x a
    matrix2: Ne x b
    """

    assert matrix1.shape[0] == matrix2.shape[0]

    dmatrix1 = shifted_mean(matrix1)
    dmatrix2 = shifted_mean(matrix2)
    
    cov = 0
    Ne = matrix1.shape[0]
    for i in range(Ne):
        cov += np.outer(dmatrix1[i,:], dmatrix2[i,:])

    cov = cov/(Ne-1)

    return cov

def update_member(m_prior:np.ndarray, g_prior:np.ndarray, g_obs:np.ndarray, cov_mg:np.ndarray, cov_gg:np.ndarray, alpha:float, cd:np.ndarray):

    """
    Update equation per ensemble member

    m_prior     : matrix of prior parameters (Ne x a)
    g_prior     : matrix of prior output (Ne x b)
    g_obs       : vector of (perturbed) observation (1 x b)
    cov_mg      : matrix of covariance between parameters and output (a x b)
    cov_gg      : matrix of output auto-covariance (b x b)
    alpha       : es-mda sub-step
    cd          : vector of measurement error (1 x b)
    """
    print(cov_mg.shape, cov_gg.shape)
    K = np.matmul(cov_mg,np.linalg.pinv(cov_gg + alpha*np.diag(cd))) 
    dg = g_obs - g_prior
    m_posterior = m_prior + np.matmul(K,dg)
    
    return m_posterior

def update(m:np.ndarray, g:np.ndarray, g_obs:np.ndarray, alpha:float, cd:np.ndarray):
    
    """
    Function responsible for updating the ensemble (prior to posterior) 
    by performing ES-MDA
    
    m       : matrix of prior parameters (Ne x a)
    g       : matrix of prior output (Ne x b)
    g_obs   : vector of measurement (1 x b)
    alpha   : es-mda sub-step
    cd      : vector of measurement error (1 x b) 
    """

    Ne = m.shape[0]

    cov_mg = calculate_covariance(m, g)
    cov_gg = calculate_covariance(g, g)

    #Main update equation
    m_post = np.zeros(m.shape)
    for j in range(Ne):
        _g_obs = np.random.normal(g_obs, np.abs(cd), g_obs.shape[0])
        m_post[j,:] = update_member(m[j,:], g[j,:], _g_obs, cov_mg, cov_gg, alpha, cd)
        
    return m_post

def ens_g_func(m:np.ndarray, g_func:callable):
    """
    Calculate forward simulation for the ensemble

    m       : matrix of prior paramters (Ne x a)
    g_func  : function dynamics
    """

    g = []
    for j in range(m.shape[0]):
        _g = g_func(m[j,:])
        g.append(_g)

    return np.array(g)


def es_mda(m:np.ndarray, g_func:callable, g_obs:np.ndarray, alphas:list, cd:list):
    """
    Function responsible for performing ES-MDA

    m       : matrix of prior parameters (Ne x a)
    g_func  : function call for the simulation
    g_obs   : vector of measurement (1 x b)
    alpha   : list of es-mda sub-step
    cd      : vector of measurement error (1 x b) 
    
    """
    assert g_obs.shape[0] == len(cd)
    assert np.sum(alphas) == 1

    g = ens_g_func(m, g_func)

    assert g.shape[1] == g_obs.shape[0]
    
    for alpha in alphas:
        m = update(m=m, g=g, g_obs=g_obs, alpha=alpha, cd=cd)
        #Calculate new forecast based on the predicted parameters
        g = ens_g_func(m, g_func)

    return m, g