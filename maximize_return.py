import cvxpy as cvx
'''
expected_returns = [60, 50] ## 

def maximize_return(expected_returns):
    weights = cvx.Variable(len(expected_returns))
    objective = cvx.Maximize(weights.T * expected_returns)
    
    problem = cvx.Problem(objective, [cvx.sum(cvx.abs(weights))<=1])
    #  problem = cvx.Problem(objective, [cvx.sum_entries(weights)<=1])
    
    problem.solve()
    
    return weights.value.round(3)

solution = maximize_return(expected_returns)

'''
#######################  ######################
import numpy as np
"""
expected_ret = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
cov = np.array([[0.02, 0 , 0 ,0 ,0 ],
                [0, 0.02 , 0. ,0 ,0],
                [0, 0 , 0.02 ,0. ,0. ],
                [0, 0 , 0. ,0.02 ,0.01 ],
                [0, 0 , 0 ,0 ,0.02 ]])

x = np.random.randn(100)
y = np.random.randn(100)
cov = np.cov(x, y)
expected_ret = np.array([0.001, 0.002])
"""
def markwitz_portpolio(means, cov, risk_aversion=0.7):
    
    weights = cvx.Variable(len(means))
    
    expected_return = weights.T*means
    expected_vol = cvx.quad_form(weights, cov)
    
    utility = expected_return - risk_aversion*expected_vol
    objective = cvx.Maximize(utility)
    
    constraints = [
        cvx.sum(weights, axis = 0) == 1,
        weights >= 0,
        ]    
    
    problem = cvx.Problem(objective, constraints)
    #  problem = cvx.Problem(objective, [cvx.sum_entries(weights)<=1])
    
    problem.solve()
    
    return np.array(weights.value.flat).round(4)
#, expected_return.value, expected_vol.value

#weights, rets, var = markwitz_portpolio(expected_ret, cov, risk_aversion = 0.2)
    
def sharpe_ratio(w, mean, cov):
    sr = (np.matmul(w.T,mean))/(np.sqrt((np.matmul(w.T,np.matmul(cov,w)))))
    return sr