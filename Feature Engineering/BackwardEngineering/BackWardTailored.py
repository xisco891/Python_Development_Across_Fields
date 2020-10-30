#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:06:58 2020

@author: xisco89
"""


### 5 Cooked recipes for Backward Selection/Engineering
 
def backwardElimination_adjR(x, SL, y):


    import statsmodels.formula.api as sm
    numVars = len(x[0])
    temp = np.zeros((x.shape[0],x.shape[1])).astype(int)


    for i in range(0, numVars):
        print("numVars: ", numVars)
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                print("num_vars: ", numVars - i)
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
                    
    regressor_OLS.summary()
    return x
 
def backwardElimination(X, y, n_var, sl, column_names):
    
    list_ind_var = [x for x in range(0, n_var)]
    x = X[:, list_ind_var]
   
    try:
        numVars = len(x[0])
        
        for i in range(0, numVars):
            
            regressor_OLS = sm.OLS(y,x).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            
            
            if maxVar > sl:
                for j in range(0, numVars - i):
                    print("numVars: ", numVars)
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        x = np.delete(x,j,1)
                        list_ind_var.pop(j)
            else:
                print(regressor_OLS.summary())
                return x, list_ind_var
            
            print("i, length_x : " + str(i) + "," + str(len(x[0])))
            
    except Exception as Ex:
        print("An Exception has been found: " + str(Ex))
        return None
                

