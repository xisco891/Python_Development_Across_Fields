#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:02:22 2020

@author: xisco89
"""



#######################################################################################
###################### MULTIPLE LINEAR/LINEAR REGRESSION  ###########################
#######################################################################################

    
def add_offset(X, n_samples):
    X = np.append(arr=np.ones((n_samples,1)).astype(int), values = X, axis = 1)
    return X    


def Multiple_Linear_Regression(X_train, X_test, y_train):
    
    ##LinearRegression library from sklearn performs feature scaling....
    from sklearn.linear_model import LinearRegression
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    return regressor, y_pred


