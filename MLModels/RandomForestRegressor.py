#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:01:32 2020

@author: xisco89
"""

#######################################################################################
####################### RANDOM FOREST REGRESSOR #######################################
#######################################################################################
    
    
    

def randomforest_Regression(X_train, X_test, y_train):

    ##Number of estimators -> Research for appropiate values. 
    
    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators=300,random_state=0)
    regressor.fit(X_train,y_train)
                  
    # Predicting a new result
    y_pred = regressor.predict(X_train)
    
    # Visualising the Regression results (for higher resolution and smoother 
    #curve)
    
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Random Forest Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    
    