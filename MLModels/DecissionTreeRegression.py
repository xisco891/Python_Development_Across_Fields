#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:00:48 2020

@author: xisco89
"""



'''
TO-DO : import missing libraries, expand functionalities, evaluation features, etc. 
'''


#######################################################################################
###################### DECISSION TREE REGRESSOR #######################################
#######################################################################################

def decission_tree_Regression(X_train, X_test, y_train):
    # Fitting the Regression Model to the dataset
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train,y_train)
    
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    
    # Visualising the Regression results
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X, y_pred, color = 'blue')
    plt.title('Truth or Bluff (Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    
    # Visualising the Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X), max(X), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X, y, color = 'red')
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
    plt.title('Truth or Bluff (Regression Model)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    