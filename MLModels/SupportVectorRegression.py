#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:59:22 2020

@author: xisco89
"""


'''
TO-DO : import missing libraries, expand functionalities, evaluation features, etc. 
'''

#######################################################################################
###################### SUPPORT VECTOR REGRESSION  ###########################
#######################################################################################

def Support_Vector_Regression(X_train, X_test, y_train):
    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train, y_train)
    
    
    #Predicting a new result
    y_pred = sc_y.inverse_transform(regressor.predict(X_test))
    
    # Visualising the Regression results
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_test, regressor.predict(X_test), color = 'blue')
    plt.title('Registration (Regression Model)')
    plt.xlabel('User Activity')
    plt.ylabel('Number of AdRequests')
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
    