


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
    return y_pred


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
    
    
