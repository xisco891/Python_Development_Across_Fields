from sklearn.metrics import mean_squared_error, r2_score
import numpy as np



def evaluation_model(y_train, X_train):
    
    import statsmodels.formula.api as sm
    regressor_OLS = sm.OLS(endog=y_train, exog=X_train).fit()
    summary = regressor_OLS.summary()
    print(summary)
    print(type(summary))

         
def calculate_measures(y_test, y_pred, n, n_var):
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test,y_pred)
    r2_adj = 1-(1-r2)*(n-1)/(n-(n_var+1))
    
    print("\n\tRMSE: " + str(rmse))
    print("\n\tr2 score: " + str(r2))
    print("\n\tr2 adjusted score: " + str(r2_adj))
