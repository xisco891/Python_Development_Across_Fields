
'''This module provides the functionalities that will be needed to perform linear regression
techniques throughout the development of Machine Learning/Data Science Projects. 
'''


######## Univariate Selection ##########

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

######## Feature Extraction with PCA ##########

import numpy as np
from pandas import read_csv
from sklearn.decomposition import PCA

######## Feature Importance with Extra Trees Classifier ##########
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

######## Feature Extraction with RFE ##########
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


####### Forward Selection Techniques ##########
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt



### 1. Univariate Selection : 
    
''' We aim at discarding those features/variables that have no statistical 
   influence over the output variable. 
'''

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)

### 1.1 Precission Metric Scores : 
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

#### 1.2 Summarize Selected Features. 
print(features[0:5,:])


### 2. Feature Extraction with PCA : 
                
### 2.1 Loading Data -> diabetes data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

### 2.2 Extracting Features. 
pca = PCA(n_components=3)
fit = pca.fit(X)

### 2.3 What variance can tell us of each component. 
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)



### 3. Bagged Decission trees : Random Forest and Extra Trees. 

### 3.1 Loading Data -> diabetes data. 
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

### 3.2 Fitting a tree model and summarizing feature importance. 
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


### 4. Forward Selection with RFE. 

### 4.1 Loading Data -> diabetes data. 

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

### 4.1 Fitting a log-regression model and summarizing features.  
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_                           



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
                




def screeplot(pca, standardised_values):

    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()
    
    
    

    
   

       
    
    
