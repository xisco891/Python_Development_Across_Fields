


######## Univariate Selection ##########

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#### Discard those features/variables that have no statistical influence over the output variable. 

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5,:])



                
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)




#### Feature Importance -> With the use of bagged decission trees like Random Forest and Extra Trees can be used to estimate the importance of features. 

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)




                   
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_                           



This module provides the functionalities that will be needed to perform linear regression
techniques throughout the development of Machine Learning/Data Science Projects. 


Notes:
    
    Keep in mind that Regression methods already perform scaling!.
    Caution : Shouldn't be this new method exported to a validation.py module?
    
                        
To-Do'S: 

    1 - Check if backwardElimination_adjR function really works.
    
    2 - backwardElimination can be enhanced to present summary data in a better fashion.
    
    3 - Research for Feature Selection/Extraction Methods and Techniques that are worth 
        adding here. -> forward_selection, Bidirectional Elimination, Score Comparison, 
                        state of the art feature extraction methods. 
    
    4 - For each regression model improve the plotting logic. 

    
    5 -    I/We need to verify that our data can be modelled with a linear regression:
           To apply a linear model you need to verify: 
        
            #1-Linearity -> Implement function to check if data is linearly separable?.
            
            #2-Homoscedasticity -> Implement function to verify is there any 
                                 homeoscedasticity in the data.
            
            #3-Multivariate Normality -> Implement function to check if multivariate data
                                        follows a normal distribution.
                                        
            #4-Independence of Errors -> Implementation a function to estimate
                                         the independence of errors. 
            
            
            #5-Lack of multicollinearity. -> Implement a function to measure how 
                                             multicollinear data is.
        

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt



#######################################################################################
###################### FORWARD SELECTION TECHNIQUES ########################
#######################################################################################


 
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
                



#######################################################################################
###################### FORWARD EXTRACTION TECHNIQUES ########################
#######################################################################################


    
def PCAnalysis(Train, Test):
    print("\n\n\tComputing Principal Component Analysis")
    
    pca = PCA(n_components=None)
    X_train = pca.fit_transform(Train)
    X_test = pca.transform(Test)
    
    extra = input("Choose to further know about your PCA results:"
                      + "\n\t 1.1 Perform PCA Summary"
                      + "\n\t 1.2 Perform visualization of the Summary"
                      + "\n\t 1.3 Not this time..."
                      + "\n\n\t Choose Your Option: ")
    
    if extra is "0":
        pca_summary(pca, X_train)
            
    elif extra is "1":
        screeplot(pca, X_train)        
    
    else:
        pass
        
    return X_train, X_test


def pca_summary(pca, standardised_data, out=True):
    
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(standardised_data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    
    if out:
        print("Importance of components:")
        print(summary)
        
    return summary

def lda(X_train, X_test, y_train, n_comp):

    print("\n\n\tComputing Linear Discriminant Analysis")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LD
    lda = LD(n_components=len(n_comp))
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    
    return X_train, X_test
    

def kernel_pca(X_train, X_test, n_comp):
    # Applying Kernel PCA
    print("\n\n\tComputing Kernel Principal Component Analysis")
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components = len(n_comp), kernel = 'rbf')
    
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)
    return X_train, X_test



def screeplot(pca, standardised_values):

    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()
    
    
    


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
    
    




#### Recursive Feature Elimination -> Removing attributes contributing in a lesser quantity to build a model. The model should serve to identify which attributes
                                  #-> and which combination of attributes contributes the most to predict a certain output/predicting the target attribute. 
                                  
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_                           




#### Principal Component Analysis -> With thi method we compress the space of features, by reducing the number of features. This is often mentioned as a reduction technique, 
                                    #that helps to tranform the space into a combination of all. 
                                    
                                    
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)


################################################################################################################################################################

#### Feature Importance -> With the use of bagged decission trees like Random Forest and Extra Trees can be used to estimate the importance of features. 

# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)




##1 Backward Elimination.

X = BackwardElimination(X, y, SL=0.05, n_var = X.shape[1])
X = add_offset(X, n_samples)
X_decoded = pd.get_store()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##2 PCA Feature Extraction. 
X = add_offset(X, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test = PCAnalysis(X_train, X_test) 
       
    
##3 PCA_LDA
X = add_offset(X, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, n_comp = PCAnalysis(X_train, X_test)
X_train, X_test = lda(X_train, X_test, y_train, n_comp)

#4 Kernel kERNEL PCA                
X = add_offset(X, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_test, n_comp = PCAnalysis(X_train, X_test)
X_train, X_test = kernel_pca(X_train, X_test, n_comp)

#5 Multiple Linear Regression. 
y_pred = Multiple_Linear_Regression(X_train, X_test, y_train )
y_pred_rescaled = sc_y.inverse_transform(y_pred)
y_expected_rescaled = sc_y.inverse_transform(y_test)
    
# Evaluate and compute the scores. 
    
   

       
    
    
