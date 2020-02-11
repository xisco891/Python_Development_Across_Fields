"""




    To-Do's:
        
        Optimization of the Model with Gradient Descent, RMSProp, AdaGrad, 
        Implement and test in your cross_validation data
        Plot learning curves to decide if more data, more features, etc, are likely to help
        Error Analysis.

"""



# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 02:21:37 2018

@author: Francisco
"""


import numpy as np 
import pandas as pd
from sklearn.cross_validation import train_test_split

from validation import is_standarised
from validation import Check_Data


from utils_statistics import convert_to_nummerical, imputting_missing_data, scale_data
from utils_statistics import label_encoder, onehot_encoder
from utils_statistics import split_data

from helper_MLR import BackwardElimination
from helper_MLR import PCAnalysis, lda, kernel_pca
from helper_MLR import Multiple_Linear_Regression, add_offset
from Summary_Statistics_Multivariate_Data import summary_statistics
from evaluation import evaluation_model, calculate_measures

#######################################################################################
###################### IMPORTING DATA #################################################
######################################################################################

#Importing the dataset
dataset = pd.read_csv("name_excel.csv")

#######################################################################################
###################### ENCODING DATA #################################################
######################################################################################

###Option 1. Encoding categorical data with sklearn.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

###Option 2. Applying One_Hot Encoding to turn Categorical Data to nummerical data.

 1 - Encoding a Column from the dataframe
e_data = pd.get_dummies(your_dataframe, columns=['column_names'], prefix=['X_'])
Take only the encoded variables, drop the remaining others
e_data = e_data.drop([e_data.columns[0], e_data.columns[1], ..., ], axis=1)
    
   Avoiding dummy variable trap
encoded_data = dummy_trap([e_data])
    

                


#######################################################################################
###################### SEARCH FOR MISSING VALUES AND FILL #############################
#######################################################################################
################################SCALE_DATA#############################################
#######################################################################################


###Imputing Missing Data for X and y 
X = imputting_missing_data(X, is_target = False)  #We indicate that is not a 1-d array
y = imputting_missing_data(y, is_target = True)   #We indicate that is a 1-d array


###Scale Data for Predictor Data and Target Data.
X, sc_X = scale_data(X)
y, sc_y = scale_data(y)


#######################################################################################
###################### SPLIT DATA INTO TRAINING/TEST SET###############################
#######################################################################################



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#######################################################################################
###################### SCIKIT LINEAR REGRESSION MODEL #################################
#######################################################################################

"""
Uncomment and Unindend the code for this section if you want to fit a linear regression
model to model your data. Keep in mind that LinearRegression object already performs
scaling for you. 

"""


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


#######################################################################################
###################### MANUAL BACKWARD_ELIMINATION LOGIC #########################
######################################################################################

##Optimizing the Model with Backward_Elimination Feature Selection technique -> In this case, 
##going one by one. 

"""
Uncomment and unindent this section.
"""

#Building the optimal model 
import statsmodels.formula.api as sm
#Now we add the first column to the matrix X, belonging to the b0 
#constant of our Multiple Linear Regression Model. 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#A matrix containing all the independent variables we have in our data. 
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() 
#Now we remove one of the variables that gives us a p_value > significance
#value which we set previously as 0.05
X_opt = X[:, [0,1,2,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() 

X_opt = X[:, [0,1,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,4]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


#####Performing Summary Statistics for the Data


#######################################################################################
###################### FEATURE SELECTION/EXTRACTION METHODS ###########################
#######################################################################################


standarisedX = pd.DataFrame(X)
if is_standarised(standarisedX) == True:
    
    opt = input("\t#-------Machine Learning and Statistics----------########"
                + "\n1 - Compute Stummary Statistics for the Data -> 0"
                + "\n\n2 - Choose Feature Selection/Extraction Method.........."
                + "\n\t 2.1 - Backward Elimination -> 1"
                + "\n\t 2.2 - PCA -> 2"
                + "\n\t 2.3 - PCA_LDA -> 3"
                + "\n\t 2.4 - PCA_KernelPCA -> 4"
                + "\n\t You choose: ")
    
    if opt is "0":
        dataframe = pd.DataFrame(X)
        summary_statistics(dataframe)
       
    elif opt is "1":
        
        X = BackwardElimination(X, y, SL=0.05, n_var = X.shape[1])
        X = add_offset(X, n_samples)
        X_decoded = pd.get_store()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        
    elif opt is "2":
        
        X = add_offset(X, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test = PCAnalysis(X_train, X_test) 
       
    
    elif opt is "3":
        
        X = add_offset(X, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test, n_comp = PCAnalysis(X_train, X_test)
        X_train, X_test = lda(X_train, X_test, y_train, n_comp)
                
    elif opt is "4":
        
        X = add_offset(X, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train, X_test, n_comp = PCAnalysis(X_train, X_test)
        X_train, X_test = kernel_pca(X_train, X_test, n_comp)

    else:
        print("No option has been defined")
        
                
    y_pred = Multiple_Linear_Regression(X_train, X_test, y_train )
    y_pred_rescaled = sc_y.inverse_transform(y_pred)
    y_expected_rescaled = sc_y.inverse_transform(y_test)
    
    
    n = X_test.shape[0]
    n_var = X_test.shape[1]
    print("n:" + str(n) + "n_var:" + str(n_var))
    evaluation_model(y_train, X_train)    
    calculate_measures(y_test, y_pred, n, n_var)   

       
    
    
    

    




