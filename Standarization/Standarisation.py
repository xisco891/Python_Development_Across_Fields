# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:36:12 2019

@author: Francisco
"""


# 13 - Scale Data for all Data, need for standarisation to look for relationships.
X, sc_X = scale_data(X)
df_data = pd.DataFrame(X)


def scale_data(data): 
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    scaled_data = sc.fit_transform(data)
    return scaled_data, sc
  
  
  
def imputting_missing_data(data, is_target, strategy=None):
    
    from sklearn.preprocessing import Imputer
    if strategy is None:
        imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
    else:
        imputer = Imputer(missing_values='NaN', strategy = strategy, axis = 0)
    if is_target == True:
        data = data.values.reshape(-1,1)
        
    imputer = imputer.fit(data)
    X = imputer.transform(data)
    return X



import pandas as pd 
import numpy as np

def convert_to_nummerical(data, array, errors):
    for i in array:
        data.iloc[:, i] = pd.to_numeric(data.iloc[:,i], errors)
    return data

def split_data(dataframe):

    cat_data = dataframe.select_dtypes(include=['object'].copy())
    non_cat_data = dataframe.select_dtypes(exclude=['object'].copy())
    return cat_data, non_cat_data



