# -*- coding: utf-8 -*-

"
import numpy as np


def is_standarised(X):
    mean = X.apply(np.mean)
    std = X.apply(np.std)
    
    print(mean)
    print(std)
    
    if np.any(mean > 1 * 10**(-3)) and np.any(std > 0.99) and np.any(std < 1.01):
        return False
        print("Data is not standarised")
    else: 
        print("Data is  standarised")
        return True



    
def is_cat_data(self, dataframe):
    
    for i in range(0, dataframe.shape[1]):
        value = dataframe.iloc[0,i]
        try:
            if is_float(value) is True:  
                self.index.append(i)
                  

def is_float(x):

    try:
        float(x)
        return True
    except Exception as Ex:
        return False

