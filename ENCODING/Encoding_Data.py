# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:35:29 2019

@author: Francisco
"""


# 1- Encoding "Campaign_name"
e_data = pd.get_dummies(cat_data, columns=['campaign_name'], prefix=['Camp_'])
#Take only the encoded variables, drop the 2 others
e_data = e_data.drop([e_data.columns[0], e_data.columns[1]], axis=1)

# 4- Avoiding dummy variable trap
encoded_data = dummy_trap([e_data])





def label_encoder(data):

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    data = label_encoder.fit_transform(data)
    classes = label_encoder.classes_
    mappings = { i : classes[i] for i in range(0, len(classes))}
    return data, mappings
    

def onehot_encoder(data, list_features=None):
    
    from sklearn.preprocessing import OneHotEncoder    
   
    if list_features is None or len(list_features) is 0:
        onehot_encoder = OneHotEncoder(categorical_features=data.shape[1])
    else:
        onehot_encoder = OneHotEncoder(categorical_features=list_features)
   
    data = onehot_encoder.fit_transform(data).toarray()
    data = np.delete(data, 0,1) #Remove one variable to avoid dummy variable trap.
    return data

def dummy_trap(dataframes):
    
    encoded_data = pd.DataFrame()
    for dataframe in dataframes:
        index = list(dataframe)
        dataframe = dataframe.drop(index[0], axis=1)
        encoded_data = pd.concat([encoded_data, dataframe],axis=1)


    return encoded_data
