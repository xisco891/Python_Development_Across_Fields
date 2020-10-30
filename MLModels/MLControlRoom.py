#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:03:31 2020

@author: xisco89
"""


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