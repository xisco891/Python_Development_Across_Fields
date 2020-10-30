#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:57:47 2020

@author: xisco89
"""

def lda(X_train, X_test, y_train, n_comp):

    print("\n\n\tComputing Linear Discriminant Analysis")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LD
    lda = LD(n_components=len(n_comp))
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    
    return X_train, X_test