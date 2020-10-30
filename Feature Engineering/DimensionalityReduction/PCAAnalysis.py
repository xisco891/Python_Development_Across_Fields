#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:56:58 2020

@author: xisco89
"""


def screeplot(pca, standardised_values):

    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")

    
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



def kernel_pca(X_train, X_test, n_comp):
    # Applying Kernel PCA
    print("\n\n\tComputing Kernel Principal Component Analysis")
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components = len(n_comp), kernel = 'rbf')
    
    X_train = kpca.fit_transform(X_train)
    X_test = kpca.transform(X_test)
    return X_train, X_test


