#!/usr/bin/env python3
"""
CSP (Common Spatial Patterns) - Minimal implementation
Finds spatial filters that maximize class separation
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6, reg=1e-6):
        self.n_components = n_components
        self.reg = reg 
        self.W_ = None  # Spatial filters

    def cov(self, X):
        """Average covariance normalized by trace"""
        covs = []
        for epoch in X:
            epoch = epoch - epoch.mean(axis=1, keepdims=True)
            cov = epoch @ epoch.T / (epoch.shape[1] - 1)
            cov /= np.trace(cov)
            covs.append(cov)
        return np.mean(covs, axis=0)
    
    def fit(self, X, y):
        """
        X: (n_trials, n_channels, n_times)
        y: (n_trials,) avec valeurs 0 ou 1
        """
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]

        C0 = self.cov(X_class0) #cov mean 
        C1 = self.cov(X_class1)

        C0 += self.reg * np.eye(C0.shape[0]) #regularisation (avoid to many noise)
        C1 += self.reg * np.eye(C1.shape[0])

        eigenvalues, eigenvectors = np.linalg.eig( # keep clean values (extremity)
            np.linalg.solve(C0 + C1, C1) # compare total variance to c1 var
        )

        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx] # reverse sort to set bigger (classe 1)at the start
                                            # and smaller (classe 0) at the end

        n_pairs = self.n_components // 2
        self.W_ = np.hstack([ #keep main signals
            eigenvectors[:, :n_pairs], # copy les n premiere values 
            eigenvectors[:, -n_pairs:] # copy les n derniere values 
        ])
        return self

    def transform(self, X):
        """Extraire les features CSP"""
        X_csp = np.asarray([self.W_.T @ epoch for epoch in X])
        features = np.log(np.var(X_csp, axis=2)) # normalise the variance of each signal
        return features

   