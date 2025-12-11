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

   