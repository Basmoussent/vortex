#!/usr/bin/env python3

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import welch


class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns for binary EEG classification.

    Finds linear combinations of channels that maximize variance ratio between classes.
    Keeps n/2 filters from each end (best for each class).
    """

    def __init__(self, n_components=4, reg=1e-4):
    # def __init__(self, n_components=6, reg=1e-6):
        self.n_components = n_components  # Total filters to keep (must be even)
        self.reg = reg  # Tiny regularization for numerical stability
        self.W_ = None  # Spatial filters (learned during fit)

    def _compute_cov(self, X):
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
        Learn CSP spatial filters from training data.

        X: (n_epochs, n_channels, n_times) - EEG epochs
        y: (n_epochs,) - binary labels (0 or 1)

        https://en.wikipedia.org/wiki/Common_spatial_pattern#Applications


        CSP vas vouloir trouver les valeurs faisant en sorte que le signal de la classe A soit fort et celui de B faible pour eviter le bruit
        l'implementation est en bas et l'explication mathematique est sur le wiki plus haut
        """


        X, y = np.array(X), np.array(y)
        classes = np.unique(y)

        # Compute average covariance for each class
        cov1 = self._compute_cov(X[y == classes[0]])
        cov2 = self._compute_cov(X[y == classes[1]])

        # Composite covariance + regularization
        cov_c = cov1 + cov2
        cov_c += self.reg * np.eye(cov_c.shape[0])

        # Whitening: eigen decomposition of composite cov
        eigvals_c, eigvecs_c = np.linalg.eigh(cov_c)
        eigvals_c = np.maximum(eigvals_c, 1e-10)  # Avoid division by zero

        # Whitening matrix: P = U @ diag(1/sqrt(λ)) @ U.T
        # -----------------------------------------------
        # C'est un peu chiant ca mais en gros imagine que on as un nuage de point bah enfaites nos valeur seront sur des echelles differentes
        # ce qui est pas bon ducoup on vas "egalise" tout ca pour que totu soit normalise
        P = eigvecs_c @ np.diag(1.0 / np.sqrt(eigvals_c)) @ eigvecs_c.T


        # Transform cov1 to whitened space
        S1 = P @ cov1 @ P.T

        # Eigen decomposition of S1
        eigvals, eigvecs = np.linalg.eigh(S1)

        # Spatial filters
        W = eigvecs.T @ P

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigvals)[::-1]
        W = W[idx]

        # Keep n/2 first (high var for class 0) and n/2 last (high var for class 1)
        n_half = self.n_components // 2
        self.W_ = np.vstack([W[:n_half], W[-n_half:]])

        return self

    def transform(self, X):
        """
        Apply CSP filters and extract features (log-variance).

        X: (n_epochs, n_channels, n_times)
        Returns: (n_epochs, n_components) - log-variance features
        """
        X = np.array(X)
        n_epochs = X.shape[0]
        features = np.zeros((n_epochs, self.n_components))

        for i, epoch in enumerate(X):
            # Project epoch onto CSP filters
            projected = self.W_ @ epoch
            features[i] = np.log(np.var(projected, axis=1) + 1e-10)
        return features


    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)



def extract_psd_features(X, fs=160):
    """
    Extract Power Spectral Density features using Fourier Transform (Welch method).

    Welch method:
    - Splits signal into overlapping segments
    - Applies FFT (Fast Fourier Transform) to each segment
    - Averages the power spectra -> reduces noise in frequency estimation

    https://fr.wikipedia.org/wiki/Densit%C3%A9_spectrale_de_puissance

    https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Welch
    """
    n_epochs, n_channels, n_times = X.shape
    features = []

    for epoch in X:
        epoch_features = []

        for ch in range(n_channels):
            freqs, psd = welch(epoch[ch], fs=fs, nperseg=256)
            
            # Extract Mu/Beta band power and store the aerage power in these bands
            mu_idx = (freqs >= 8) & (freqs <= 12)
            mu_power = np.mean(psd[mu_idx])

            beta_idx = (freqs >= 12) & (freqs <= 30)
            beta_power = np.mean(psd[beta_idx])

            epoch_features.extend([mu_power, beta_power])

        features.append(epoch_features)

    return np.array(features)


