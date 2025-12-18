#!/usr/bin/env python3

import numpy as np
import mne
import sys
import json
import pickle
from pathlib import Path
from visualize import load_data, apply_filter
from csp import CSP, extract_psd_features  # CSP spatial filtering + Fourier PSD features
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

subject_ids = -1
mode = "unknown"
exp_id = -1



def save_model(pipeline, filename_prefix="model"):
    """
    Save entire pipeline to file using pickle.
    Works with any method: csp, fourier, or both.
    """
    with open(f"{filename_prefix}_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {filename_prefix}_pipeline.pkl")


def load_model(filename_prefix="model"):
    """
    Load entire pipeline from file.
    Works with any method: csp, fourier, or both.
    """
    with open(f"{filename_prefix}_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    print(f"Model loaded from {filename_prefix}_pipeline.pkl")
    return pipeline


def check_args():
    global subject_ids, exp_id, mode
    args = sys.argv[1:]
    len_arg = len(args)
    if len_arg == 3:
        try:
            subject_ids = int(args[1])
            exp_id = int(args[0])
        except Exception as e:
            print("Need an int as argument for the number of the experiments and/or for the subject")
            exit(1)
        mode = args[2]
        if (mode != "train" and mode != "predict"):
            print("The last arguments must be \'train\' or \'predict\'")
            exit(1)
        return 
    if len_arg == 0:
        subject_ids = -1 # Quand pas d arg et qu on doit faire train et predict si -1 faire tout les sujets et toutes les experiments
        return 
    raise ValueError()





def train(pipeline, X, y):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    accuracy = np.mean(y_pred == y)
    save_model(pipeline)  # sauvegarde apres entrainement





def predict(pipeline, X, y, flag): # flag a 1 quand only predict et a 0 qunad on doit faire tous les sujet/exp
    flag = True #toremove (only bc for all isnt implemented yet)
    if (flag):
        equal = False
        y_pred = pipeline.predict(X)
        print("epoch nb: [prediction] [truth] equal?")
        for i, (pred, truth) in enumerate(zip(y_pred, y)):
            equal =  True if int(truth) == int(pred) else False
            print(f"epoch {i:02d}\t\t[{truth}]\t[{pred}] {equal}")
    else:
        print("NEED TO IMPLEMENT THE VERSION FOR ALL THE SUBJECTS/EXP")





def main():
    """Test CSP pipeline on real EEG data"""
    global subject_ids, exp_id, mode

    # Options: "csp", "fourier", "both"
    method = "fourier"
    try:
        check_args()
    except ValueError:
        print("Error\nUsage: python <file> <experiment> <subject> <mode> \\")
        print("Usage: python <file>")
        exit(1)

    subject_id = subject_ids
    runs = exp_id

    #toremove ########
    if (subject_ids == -1 and exp_id == -1):
        subject_id = 14
        runs = 4
    ##################

	# on load juste de la data c'est pas interessant 
    try:
        raw = load_data(subject_id, runs)
    except ValueError:
        print("inexisiting subject or experiment")
        exit(1)

    raw_filtered = apply_filter(raw, l_freq=8.0, h_freq=30.0)

    events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)
    event_id_selected = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
	# jsuque la
    
	# on load les signaux ("epochs" c;est grossomodo les signaux )
    try :
        epochs = mne.Epochs(
            raw_filtered,
            events,
            event_id=event_id_selected,
            tmin=0.0,
            tmax=2.0,
            baseline=None,
            preload=True,
            verbose=False
        )
    except ValueError:
        print("Experiment value or subject invalid")
        exit(1)

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2]

    # Convert labels to 0 and 1
    unique_labels = np.unique(y)    
    y = (y == unique_labels[1]).astype(int)



    # print(f"\nData: {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} samples")
    # print(f"Classes: {np.sum(y==0)} vs {np.sum(y==1)}")

    # print("\n" + "-"*60)
    # print("CSP Dimensionality Reduction")
    # print("-"*60)

    # csp = CSP(n_components=6)
    # csp.fit(X, y)
    # X_csp = csp.transform(X)

    # print(f"Input:  {X.shape}")
    # print(f"Output: {X_csp.shape}")
    # print(f"Reduction: {X.shape[1]} channels → {csp.n_components} features")

    # print("\n" + "-"*60)
    # print("Pipeline Test (CSP + LDA)")
    # print("-"*60)

    # ==================================================================
    # FEATURE EXTRACTION based on method choice
    # ==================================================================
    print("\n" + "="*60)
    print(f"Method selected: {method.upper()}")
    print("="*60)

    if method == "csp":
        # Option 1: CSP spatial filtering only
        print("Using CSP (Common Spatial Patterns) - Spatial filtering")
        print(f"Original data: {X.shape} (epochs, channels, time_samples)")

        pipeline = Pipeline([
            ('csp', CSP(n_components=4, reg=1e-4)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X
        print(f"Pipeline: CSP -> StandardScaler -> LDA")

    elif method == "fourier":
        # Option 2: Fourier PSD features only
        print("Using Fourier Transform (Welch PSD) - Frequency domain")
        X_fourier = extract_psd_features(X, fs=160)
        print(f"Original data:     {X.shape} (epochs, channels, time_samples)")
        print(f"Fourier features:  {X_fourier.shape} (epochs, frequency_features)")
        print(f"Feature reduction: {X.shape[1] * X.shape[2]} -> {X_fourier.shape[1]}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X_fourier
        print(f"Pipeline: Fourier PSD -> StandardScaler -> LDA")

    elif method == "both":
        # Option 3: CSP + Fourier combined
        print("Using CSP + Fourier (Combined) - Spatial + Frequency domain")

        # Extract both feature types
        csp = CSP(n_components=4, reg=1e-4)
        X_csp = csp.fit_transform(X, y)
        X_fourier = extract_psd_features(X, fs=160)

        # Combine features
        X_combined = np.hstack([X_csp, X_fourier])

        print(f"Original data:     {X.shape} (epochs, channels, time_samples)")
        print(f"CSP features:      {X_csp.shape} (epochs, csp_components)")
        print(f"Fourier features:  {X_fourier.shape} (epochs, frequency_features)")
        print(f"Combined features: {X_combined.shape} (epochs, total_features)")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X_combined
        print(f"Pipeline: CSP+Fourier -> StandardScaler -> LDA")

    else:
        raise ValueError(f"Invalid method: {method}. Choose 'csp', 'fourier', or 'both'")

    print("="*60 + "\n")

    # Train/Predict/Evaluate
    if mode == "train":
        train(pipeline, X_features, y)
    if mode == "predict":
        predict(pipeline, X_features, y, True)
    if mode == "unknown":
        train(pipeline, X_features, y)
        pipeline = load_model()
        predict(pipeline, X_features, y, False)

    # Cross-validation to evaluate generalization performance
    scores = cross_val_score(pipeline, X_features, y, cv=5)
    print(f"\n5-Fold CV scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

if __name__ == "__main__":
    main()
