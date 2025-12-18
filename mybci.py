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


def load_model(filename_prefix="model"):
    """
    Load entire pipeline from file.
    Works with any method: csp, fourier, or both.
    """
    with open(f"{filename_prefix}_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def check_args():
    global subject_ids, exp_id, mode
    args = sys.argv[1:]
    len_arg = len(args)
    if len_arg == 3:
        try:
            subject_ids = int(args[1])
            exp_id = int(args[0])
            if exp_id < 0 or subject_ids < 0:
                raise ValueError("N)eed only positive arguments")
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





def train(pipeline, X, y, flag):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    accuracy = np.mean(y_pred == y)
    if  flag:
        scores = cross_val_score(pipeline, X, y, cv=5)
        print(scores)
        print(f"cross_val_scores: {scores.mean():.5f}")
    save_model(pipeline)  # sauvegarde apres 





def predict(pipeline, X, y, flag): # flag a 1 quand only predict et a 0 qunad on doit faire tous les sujet/exp
    global runs
    accuracy = 0
    if (flag):
        equal = False
        y_pred = pipeline.predict(X)
        print("epoch nb: [prediction] [truth] equal?")
        for i, (pred, truth) in enumerate(zip(y_pred, y)):
            equal =  True if int(truth) == int(pred) else False
            if equal == True:
                accuracy += 1
            print(f"epoch {i:02d}\t\t[{truth}]\t[{pred}] {equal}")
        print(f"Accuracy: {accuracy / len(y_pred)}")
    else:
        #laod data
        #train
        #acccuracies = [[]]
        for run in range(runs):
            for subject in list(range(1, 110)):
                pass
                #predict
                #accuracy = accuracy / len(y_pred)  need to check how to get the real positive
                #print(f"experiment {run}: subject {subject}: accuracy = {accuracy})
                #accuracies[run].append(accuracy)
            #accuracies[run]

        #print("Mean accuracy of the six different experiments for all 109 subjects:")
        #accuracies_2 = []
        #for exp in range(accuracies):
            #print(f"experiment {exp}: \t accuracy = {accuracies[exp].accuracy}) need to find how to get the accuracy for each exp
            #accuracies_2.append(accuracies[exp].accuracy)
        #print(f"Mean accuracy of 6 experiments{accuracies_2.accuracy}")




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

    if ( exp_id == -1):
        runs = [4, 6, 8, 10, 12, 14]  # 6 runs au lieu de 3 pour plus de données

    subject_id = subject_ids
    runs = exp_id


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


    if method == "csp":
        # Option 1: CSP spatial filtering only
        pipeline = Pipeline([
            ('csp', CSP(n_components=4, reg=1e-4)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X

    elif method == "fourier":
        # Option 2: Fourier PSD features only
        X_fourier = extract_psd_features(X, fs=160)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X_fourier

    elif method == "both":
        # Option 3: CSP + Fourier combined

        # Extract both feature types
        csp = CSP(n_components=4, reg=1e-4)
        X_csp = csp.fit_transform(X, y)
        X_fourier = extract_psd_features(X, fs=160)

        # Combine features
        X_combined = np.hstack([X_csp, X_fourier])

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])
        X_features = X_combined

    else:
        raise ValueError(f"Invalid method: {method}. Choose 'csp', 'fourier', or 'both'")


    # Train/Predict/Evaluate
    if mode == "train":
        train(pipeline, X_features, y, True)
    if mode == "predict":
        pipeline = load_model()
        predict(pipeline, X_features, y, True)
    if mode == "unknown":
        train(pipeline, X_features, y, False)
        pipeline = load_model()
        predict(pipeline, X_features, y, False)


if __name__ == "__main__":
    main()
