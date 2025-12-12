#!/usr/bin/env python3

import numpy as np
import mne
import sys
import json
from pathlib import Path
from visualize import load_data, apply_filter
from csp import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

subject_ids = 0
mode = "unknown"
exp_id = 0



def save_settings(pipeline, subject_id, exp_id):
    """Save trained model parameters to settings.json"""
    csp = pipeline.named_steps['csp']
    lda = pipeline.named_steps['lda']
    
    settings = {
        'subject_id': subject_id,
        'exp_id': exp_id,
        'filter': {
            'l_freq': 8.0,
            'h_freq': 30.0
        },
        'csp': {
            'n_components': csp.n_components,
            'W': csp.W_.tolist(),  # Spatial filters
            'reg': csp.reg
        },
        'lda': {
            'means': lda.means_.tolist(),
            'covariance': lda.covariance_.tolist(),
            'priors': lda.priors_.tolist(),
            'scalings': lda.scalings_.tolist()
        }
    }
    
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    
    print("Settings saved to settings.json")



def load_settings():
    """Load model parameters from settings.json"""
    if not Path('settings.json').exists():
        print("Error: settings.json not found. Train first!")
        exit(1)
    
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    
    # Reconstruct CSP
    csp = CSP(n_components=settings['csp']['n_components'], 
              reg=settings['csp']['reg'])
    csp.W_ = np.array(settings['csp']['W'])
    
    # Reconstruct LDA
    lda = LinearDiscriminantAnalysis()
    lda.means_ = np.array(settings['lda']['means'])
    lda.covariance_ = np.array(settings['lda']['covariance'])
    lda.priors_ = np.array(settings['lda']['priors'])
    lda.scalings_ = np.array(settings['lda']['scalings'])
    lda.classes_ = np.array([0, 1])
    
    pipeline = Pipeline([
        ('csp', csp),
        ('lda', lda)
    ])
    
    print(f"Settings loaded (subject={settings['subject_id']}, exp={settings['exp_id']})")
    return pipeline, settings




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
        subject_ids = -1 # si -1 faire tout les sujets et toutes les experiments
        return 
    raise ValueError()





def train(pipeline, X, y):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Training accuracy: {accuracy:.3f}")
    save_settings(pipeline, subject_ids, exp_id)





def predict(pipeline, X, y):
    pipeline, settings = load_settings()
    equal = False
    y_pred = pipeline.predict(X)
    print("epoch nb: [prediction] [truth] equal?")
    for i, (pred, truth) in enumerate(zip(y_pred, y)):
        equal =  True if truth == pred else False
        print(f"epoch {i:0<1}\t\t["f"{truth}""]\t["f"{pred}""] "f"{equal}")





def main():
    """Test CSP pipeline on real EEG data"""
    global subject_ids, exp_id, mode
    try:
        check_args()
    except ValueError:
        print("Error\nUsage: python <file> <experiment> <subject> <mode> \\")
        print("Usage: python <file>")
        exit(1)



    subject_id = subject_ids
    runs = exp_id

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



    print(f"\nData: {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} samples")
    print(f"Classes: {np.sum(y==0)} vs {np.sum(y==1)}")

    print("\n" + "-"*60)
    print("CSP Dimensionality Reduction")
    print("-"*60)

    csp = CSP(n_components=6)
    csp.fit(X, y)
    X_csp = csp.transform(X)

    print(f"Input:  {X.shape}")
    print(f"Output: {X_csp.shape}")
    print(f"Reduction: {X.shape[1]} channels → {csp.n_components} features")

    print("\n" + "-"*60)
    print("Pipeline Test (CSP + LDA)")
    print("-"*60)

	# Lis la def de LinearDiscrimnantAnlylis elle est pas longue et tres comprehensible
    pipeline = Pipeline([
        ('csp', CSP(n_components=6)),
        ('lda', LinearDiscriminantAnalysis())
    ])

    if mode == "train":
        train(pipeline, X, y)
    if mode == "predict":
        predict(pipeline, X, y)
    if mode == "unknown":
        train(pipeline, X, y)
        predict(pipeline, X, y)


    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"5-Fold CV scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

if __name__ == "__main__":
    main()
