#!/usr/bin/env python3

import numpy as np
import mne
import sys
import json
from pathlib import Path
from visualize import load_data, apply_filter
from csp import CSP
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

subject_ids = -1
mode = "unknown"
exp_id = -1



def save_model(pipeline, filename_prefix="model"):
    """Sauvegarde CSP et LDA séparément"""
    csp = pipeline.named_steps['csp']
    lda = pipeline.named_steps['lda']

    # CSP
    csp_params = {"W": csp.W_.tolist()}
    with open(f"{filename_prefix}_csp.json", "w") as f:
        json.dump(csp_params, f)

    # LDA
    lda_params = {
        "coef": lda.coef_.tolist(),
        "intercept": lda.intercept_.tolist(),
        "classes": lda.classes_.tolist()
    }
    with open(f"{filename_prefix}_lda.json", "w") as f:
        json.dump(lda_params, f)




def load_model(n_components=6, filename_prefix="model"):
    """Charge CSP et LDA depuis fichiers et retourne le pipeline"""

    # Charger CSP
    with open(f"{filename_prefix}_csp.json", "r") as f:
        csp_data = json.load(f)
    W = np.array(csp_data["W"])
    csp = CSP(n_components=W.shape[0])
    csp.W_ = W

    # Charger LDA
    with open(f"{filename_prefix}_lda.json", "r") as f:
        lda_data = json.load(f)
    lda = LinearDiscriminantAnalysis()
    lda.coef_ = np.array(lda_data["coef"])
    lda.intercept_ = np.array(lda_data["intercept"])
    lda.classes_ = np.array(lda_data["classes"])

    # Pipeline final
    pipeline = Pipeline([
        ('csp', csp),
        ('lda', lda)
    ])
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





def train(pipeline, X, y, flag):
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    accuracy = np.mean(y_pred == y)
    scores = cross_val_score(pipeline, X, y)
    if flag:
        print(f"{scores}\ncross_val_scores: {scores.mean()}")
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
        # print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    # print(f"Accuracy{"accuracy"}")
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.3f}")







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

    #toremove ########
    if (subject_ids == -1 and exp_id == -1):
        subject_id = 14
        runs = [4, 8, 12]
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


    pipeline = Pipeline([
        ('csp', CSP(n_components=4, reg=1e-4)),
        ('scaler', StandardScaler()), # need to normalise and scaling features/values
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')) # check to edit algo/settings
    ])

    if mode == "train":
        train(pipeline, X, y, True)
    if mode == "predict":
        predict(pipeline, X, y, True)
    if mode == "unknown":
        train(pipeline, X, y, False)
        pipeline = load_model()
        predict(pipeline, X, y, False)



if __name__ == "__main__":
    main()
