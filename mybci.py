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
from sklearn.model_selection import cross_val_score, train_test_split

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





def train(pipeline, X_train, y_train, X_test, y_test, flag):
    #
    # Ducoup maitenant on as 2 data train et test
    #
    if flag:
        scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        print(f"Cross-val scores: {scores}")
        print(f"Cross-val mean: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train)

    y_test_pred = pipeline.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)

    if flag:
        print(f"\nTrain accuracy: {train_accuracy:.3f}")
        print(f"Test accuracy: {test_accuracy:.3f}")
        if train_accuracy - test_accuracy > 0.15:
            print(f"ovverfitting detecteeeeed boi: {train_accuracy - test_accuracy}")


    save_model(pipeline)





def predict(pipeline, X_test, y_test, flag):
    """Prediction sur test set uniquement"""
    y_pred = pipeline.predict(X_test)

    if flag:
        print("epoch nb: [truth] [prediction] equal?")
        for i, (truth, pred) in enumerate(zip(y_test, y_pred)):
            equal = True if int(truth) == int(pred) else False
            print(f"epoch {i:02d}\t\t[{truth}]\t[{pred}]\t {equal}")

    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest accuracy: {accuracy:.3f}")







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
        runs = [4, 6, 8, 10, 12, 14]  # 6 runs au lieu de 3 pour plus de données
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

    # Split train/test - 80% train, 20% test (plus de données pour entraîner)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split: {len(X_train)} train epochs, {len(X_test)} test epochs")

    pipeline = Pipeline([
        ('csp', CSP(n_components=2, reg=1e-3)),  # 2 composants au lieu de 4, plus de régularisation
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')) # check to edit algo/settings
    ])

    if mode == "train":
        train(pipeline, X_train, y_train, X_test, y_test, True)
    if mode == "predict":
        pipeline = load_model()
        predict(pipeline, X_test, y_test, True)
    if mode == "unknown":
        train(pipeline, X_train, y_train, X_test, y_test, True)  # True pour voir les détails
        pipeline = load_model()
        predict(pipeline, X_test, y_test, True)  # True pour voir les prédictions



if __name__ == "__main__":
    main()
