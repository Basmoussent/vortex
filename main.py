#!/usr/bin/env python3

import numpy as np
import mne
from visualize import load_data, apply_filter
from csp import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


def main():
    """Test CSP pipeline on real EEG data"""

    subject_id = 1
    runs = [4, 8, 12]  # Motor imagery left/right

	# on load juste de la data c'est pas interessant 
    raw = load_data(subject_id, runs)
    raw_filtered = apply_filter(raw, l_freq=8.0, h_freq=30.0)

    events, event_id = mne.events_from_annotations(raw_filtered, verbose=False)
    event_id_selected = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
	# jsuque la
    
	# on load les signaux ("epochs" c;est grossomodo les signaux )
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

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"5-Fold CV scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

if __name__ == "__main__":
    main()
