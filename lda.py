from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda():
    clf = LinearDiscriminantAnalysis(
        solver='svd',       # le plus courant (par defaut)
        shrinkage=None,
        priors=None,
    )
    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)
    