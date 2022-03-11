from sklearn import ensemble 

MODELS = {
    "rfc": ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1, verbose=2),
    "etc": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}