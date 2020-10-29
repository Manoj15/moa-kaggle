from sklearn import ensemble

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=10, n_jobs=-1, verbose=2),
}