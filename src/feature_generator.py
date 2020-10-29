# Import Modules
from imblearn.over_sampling import SVMSMOTE 

# Oversampling function
def oversample_minority_svm(X, y):
    sm = SVMSMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res