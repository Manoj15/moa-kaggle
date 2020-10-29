import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
from configparser import ConfigParser
from .create_folds import create_kfolds

print(os.getcwd())
from . import dispatcher

from .feature_generator import oversample_minority_svm

from .categorical import CategoricalFeatures

# Read Config
config = ConfigParser()
config.read('./src/config.ini')
TRAINING_DATA_X = config.get('main', 'TRAINING_DATA_X')
TRAINING_DATA_Y = config.get('main', 'TRAINING_DATA_Y')
TEST_DATA = config.get('main', 'TEST_DATA')
MODEL = config.get('main', 'MODEL')
SAVE = config.get('main', 'SAVE')

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df_x = pd.read_csv(TRAINING_DATA_X)
    df_y = pd.read_csv(TRAINING_DATA_Y)
    df_test = pd.read_csv(TEST_DATA)
    target_cols = df_y.columns.tolist()
    target_cols.remove('sig_id')
    predictions = None

    df = pd.merge(df_x, df_y, on = 'sig_id', how = 'left')

    # Preprocessing
    # Converting Categorical Data to Numerical Data
    cat_feats = CategoricalFeatures(df, 
                                    categorical_features=['cp_dose','cp_type','cp_time'], 
                                    encoding_type="label",
                                    handle_na=True, save=True)
    df = cat_feats.fit_transform()

    for target_col in target_cols:

        df = create_kfolds(df, target_col)

        for FOLD in FOLD_MAPPPING.keys():

            train_df = df[(df.kfold.isin(FOLD_MAPPPING.get(FOLD))) & (df['cp_type'] != "ctl_vehicle")].reset_index(drop=True)
            valid_df = df[(df.kfold==FOLD) & (df['cp_type'] != "ctl_vehicle")].reset_index(drop=True)

            ytrain = train_df[target_col].values
            yvalid = valid_df[target_col].values

            train_df = train_df.drop(["sig_id","kfold"] + target_cols, axis=1)
            valid_df = valid_df.drop(["sig_id", "kfold"] + target_cols, axis=1)

            # Sort columns based on train df
            valid_df = valid_df[train_df.columns]            
            
            # Oversampling
            train_df, ytrain = oversample_minority_svm(train_df, ytrain)

            # data is ready to train
            print(MODEL)
            clf = dispatcher.MODELS[MODEL]
            setattr(clf, 'random_state', 123) 
            print(target_col)
            clf.fit(train_df, ytrain)
            preds = clf.predict_proba(valid_df)[:, 1]
            print("Fold : ", FOLD)
            print("train_shape : ", str(train_df.shape))
            print("valid_shape : ", str(valid_df.shape))
            print('Train Class Ratio : ', str((np.count_nonzero(ytrain == 1)/ytrain.shape[0])*100))
            print('Valid Class Ratio : ', str((np.count_nonzero(yvalid == 1)/ytrain.shape[0])*100))
            # print('Class Ratio : ', str(np.count_nonzero(ytrain == 1)))
            print('AUC of {0} is '.format(target_col),metrics.roc_auc_score(yvalid, preds))
            print('Log Loss of {0} is '.format(target_col),metrics.log_loss(yvalid, preds))
            # auc = []
            # auc.append(metrics.roc_auc_score(yvalid, preds))
            # print(auc)
            # print(preds[:5])

        break

        if SAVE == True:
            joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
            joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
