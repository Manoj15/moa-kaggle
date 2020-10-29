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

    for target_col in target_cols:

        df = create_kfolds(df, target_col)

        for FOLD in FOLD_MAPPPING.keys():

            train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
            valid_df = df[df.kfold==FOLD].reset_index(drop=True)

            ytrain = train_df[target_col].values
            yvalid = valid_df[target_col].values

            train_df = train_df.drop(["sig_id","kfold"] + target_cols, axis=1)
            valid_df = valid_df.drop(["sig_id", "kfold"] + target_cols, axis=1)

            valid_df = valid_df[train_df.columns]

            label_encoders = {}
            for c in ['cp_dose','cp_type','cp_time']:
                lbl = preprocessing.LabelEncoder()
                train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
                valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
                df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
                lbl.fit(train_df[c].values.tolist() + 
                        valid_df[c].values.tolist() + 
                        df_test[c].values.tolist())
                train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
                valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
                label_encoders[c] = lbl
            
            # data is ready to train
            clf = dispatcher.MODELS[MODEL]
            print(target_col)
            clf.fit(train_df, ytrain)
            print('FINISHED TRAINING')
            print(ytrain)
            preds = clf.predict_proba(valid_df)[:, 1]

            if FOLD == 0:
                predictions = preds
            else:
                predictions += preds
    
            predictions /= 5

            print('AUC of {0} is '.format(target_col),metrics.roc_auc_score(yvalid, predictions))

            break

            if SAVE == True:
                joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
                joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")