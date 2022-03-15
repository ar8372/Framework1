import os
import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn import preprocessing 
from sklearn import metrics 
import joblib

from . import dispatcher 

# get as env variable # We remove it for making inference kernel
# TEST_DATA = os.environ.get("TEST_DATA") #changed to test_data_path
# MODEL = os.environ.get("MODEL") # changed to model_type

def predict(test_data_path, model_type, model_path):
    # Fold 0 is validation fold here
    df_test = pd.read_csv(test_data_path)  #- test
    test_idx = df_test.id.values
    predictions = None 

    for FOLD in range(5):
        df = df_test.copy() #- to prevent changing original test set 

        encoders = joblib.load(os.path.join(model_path,f"{model_type}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path,f"{model_type}_{FOLD}_columns.pkl"))

        for c in cols:
            # encode each column 
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist()) #- test

        # data is ready to train 
        clf = joblib.load(os.path.join(model_path,f"{model_type}_{FOLD}.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds 
        else:
            predictions += preds 
    predictions /= 5 

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id","target"])
    
    return sub

if __name__ == "__main__":
    test_data_path="input/test_cat2.csv"
    model_type="rfc"
    model_path = "models"
    submission = predict(test_data_path, model_type, model_path)
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/{model_type}_sub.csv",index=False)

    
        
