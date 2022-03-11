import os
import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn import preprocessing 
from sklearn import metrics 
import joblib

from . import dispatcher 

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL") 

def predict():
    # Fold 0 is validation fold here
    df_test = pd.read_csv(TEST_DATA)  #- test
    test_idx = df_test.id.values.astype(int)
    predictions = None 

    for FOLD in range(5):
        df = df_test.copy() #- to prevent changing original test set 

        encoders = joblib.load(os.path.join('models',f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))

        for c in cols:
            # encode each column 
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist()) #- test

        # data is ready to train 
        clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds 
        else:
            predictions += preds 
    predictions /= 5 

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)),columns=['id','target'])
    
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}_sub.csv",index=False)

    
        
