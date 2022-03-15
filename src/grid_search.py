import pandas as pd 
import numpy as np 

from sklearn import ensemble 
from sklearn import metrics 

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

# for cross validation us cv
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/Mobile_train.csv")
    X = df.drop("price_range", axis=1).values 
    y = df.price_range.values 

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)

    classifier = pipeline.Pipeline(
        [
            ("scaling",scl),
            ("pca", pca),
            ("rf", rf)
        ]
    )

    # pca__ : it is the key
    param_grid = {
        "pca__n_components": [5,6,7,8,9,10],
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [1,3],
        "rf__criterion": ["gini", "entropy"],
    }

    #RandomizedSearchCV => n_iter = 10
    model = model_selection.GridSearchCV(estimator=classifier,
                                        param_grid=param_grid,
                                        scoring="accuracy",
                                        verbose=10,
                                        n_jobs=1,
                                        cv=5, #by deafault 5 fold 
                                        # if we have classification estimator then it will use stratified kfold for creating folds 
                                        # for regression kfold
                                        )
    model.fit(X,y)

    print(model.best_score_)
    print(model.best_estimator_.get_params())