import pandas as pd 
import numpy as np 

from sklearn import ensemble 
from sklearn import metrics 

from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

# for cross validation use cv
from sklearn import model_selection

from functools import partial
from skopt import space, gp_minimize

def optimize(params, param_names, x,y):
    # note here we can also define params which are not of mmodel but make sure you take them out.. 
    # just like we did in NN model we had no_levels , no_units etc...
    # x is our features
    # y is target
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    for i,(train_idx, test_idx) in enumerate(kf.split(X=x, y=y)):
        xtrain= x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)

        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies) # since minimize it (gp_mimimize)


if __name__ == "__main__":
    df = pd.read_csv("../input/Mobile_train.csv")
    X = df.drop("price_range", axis=1).values 
    Y = df.price_range.values 


    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100,600, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]
    optimization_function = partial(
        optimize, # takes list of params as argument
        param_names=param_names,
        x= X,
        y= Y
    )
    result = gp_minimize(
        optimization_function,
        dimensions = param_space,
        n_calls = 15,
        n_random_starts = 10,
        verbose= 10
    )
    print(
        dict(
        zip(param_names, result.x)
        )
    )