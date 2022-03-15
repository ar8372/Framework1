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

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

def optimize(params, x,y):
    # params is a dictionary

    # note here we can also define params which are not of mmodel but make sure you take them out.. 
    # just like we did in NN model we had no_levels , no_units etc...
    # x is our features
    # y is target
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

    #space.Integer --> hp.quniform hp.choice('max_depth', np.arange(1, 13+1, dtype=int)
    #space.Categorical --> hp.choice
    #space.Real prior="uniform" --> hp.uniform
    # scope.int() to wrap it into int if giving float error
    param_space = {
        "max_depth": hp.quniform("max_depth", 3, 15, 1),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100,600,1)), #hp.choice('n_estimators', np.arange(100,600, dtype=int)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_featurs",0.01, 1),
    }
    optimization_function = partial(
        optimize, 
        x= X,
        y= Y
    )

    trials = Trials()

    result = fmin(
        fn=optimization_function,
        algo=tpe.suggest,
        space=param_space,
        max_evals=15,
        trials=trials
    )
    print(result)