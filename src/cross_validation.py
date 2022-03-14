import pandas as pd 
from sklearn import model_selection 
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
from collections import defaultdict, Counter
import random

import pandas as pd
import tqdm

"""
--- binary classification 
--- multiclass classification
--- multilabel classification 
--- single column regression 
--- multi column regression 
--- holdout
"""

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type = "binary_classification",
            multilabel_delimiter=",",
            num_folds = 5,
            random_state=42,
        ):
        self.dataframe = df 
        self.target_cols = target_cols 
        self.num_targets = len(target_cols)
        self.shuffle = shuffle 
        self.problem_type = problem_type
        self.multilabel_delimiter = multilabel_delimiter
        self.num_folds = num_folds 
        self.random_state = random_state 

        assert type(self.dataframe) is pd.DataFrame, "Not a dataframe"
        if self.shuffle == True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)  #frac=1 mean frac of item to be returned is all items
        self.dataframe['fold'] = -1 

    def split(self):
        problem_name = self.problem_type 
        if problem_name in ("biniary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("num_targets != 1")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique() # all classes 
            if unique_values == 1:
                raise Exception("Only one unique value found")
            elif unique_values > 1:
                # use StratifiedKFold for both balanced and unbalanced targets 
                skf = model_selection.StratifiedKFold(n_splits= self.num_folds,shuffle=False) # we shuffle separately 
                for fold, (train_idx, val_idx) in enumerate(skf.split(X=self.dataframe, y= self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'fold'] = fold 
                self.dataframe.fold = self.dataframe.fold.astype(int)

                # Display distribution>
                for i in range(self.num_folds):
                    t = self.dataframe.loc[self.dataframe['fold']==i, target].value_counts()
                    t = (t/t.sum()).tolist()
                    print(f"For fold {i} class ratio is: ",end="")
                    for j in range(unique_values):
                        print(f"{t[j]},",end="")
                    print()

        elif problem_name in ("single_col_regression", "multi_col_regression"):
            # since it is continuous value we can't stratify on target 
            # M1 -> KFold 
            # M2 -> split such that each fold has same distribution

            # M1 
            if self.num_targets != 1 and problem_name == "single_col_regression":
                raise Exception(f"{self.num_targets} != 1 for {problem_name}")
            if self.num_targets <2 and problem_name == "multi_col_regression":
                raise Exception(f"{self.num_targets} <2 for {problem_name}")
            kf = model_selection.KFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                self.dataframe.loc[val_idx,'fold'] = fold 
            # Display distribution>
            target = self.target_cols[0]
            sns.kdeplot(data=self.dataframe, x=target,hue='fold',fill=True)
            plt.show()

        
        elif problem_name.startswith("holdout_"):
            # use for time series Note: don't shuffle data
            # when huge dataset in classification or regression like 10M so take 100,000 only 
            holdout_percentage = int(self.problem_type.split("_")[1]) #picked the no 
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            # hold -->0 , use -->1 
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, 'fold'] = 0
            self.dataframe.loc[(len(self.dataframe) - num_holdout_samples) :, 'fold'] = 1 
        
        elif problem_name.startswith("multilabel_classification"):
            if problem_name == "multilabel_classification1":
                # M1
                # Stratified KFold split based on number of classes present in a datapoint 
                if self.num_targets != 1:
                    raise Exception(f"{self.num_targets} != 1 for {problem_name}")
                targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
                skf = model_selection.StratifiedKFold(n_splits= self.num_folds,
                                                            shuffle=False) # we shuffle separately 
                for fold, (train_idx, val_idx) in enumerate(skf.split(X=self.dataframe, y= targets)):
                    self.dataframe.loc[val_idx, 'fold'] = fold 
                self.dataframe.fold = self.dataframe.fold.astype(int)
            elif problem_name == "multilabel_classification2":
                # M2 :- https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/make_folds.py
                #The idea was to spread rare classes among folds first, and only then spread more 
                #common classes, because rare classes are more affected by chance.
                df = self.dataframe
                n_folds = self.num_folds
                cls_counts = Counter(cls for classes in df['target'].str.split(self.multilabel_delimiter)
                                    for cls in classes)
                fold_cls_counts = defaultdict(int)
                print(cls_counts)
                folds = [-1] * len(df)
                tj = 0
                for item in tqdm.tqdm(df.sample(frac=1, random_state=42).itertuples(),
                                    total=len(df)):
                    # picking the rarest label in given row
                    cls = min(item.target.split(self.multilabel_delimiter), key=lambda cls: cls_counts[cls])
                    print(cls, item.target.split(self.multilabel_delimiter),'this is rare')
                    fold_counts = [(f, fold_cls_counts[f, cls]) for f in range(n_folds)]
                    print(fold_counts,'this is fold count')
                    min_count = min([count for _, count in fold_counts])
                    print(min_count)
                    random.seed(item.Index)
                    fold = random.choice([f for f, count in fold_counts
                                        if count == min_count])
                    folds[item.Index] = fold
                    for cls in item.target.split(self.multilabel_delimiter):
                        fold_cls_counts[fold, cls] += 1
                    if tj <15:
                        tj += 1
                    else:
                        break
                df['fold'] = folds 
                self.dataframe = df
        else:
            # none 
            raise Exception("Problem type not understood!")

        return self.dataframe

if __name__ == "__main__":
    #- Binary and Multiclass classification problem
    # df = pd.read_csv("input/train_multiclass.csv")
    # target_cols =  ['target']
    # problem_type = 'multiclass_classification'
    # shuffle=True

    #- Single col Regression
    # df = pd.read_csv("../input/train_reg.csv")
    # target_cols =  ['SalePrice']
    # problem_type = 'single_col_regression'
    # shuffle=True

    #- Holdout
    # df = pd.read_csv("../input/train_reg.csv")
    # target_cols =  ['SalePrice']
    # problem_type = 'holdout_50'
    # shuffle=True

    #- multilabel M2 , multilabel M1
    df = pd.read_csv("../input/train_multilabel.csv")
    for i,col in enumerate(df.columns[1:]):
        df[col] = df[col]*(i+1)
    df[df.columns[1:]]= df[df.columns[1:]].replace(0,"_")
    df['target'] = df[df.columns[1:]].apply(lambda x: "_".join(x.dropna().astype(str)), axis=1)
    df['target']= df['target'].apply(lambda x: x.replace("_"," "))
    df['target']= df['target'].apply(lambda x: "_".join([str(i) for i in list(map(int, x.split())) ]))
    df = df[['id','target']]
    target_cols =  ['target']
    problem_type = 'multilabel_classification2'
    multilabel_delimiter = "_"
    shuffle=True

    cv = CrossValidation(df, shuffle=shuffle, target_cols=target_cols,
                        problem_type=problem_type,multilabel_delimiter=multilabel_delimiter)
    df_split = cv.split()

    print(df_split.head())
    print()
    print(df_split.fold.value_counts())
