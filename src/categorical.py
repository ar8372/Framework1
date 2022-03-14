from sklearn import preprocessing

"""
- Label Encoding
- One hot encoding 
- binarization
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na = False):
        """
        df: pandas dataframe 
        categorical_features: list of column names, e.g. ["ord_1","nom_0"]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
    
        self.df =df 
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        self.handle_na = handle_na
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True) # creates deep copy 

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl # to do some transformation later
        return self.output_df 

    def _label_binarization(self):
        old_names = []
        new_names = []
        store_df = None
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values) #  array 
            old_names.append(c)
            new_names+= [c+ f"__bin_{j}" for j in range(val.shape[1])]
            if store_df is None:
                store_df = pd.DataFrame(val)
            else:
                store_df = pd.concat([store_df, pd.DataFrame(val)], axis=1)
            self.binary_encoders[c] = lbl
        self.output_df = self.output_df.drop(old_names, axis=1)
        temp_col  = self.output_df.columns.tolist()
        self.output_df= pd.concat([self.output_df, store_df],axis=1) 
        self.output_df.columns = temp_col+ new_names
        return self.output_df

    def _one_hot(self):
        # all columns at once 
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        self.ohe = ohe
        # sanity check --\ transform in same order as fit
        return ohe.transform(self.df[self.cat_feats].values) 

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:,c] = dataframe.loc[:, c].astype(str).fillna("-999999")

        if self.enc_type == "label":
            old_names = []
            new_names = []
            store_df = []
            for c,lbl in self.label_encoders.items():
                val = lbl.transform(dataframe[c].values) #  array 
                old_names.append(c)
                store_df.append(list(val))
            dataframe[old_names] =  pd.DataFrame(store_df).T
            return dataframe

        elif self.enc_type == "binary":
            old_names = []
            new_names = []
            store_df = None
            for c,lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values) #  array 
                old_names.append(c)
                new_names += [c+ f"__bin_{j}" for j in range(val.shape[1])]
                if store_df is None:
                    store_df = pd.DataFrame(val)
                else:
                    store_df = pd.concat([store_df, pd.DataFrame(val)], axis=1)
            dataframe = dataframe.drop(old_names, axis=1)
            temp_col = dataframe.columns.tolist()
            dataframe = pd.concat([dataframe, store_df], axis=1)
            dataframe.columns = temp_col + new_names 
            return dataframe

        elif self.enc_type == "ohe":
            # No need since both train and test are converted to one hot together
            dataframe = self.ohe.transform(dataframe[self.cat_feats])
            return dataframe   
        else:
            raise Exception(f"Not valid encoding type: {self.env_type}")     
        

if __name__ == "__main__":
    """
    # for LabelEncoding and BinaryEncoding
    few class of test may not be in train then it will cause error
    sol:- combine tran-test and fit it on label encoder
    #- to maintain train test separable 
    # M1
    df_test["isTrain"] = 0
    df["isTrain"] = 1       
    full_data[full_data['isTrain'== 1]] #=> to extract
    # M2 
    train_idx = df["id"].values 
    test_idx = df_test["id"].values 
    full_data[full_data["id"].isin(train_idx)] #=> to extract
    # M3 
    train_len = df.shape[0]
    full_data_transformed[:train_len]

    df_test['target'] = -1
    full_data = pd.concat([df, df_test])
    """
    import pandas as pd
    import numpy as np
    from sklearn import linear_model

    df = pd.read_csv("../input/train_cat2.csv")#.head(500)
    df_test =pd.read_csv("../input/test_cat2.csv")#.head(500)
    sample = pd.read_csv("../input/sample_cat2.csv")
    # M1 
    # df_test["isTrain"] = 0
    # df["isTrain"] = 1
    # M2 
    train_idx = df["id"].values 
    test_idx = df_test["id"].values 
    # M3 
    train_len = df.shape[0]


    df_test['target'] = -1
    full_data = pd.concat([df, df_test])

    cols = [c for c in df.columns if c not in ["id","target"]]
    print(cols)
    cat_feats = CategoricalFeatures(full_data, 
                                    categorical_features= cols, 
                                    encoding_type="ohe",
                                    handle_na=True)
    cat_feats = cat_feats.fit_transform()
    X = cat_feats[:train_len, :]
    X_test = cat_feats[train_len:, :]
    print(X.shape)
    print()
    print(X_test.shape)

    model = linear_model.LogisticRegression()
    model.fit(X, df["target"])
    pred = model.predict_proba(X_test)[:,1]

    sample["target"] = pred
    sample.to_csv("../models/sub1_cat2.csv", index=False)