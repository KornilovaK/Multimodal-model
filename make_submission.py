import pandas as pd
import numpy as np
import joblib
import re
import ast

import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
import lightautoml

from baseline import make_dataset 

pd.options.mode.chained_assignment = None

def prepare_test_data(df):
    columns = df.columns.values[2:]
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df[columns]

def main():
    df = make_dataset()
    X_test = prepare_test_data(df)
    
    model = joblib.load('model_lightautoml_without.pkl')
    predictions_prob = model.predict(X_test).data[:, 0]
    predictions = (predictions_prob >= 0.5).astype(int) 

    df['target'] = predictions
    submission = df[['variantid1', 'variantid2', 'target']]
    submission.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    main()
