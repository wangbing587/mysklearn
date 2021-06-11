import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def train(model, x, y, epochs=100, scale=True, n_splits=5):
    if scale:
        x = StandardScaler().fit_transform(x)
    trainpred = np.zeros((x.shape[0], epochs))
    trainprob = np.zeros((x.shape[0], epochs))
    
    for i in tqdm(range(epochs)):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model.fit(x_train, y_train)
            print(model.score(x_val, y_val))
            trainpred[val_index, i] = model.predict(x_val)
            trainprob[val_index, i] = model.predict_proba(x_val)[:, 1]
    return trainpred, trainprob
