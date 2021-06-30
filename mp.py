import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings
from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
from pathos.multiprocessing import cpu_count
warnings.filterwarnings("ignore")
from sklearn.svm import SVC


def getparams(C, gamma):
    df_train = data[data[label] != 'unknown']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    x = df_train.drop(label, axis=1).values
    y = df_train[label].values
    f1scores = []
    for k in range(100):
        y_pred = np.empty_like(y)
        for train_index, val_index in skf.split(x, y):
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]
            std = StandardScaler()
            x_train_std = std.fit_transform(x_train)
            x_val_std = std.transform(x_val)
            clf = SVC(kernel='rbf', C=C, gamma=gamma,
                      class_weight='balanced',
                      probability=True)
            clf.fit(x_train_std, y_train)
            y_hat = clf.predict(x_val_std)
            y_pred[val_index] = y_hat
        f1scores.append(f1_score(y, y_pred, average='macro'))
    return C, gamma, sum(f1scores) / 100


label = 'markers'
n_splits = 5

Cs = []
gammas = []
for C in [1, 2, 4, 8, 16, 32, 64, 128]:
    for gamma in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2]:
        Cs.append(C)
        gammas.append(gamma)

data0 = pd.read_csv('MS2PGCsm.csv', index_col=0)

data = data0[data0[label] != 'unknown']
core = int(cpu_count() * 0.95)
pool3 = Pool(core)
a = pool3.map(getparams, Cs, gammas)
pool3.close()
pool3.join()
