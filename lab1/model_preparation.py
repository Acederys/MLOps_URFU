import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

train_features = pd.read_csv('train/train_features.csv')
train_labels = pd.read_csv('train/train_labels.csv')

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(train_features, train_labels.values.ravel())

with open('model.pickle', 'wb') as f:
    pickle.dump(log_reg, f)
