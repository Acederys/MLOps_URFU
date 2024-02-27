import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

test_features = pd.read_csv('test/test_features.csv')
test_labels = pd.read_csv('test/test_labels.csv')

with open('model.pickle', 'rb') as f:
    log_reg = pickle.load(f)

y_pred = log_reg.predict(test_features)

accuracy = round(accuracy_score(test_labels, y_pred), 3)
precision = round(precision_score(test_labels, y_pred), 3)
recall = round(recall_score(test_labels, y_pred), 3)
f1 = round(f1_score(test_labels, y_pred), 3)

print('Accuracy: {} || Precision: {} || Recall: {} || F1 score: {}'.format(accuracy, precision, recall, f1))
