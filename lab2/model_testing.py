import pandas as pd
from sklearn.metrics import accuracy_score
import pickle


file3 = r'test/test_features.csv'
file4 = r'test/test_labels.csv'

X_test = pd.read_csv(file3)
y_test = pd.read_csv(file4)

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')