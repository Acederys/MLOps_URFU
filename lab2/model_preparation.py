import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

"""Обучение модели"""
file1 = r'D:\1.УЧЕБА ИИИ 2023-2025\Второй семестр 2024\3. АВТОМАТИЗАЦИЯ МО\Задание 2\train_features.csv'
file2 = r'D:\1.УЧЕБА ИИИ 2023-2025\Второй семестр 2024\3. АВТОМАТИЗАЦИЯ МО\Задание 2\train_labels.csv'

X_train = pd.read_csv(file1)

y_train = pd.read_csv(file2)

model = LogisticRegression()
model.fit(X_train, np.ravel(y_train,order='C'))

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)
