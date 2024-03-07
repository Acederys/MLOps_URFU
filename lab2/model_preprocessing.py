import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler


def data_preprocessing(df, data=''):

    # Выделение целевой колонки
    x = df.drop(['chd'], axis=1)
    y = df['chd']

    # Работа с отсутствующими значениями: пропущенных значений нет

    # Преобразование (кодировка) целевой переменной
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name='chd')

    # Преобразование (кодировка) категориальных колонок
    category_cols = ['famhist']
    enc = OrdinalEncoder()
    x[category_cols] = enc.fit_transform(x[category_cols])

    # Стандартизация числовых колонок
    numeric_cols = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']
    scaler = StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    if data == 'train':
        x.to_csv("train/train_features.csv", index=False)
        y.to_csv("train/train_labels.csv", index=False)
    elif data == 'test':
        x.to_csv("test/test_features.csv", index=False)
        y.to_csv("test/test_labels.csv", index=False)


train = pd.read_csv('train/train.csv')
test = pd.read_csv('test/val.csv')
data_preprocessing(train, data='train')
data_preprocessing(test, data='test')
