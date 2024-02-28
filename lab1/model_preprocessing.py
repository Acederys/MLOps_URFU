import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler


def data_preprocessing(df, data=''):

    # Выделение целевой колонки
    x = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    y = df['Loan_Status']

    # Работа с отсутствующими значениями
    x.loc[:, 'Gender'] = x['Gender'].fillna('Male')
    x.loc[:, 'Self_Employed'] = x['Self_Employed'].fillna('No')
    x.loc[:, 'Dependents'] = x['Dependents'].fillna('0')
    x.loc[:, 'Loan_Amount_Term'] = x['Loan_Amount_Term'].fillna(360)
    x.loc[:, 'Credit_History'] = x['Credit_History'].fillna(0)

    # Преобразование (кодировка) целевой переменной
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name='Loan_Status')

    # Преобразование (кодировка) категориальных колонок
    category_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Dependents']
    enc = OrdinalEncoder()
    x[category_cols] = enc.fit_transform(x[category_cols])

    # Стандартизация числовых колонок
    numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    scaler = StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

    if data == 'train':
        x.to_csv("train/train_features.csv", index=False)
        y.to_csv("train/train_labels.csv", index=False)
    elif data == 'test':
        x.to_csv("test/test_features.csv", index=False)
        y.to_csv("test/test_labels.csv", index=False)


train = pd.read_csv('train/train.csv')
test = pd.read_csv('test/test.csv')
data_preprocessing(train, data='train')
data_preprocessing(test, data='test')
