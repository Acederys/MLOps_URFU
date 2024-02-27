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


# Старый код

# """РАБОТА С ЧИСЛОВЫМИ ПЕРЕМЕННЫМИ"""
#
# df_train = pd.read_csv('train/train.csv')
# X_train = df_train[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
# y_train = df_train[['Loan_Status']]
# df_test = pd.read_csv('test/test.csv')
# X_test = df_test[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
# y_test = df_test[['Loan_Status']]
#
# # Замена пропущенных значений на 1, значений "3+" на 4 и Male на 1
# for index, row in X_train.iterrows():
#   if row['Dependents'] == "3+":
#     X_train.loc[index, 'Dependents'] = "3"
#   elif pd.isnull(row['Dependents']):
#     X_train.loc[index, 'Dependents'] = "1"
#
# # Конвертируем колонку Dependents в float
# X_train['Dependents'] = X_train['Dependents'].astype(float)
#
# # Замена пропущенных значений на 1, значений "3+" на 4 и Male на 1
# for index, row in X_test.iterrows():
#   if row['Dependents'] == "3+":
#     X_test.loc[index, 'Dependents'] = "3"
#   elif pd.isnull(row['Dependents']):
#     X_test.loc[index, 'Dependents'] = "1"
#
# # Конвертируем колонку Dependents в float
# X_test['Dependents'] = X_test['Dependents'].astype(float)
#
# # Заменяем пропущенные значение в колонке Gender на значения male
# for index, row in X_train.iterrows():
#   if pd.isnull(row['Gender']):
#     # print(index)
#     X_train.loc[index, 'Gender'] = "Male"
#
# # Заменяем пропущенные значение в колонке Gender на значения male
# for index, row in X_test.iterrows():
#   if pd.isnull(row['Gender']):
#     # print(index)
#     X_test.loc[index, 'Gender'] = "Male"
#
# # Заменяем пропущенные значение в колонке Self_Employed на значения No
# for index, row in X_train.iterrows():
#   if pd.isna(row['Self_Employed']):
#     X_train.loc[index, 'Self_Employed'] = "No"
#
# # Заменяем пропущенные значение в колонке Self_Employed на значения No
# for index, row in X_test.iterrows():
#   if pd.isna(row['Self_Employed']):
#     X_test.loc[index, 'Self_Employed'] = "No"
#
# print(f" Credit_History {X_train['Self_Employed'].unique()}")
# print(f" Credit_History {X_test['Self_Employed'].unique()}")
#
# # Пропущенные значения в колонке Loan_Amount_Term заменить на среднее по колонке
# X_train['Loan_Amount_Term'] = (X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean())).round(0)
#
# # Пропущенные значения в колонке Loan_Amount_Term заменить на среднее по колонке
# X_test['Loan_Amount_Term'] = (X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mean())).round(0)
#
# # Преобразуем к типу object
# X_train['Credit_History'] = X_train['Credit_History'].astype('object')
# # Пропущенные значения в колонке Credit_History заменить на 0
# for index, row in X_train.iterrows():
#   if pd.isna(row['Credit_History']) or row['Credit_History'] == 0:
#     X_train.loc[index, 'Credit_History'] = 'No'
#   elif row['Credit_History'] == 1 :
#     X_train.loc[index, 'Credit_History'] = 'Yes'
#
# # Преобразуем к типу object
# X_test['Credit_History'] = X_test['Credit_History'].astype('object')
# # Пропущенные значения в колонке Credit_History заменить на 0
# for index, row in X_test.iterrows():
#   if pd.isna(row['Credit_History']) or row['Credit_History'] == 0:
#     X_test.loc[index, 'Credit_History'] = 'No'
#   elif row['Credit_History'] == 1 :
#     X_test.loc[index, 'Credit_History'] = 'Yes'
#
# print(f" Credit_History {X_train['Credit_History'].unique()}")
# print(f" Credit_History {X_test['Credit_History'].unique()}")
#
#
# print(X_test.info())
#
# print(X_train.info())
#
# # from sklearn.preprocessing import MinMaxScaler
#
# # minmax = MinMaxScaler(feature_range = (0, 1))
# # # minmax()
#
# # numerical_df = df[['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']]
# # minmax.fit(numerical_df)
#
# # numerical_df = minmax.fit_transform(numerical_df)
#
# """РАБОТА С КАТЕГОРИАЛЬНЫМИ ПЕРЕМЕННЫМИ"""
#
# # Пропущенные значения в колонке Self_Employed заменить на No
# for index, row in X_train.iterrows():
#   if pd.isna(row['Self_Employed']):
#     # print(index)
#     X_train.loc[index, 'Self_Employed'] = "No"
#
# # Пропущенные значения в колонке Self_Employed заменить на No
# for index, row in X_test.iterrows():
#   if pd.isna(row['Self_Employed']):
#     # print(index)
#     X_test.loc[index, 'Self_Employed'] = "No"
#
# # Пропущенные значения в колонке Property_Area заменить на No
# for index, row in X_train.iterrows():
#   if pd.isna(row['Property_Area']):
#     # print(index)
#     X_train.loc[index, 'Property_Area'] = "No"
#
# category_X_train = X_train[['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']]
#
# # Пропущенные значения в колонке Property_Area заменить на No
# for index, row in X_test.iterrows():
#   if pd.isna(row['Property_Area']):
#     # print(index)
#     X_test.loc[index, 'Property_Area'] = "No"
#
# category_X_test = X_test[['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']]
#
# cat_columns = []
# num_columns = []
#
# for column_name in X_test.columns:
#     if (X_test[column_name].dtypes == object):
#         cat_columns +=[column_name]
#     else:
#         num_columns +=[column_name]
#
# print('categorical columns:\t ',cat_columns, '\n len = ',len(cat_columns))
#
# print('numerical columns:\t ',  num_columns, '\n len = ',len(num_columns))
#
# cat_columns = []
# num_columns = []
#
# for column_name in X_train.columns:
#     if (X_train[column_name].dtypes == object):
#         cat_columns +=[column_name]
#     else:
#         num_columns +=[column_name]
#
# print('categorical columns:\t ',cat_columns, '\n len = ',len(cat_columns))
#
# print('numerical columns:\t ',  num_columns, '\n len = ',len(num_columns))
#
# """Приведем целевые метки и 0 и 1. Для этого воспользуемся объектом LabelEncoder() из модуля preprocessing
#
# Применение преобразований уже стандартное для нас
#
# Создаем объект
# обучаем методом .fit()
# Смотрим что получилось
# """
#
# Label_train = LabelEncoder()
# Label_train.fit(y_train) # задаем столбец, который хотим преобразовать
# print(f'в аттрибуте .classes_ хранится информация "какой класс как шифруетс y_train{Label_train.classes_}') # в аттрибуте .classes_ хранится информация "какой класс как шифруется"
#
# Label_test = LabelEncoder()
# Label_test.fit(y_test) # задаем столбец, который хотим преобразовать
#  # в аттрибуте .classes_ хранится информация "какой класс как шифруется"
# print(f'в аттрибуте .classes_ хранится информация "какой класс как шифруетс y_test{Label_test.classes_}') # в аттрибуте .classes_ хранится информация "какой класс как шифруется"
#
# target_train = Label_train.transform(y_train)
# print(f'Преобразование целевой пересенной y_train {target_train}')
#
# target_test = Label_test.transform(y_test)
# print(f'Преобразование целевой пересенной y_test {target_test}')
#
#
# print(f'категориальные переменные {cat_columns}')
#
# # создаем "полотно", на котором будем "рисовать" графики
# fig, axs = plt.subplots(1,5,figsize=(20,  5))
# #Говорим что у нас будет 1 строка и 5 столбца
#
# X_train.hist(column = num_columns, ax = axs )
#
# # создаем "полотно", на котором будем "рисовать" графики
# fig, axs = plt.subplots(1,5,figsize=(20,  5))
#
# #Говорим что у нас будет 1 строка и 5 столбца
#
# X_test.hist(column = num_columns, ax = axs )
#
# """СТАНДАРТИЗАЦИЯ И НОРМАЛИЗАЦИЯ ПРИЗНАКОВ (ЧИСЛОВЫЕ И КАТЕГОРИАЛЬНЫЕ)"""
#
# # Приведение числовых признаков
# scale_test = StandardScaler()
# scale_test.fit(X_test[num_columns])
# print(f'Приведение числовых признаков test {scale_test.mean_},{scale_test.scale_}')
#
# # Приведение числовых признаков
# scale_train = StandardScaler()
# scale_train.fit(X_train[num_columns])
# print(f'Приведение числовых признаков train {scale_train.mean_}, {scale_train}')
#
# scaled_test = scale_test.transform(X_test[num_columns])
# df_standard_test = pd.DataFrame(scaled_test, columns= num_columns)
#
# scaled_train = scale_train.transform(X_train[num_columns])
# df_standard_train = pd.DataFrame(scaled_train, columns= num_columns)
#
# df_standard_train.hist(figsize = (20,5), layout= (1,5))
#
# df_standard_test.hist(figsize = (20,5), layout= (1,5))
#
# # Приведение категориальных признаков
# # создадим объект класса OneHotEncoder
# # параметр sparse = True выдал бы результат в сжатом формате
# onehotencoder = OneHotEncoder(sparse_output = False)
#
# df_onehot_train = pd.DataFrame(onehotencoder.fit_transform(X_train[cat_columns]))
#
# # Приведение категориальных признаков
# # создадим объект класса OneHotEncoder
# # параметр sparse = True выдал бы результат в сжатом формате
#
# df_onehot_test = pd.DataFrame(onehotencoder.fit_transform(X_test[cat_columns]))
#
#
# # Объединяем нормализованные и стандиртизированные признаки
# df1_train = pd.DataFrame(df_standard_train)
# df2_train = pd.DataFrame(df_onehot_train)
#
# result_df_train = pd.concat([df1_train, df2_train], axis=1)
#
#
# # Объединяем нормализованные и стандиртизированные признаки
# df1_test = pd.DataFrame(df_standard_test)
# df2_test = pd.DataFrame(df_onehot_test)
#
# result_df_test = pd.concat([df1_test, df2_test], axis=1)
#
#
# plt.figure(figsize=(15, 8))
# sns.heatmap(result_df_train.corr(), annot=True, cmap="YlGnBu")
#
# plt.figure(figsize=(15, 8))
# sns.heatmap(result_df_test.corr(), annot=True, cmap="YlGnBu")
#
# with open('test/y_test.npy', 'wb') as f:
#     np.save(f, target_test)
# with open('train/y_train.npy', 'wb') as f:
#     np.save(f, target_train)
#
# result_df_test.to_csv('test/X_test.csv')
# result_df_train.to_csv('train/X_train.csv')