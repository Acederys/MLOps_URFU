import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import os


# path = 'D:/MLops/work1/'

df = pd.read_csv('D:/MLops/work1/test.csv')
# print(df)

#######################################################################################
######################  ПРЕОБРАЗОВАНИЯ ЧИСЛОВЫХ ПРИЗНАКОВ #############################
#######################################################################################

# Замена пропущенных значений на 1, значений "3+" на 4 и Male на 1
for index, row in df.iterrows():
  if row['Dependents'] == "3+":
    df.loc[index, 'Dependents'] = "4"
  elif pd.isnull(row['Dependents']):
    df.loc[index, 'Dependents'] = "1"
  elif row['Dependents'] == "Male":
    df.loc[index, 'Dependents'] = "1"

# Заменяем пропущенные значение в колонке Gender на значения male
for index, row in df.iterrows():
  if pd.isnull(row['Gender']):
    # print(index)
    df['Gender'] = df.loc[index, 'Gender'] = "Male"

# Заменяем пропущенные значение в колонке Self_Employed на значения No
for index, row in df.iterrows():
  if pd.isnull(row['Self_Employed']):
    df['Self_Employed'] = df.loc[index, 'Self_Employed'] = "No"

# Пропущенные значения в колонке Loan_Amount_Term заменить на среднее по колонке
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

# Пропущенные значения в колонке Credit_History заменить на 0
for index, row in df.iterrows():
  if pd.isnull(row['Credit_History']):
    df.loc[index, 'Credit_History'] = 0

# Пропущенные значения в колонке Loan_Amount_Term заменить на среднее по колонке
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

# Конвертируем колонку Dependents в float
df['Dependents'] = df['Dependents'].astype(float)

#######################################################################################
###########  НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ ТИПА FLOAT ЧЕРЕЗ MinMaxScaler ############
#######################################################################################

minmax = MinMaxScaler(feature_range = (0, 1))
# minmax()

numerical_df = df[['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
minmax.fit(numerical_df)

numerical_df = minmax.fit_transform(numerical_df)


#######################################################################################
######################  ПРЕОБРАЗОВАНИЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ  ######################
#######################################################################################

# ДРОПНУТЬ Loan_ID
# КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ - 'Gender'(2), 'Married'(2), 'Education'(2), 'Self_Employed' (['No', 'Yes', nan]) , 'Property_Area'(['Rural', 'Urban', 'Semiurban'])
# 'Loan_Status'(2) отдельно

# Пропущенные значения в колонке Self_Employed заменить на No
for index, row in df.iterrows():
  if pd.isna(row['Self_Employed']):
    # print(index)
    df.loc[index, 'Self_Employed'] = "No"


# Пропущенные значения в колонке Property_Area заменить на No
for index, row in df.iterrows():
  if pd.isna(row['Property_Area']):
    # print(index)
    df.loc[index, 'Property_Area'] = "No"

category_df = df['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

#######################################################################################
########  НОРМАЛИЗАЦИЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ТИПА OBJECT ЧЕРЕЗ OneHotEncoder #######
#######################################################################################
# создадим объект класса OneHotEncoder
# параметр sparse = True выдал бы результат в сжатом формате
onehotencoder = OneHotEncoder(sparse_output = False)

category_df = pd.DataFrame(onehotencoder.fit_transform(category_df))



####################################################################################
################### ОБЪЕДИНЕНИЕ DF И СОХРАНЕНИЕ В ОТДЕЛЬНЫЙ ФАЙЛ ###################
####################################################################################
# Объединяем результаты преобразований

df1 = pd.DataFrame(numerical_df)
df2 = pd.DataFrame(category_df)

result_df = pd.concat([df1, df2], axis=1)
print(result_df)

# print(norm_test_data)

np.savetxt('D:/MLops/work1/result_df.csv', result_df, delimiter =',')

####################################################################################
#############################№№№№№№№№№№№№№№№№№№№№№##################################
####################################################################################