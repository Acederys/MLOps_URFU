import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("loan_data.csv")
# print(df)

# разбиваем данные на train/test в соотношении тренировочный набор 70%, тестовый 20%
train, test = train_test_split(df, test_size=0.2, random_state=42)

# сохраняем тренировочный и тестовый наборы данных
test.to_csv('test/test.csv', index=False)
train.to_csv('train/train.csv', index=False)



