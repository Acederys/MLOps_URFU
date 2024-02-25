import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import os

df = pd.read_csv("D:\MLops\work1\loan_data.csv")
# print(df)

# разбиваем данные на train/test в соотношении тренирвочный набор 70%, тестовый 30%
train, test = train_test_split(df, test_size = 0.2, random_state = 42)

# сохраняем в файлы на диске D:\MLops\work1
os.makedirs('D:\MLops\work1', exist_ok=True)
test.to_csv('D:/MLops/work1/test.csv')

os.makedirs('D:\MLops\work1', exist_ok=True)
train.to_csv('D:/MLops/work1/train.csv')



