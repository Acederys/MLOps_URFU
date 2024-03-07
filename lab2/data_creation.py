import pandas as pd
from sklearn.model_selection import train_test_split

df  = pd.read_csv('SAHeart.csv')
df = df.drop(columns=['row.names'])
# print(df)


# не забываем удалить целевую переменную цену из признаков
X,y = df.drop(columns = ['chd']), df['chd']

# разбиваем на тестовую и валидационную
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

DF_train = X_train.copy()
DF_train['chd'] = y_train
DF_train.to_csv('train/train.csv', index=False)
# объединили в один датасет признаки и целевую переменную

DF_val = X_val.copy()
DF_val['chd'] = y_val
DF_val.to_csv('test/val.csv', index=False)
# объединили в один датасет признаки и целевую переменную