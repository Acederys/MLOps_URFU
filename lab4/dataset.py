from catboost.datasets import titanic
import pandas as pd

titanic_train, titanic_test = titanic()

df = pd.concat([titanic_train, titanic_test])[['Pclass', 'Sex', 'Age', 'Survived']]

df['Age'] = df['Age'].fillna(df['Age'].mean())

df.to_csv('titanic.csv', index=False)
