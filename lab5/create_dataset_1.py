import pandas as pd
import numpy as np

# Генерация данных
n_obs = 800
id_column = range(1, n_obs + 1)
gender = np.random.randint(0, 2, size=n_obs)
age = np.random.randint(18, 41, size=n_obs)
medal = np.random.randint(0, 14, size=n_obs)
salary = 2 * age + 8 * medal + gender + 25 * np.random.randn(n_obs) + 30
salary = np.round(salary).astype(int)
# Создание DataFrame
df = pd.DataFrame({
    'id': id_column,
    'пол': gender,
    'возраст': age,
    'наличие_медали': medal,
    'зарплата_тыс': salary
})

df.to_cdv('salary_sport.csv')