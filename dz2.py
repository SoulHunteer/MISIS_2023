import pandas as pd
from sklearn.linear_model import LinearRegression

# загрузить базу спортсменов из csv файла
df = pd.read_csv(r'C:\Users\Sergey\PycharmProjects\MISIS_Python\MISIS_2022\season_2\data\athletes_sochi.csv')

# выбрать нужные столбцы
df = df[['age', 'height', 'weight', 'gold_medals']]

# удалить строки, в которых есть пропущенные значения
df = df.dropna()

# определить правильные ответы
y_true = df['gold_medals']

# определить признаки
X = df[['age', 'height', 'weight']]

# создать объект линейной регрессии
lr = LinearRegression()

# обучить модель на тренировочных данных
lr.fit(X, y_true)

# предсказать значения на тестовых данных
y_pred = lr.predict(X)

# рассчитать accuracy
accuracy = sum(y_true == y_pred.round()) / len(y_true) * 100
print(f"Accuracy: {accuracy:.2f}%")
