import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Загрузка данных из файла
df = pd.read_excel(r'C:\Users\Sergey\PycharmProjects\MISIS_Python\MISIS_2022\season_2\data\DataSet.xlsx',
                   sheet_name="Испорченные факты")

df = pd.pivot_table(df, values=['Продажи, руб', 'Продажи, шт', 'Повторение заказа', 'Маржинальная прибыль',
                                'Повторение товара'], index=["Факты.Товар ID"],
                    aggfunc={'Продажи, шт': [np.median, np.sum],
                             'Продажи, руб': np.sum,
                             'Повторение заказа': np.sum,
                             'Маржинальная прибыль': np.sum

                             })

newname = df.columns.map('_'.join)
df.columns = newname
df = df.reset_index()
total_sale = df['Продажи, руб_sum'].sum()
df['Доля'] = df['Продажи, руб_sum'] / total_sale * 100
df['Доля'].sum()
df = df.sort_values(by=('Продажи, руб_sum'), ascending=False)
df = df.assign(sum_d=df['Доля'].cumsum())
df.loc[(df['sum_d'] <= 80), 'ABC'] = 'A'
df.loc[(df['sum_d'] > 80) & (df['sum_d'] <= 95), 'ABC'] = 'B'
df.loc[(df['sum_d'] > 95), 'ABC'] = 'C'
df['Стоимость, руб'] = df['Продажи, руб_sum'] / df['Продажи, шт_sum']
df['Продажи в следующем периоде'] = (df['Продажи, шт_sum'] + df['Продажи, шт_median']) * df['Стоимость, руб']
total_sale_next = df['Продажи в следующем периоде'].sum()
df['Доля_будущая'] = df['Продажи в следующем периоде'] / total_sale_next * 100
df = df.sort_values(by=('Продажи в следующем периоде'), ascending=False)
df = df.assign(sum_d_next=df['Доля_будущая'].cumsum())
df.loc[(df['sum_d_next'] <= 80), 'ABC_next'] = 'A'
df.loc[(df['sum_d_next'] > 80) & (df['sum_d_next'] <= 95), 'ABC_next'] = 'B'
df.loc[(df['sum_d_next'] > 95), 'ABC_next'] = 'C'
df.loc[(df['ABC'] != df['ABC_next']), 'Изменение класса'] = 1
df.loc[(df['ABC'] == df['ABC_next']), 'Изменение класса'] = 0
df.loc[(df['sum_d'] <= 80), 'class'] = '0'
df.loc[(df['sum_d'] > 80) & (df['sum_d'] <= 95), 'class'] = '1'
df.loc[(df['sum_d'] > 95), 'class'] = '2'
df.loc[(df['sum_d_next'] <= 80), 'class_next'] = '0'
df.loc[(df['sum_d_next'] > 80) & (df['sum_d_next'] <= 95), 'class_next'] = '1'
df.loc[(df['sum_d_next'] > 95), 'class_next'] = '2'
old_df = df

# Удаление ненужных столбцов
df.drop(['Факты.Товар ID'], axis=1, inplace=True)

# Заполнение пропущенных значений
df.fillna(method='ffill', inplace=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
# df[df > np.finfo(np.float64).max] = np.finfo(np.float64).max
# df[df < np.finfo(np.float64).min] = np.finfo(np.float64).min

# Отбор признаков
features = ['Маржинальная прибыль_sum', 'Повторение заказа_sum', 'Продажи, руб_sum', 'Продажи, шт_median',
            'Продажи, шт_sum', 'Доля', 'sum_d', 'Стоимость, руб', 'Продажи в следующем периоде', 'Доля_будущая',
            'sum_d_next']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df[features], df['ABC_next'], test_size=0.3, random_state=42)

# Обучение RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Оценка качества модели на тестовой выборке
y_pred = rfc.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Использование лучшей модели для предсказания будущего ABC класса на новых данных
new_data = old_df
new_data[features] = scaler.transform(new_data[features])
y_new_pred = rfc.predict(new_data[features])
print('New data predictions:', y_new_pred)
