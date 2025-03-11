import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data_path = 'zoo.data'

# Имена столбцов
names = ["animal name",
   "hair",
   "feathers",
   "eggs",
   "milk",
   "airborne",
   "aquatic",
   "predator",
   "toothed",
   "backbone",
   "breathes",
   "venomous",
   "fins",
   "legs",
   "tail",
   "domestic",
   "catsize",
   "type"]

# Чтение данных с указанием названий столбцов
data = pd.read_csv(data_path, names=names)

# Просмотр первых строк данных
print(data.head())

# Вывод информации о данных
data.info()

# Построение гистограммы для всех числовых столбцов
data.hist()
plt.show()

# Гистограмма для столбца "legs"
data['legs'].hist()
plt.show()

# Построение коробчатой диаграммы для столбца "legs"
sns.boxplot(data["legs"])
plt.show()

# Построение таблицы с типами животных и количеством ног
top_data = data[["type", "legs"]]
print(top_data)

# Группировка по типу животного и подсчет суммы
top_data = top_data.groupby("type").sum()
print(top_data)

# Выборка первых 10 типов животных с наибольшим количеством
top_data = top_data[:10].index.values
print(top_data)

# Построение коробчатой диаграммы для количества ног по типам животных
sns.boxplot(y="legs", x="type", data=data[data.type.isin(top_data)], palette="Set3")
plt.show()

# Построение countplot для хвоста
sns.countplot(x= data["tail"], palette="pastel")
plt.show()

# Построение countplot для 5 наиболее часто встречающихся типов животных
sns.countplot(x= "type", data=data[data["type"].isin(data["type"].value_counts().head(5).index)], palette="pastel")
plt.show()

# Построение pairplot для столбцов "type", "legs", "tail" с разбиением по хвостам
custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
sns.pairplot(data=data[["type", "legs", "tail"]], hue="tail", palette=custom_palette)
plt.show()

# Построение графика рассеяния для "legs" против "type"
plt.scatter(data["legs"], data["type"], color="lightblue", edgecolors="blue")
plt.xlabel("Количество ног")
plt.ylabel("Тип")
plt.show()

# Построение графика рассеяния с цветами в зависимости от хвоста
c = data["tail"].map({1: 'lightblue', 0: 'orange'})
edge_c = data["tail"].map({1: 'blue', 0: 'red'})
plt.scatter(data["legs"], data["type"], color=c, edgecolors=edge_c)
plt.xlabel("Количество ног")
plt.ylabel("Тип")
plt.show()

# Вывод статистики по данным
print(data.head())
print(data.columns)

# Построение корреляционной матрицы для числовых столбцов
data_corr = data.drop(columns=["animal name"])
corr_matrix = data_corr.corr()
print(corr_matrix)

# Построение тепловой карты корреляции
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.show()