import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

np.set_printoptions(precision=2, suppress=True)

# Загрузка данных
dataset1 = pd.read_csv('Salary_dataset.csv')  # Замените на свой путь к файлу
print("Первые строки набора данных:")
print(dataset1.head())

# Оценка статистических характеристик
means = np.mean(dataset1, axis=0)
min_values = np.min(dataset1, axis=0)
max_values = np.max(dataset1, axis=0)
missing_values = np.sum(np.isnan(dataset1), axis=0)

# Вывод статистических характеристик
print("Средние значения для каждого признака:", means)
print("Минимальные значения для каждого признака:", min_values)
print("Максимальные значения для каждого признака:", max_values)
print("Количество пропущенных значений для каждого признака:", missing_values)

# Загрузка второго набора данных
dataset = pd.read_csv('Salary_dataset.csv')
print(dataset.head())

# Выделение признаков и целевой переменной
X = dataset.iloc[:, 1:-1].values  # Все столбцы, кроме первого и последнего
y = dataset.iloc[:, -1].values    # Последний столбец - целевая переменная
print("Матрица признаков"); print(X)
print("Зависимая переменная"); print(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Создание универсального пайплайна
pipeline = make_pipeline(
    StandardScaler(),       # Стандартизация признаков
    LinearRegression()      # Модель линейной регрессии
)

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = pipeline.predict(X_test)
print("Предсказанные значения:", y_pred)

# Оценка качества модели
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")
print(f"Коэффициент детерминации(R^2): {r2}")

# Визуализация результатов (обучающая выборка)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, pipeline.predict(X_train), color='blue')
plt.title('Observation vs Time (Training set)')
plt.xlabel('Time (years)')
plt.ylabel('Observation')
plt.show()

# Визуализация результатов (тестовая выборка)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, pipeline.predict(X_train), color='blue')
plt.title('Observation vs Time (Test set)')
plt.xlabel('Time (years)')
plt.ylabel('Observation')
plt.show()
