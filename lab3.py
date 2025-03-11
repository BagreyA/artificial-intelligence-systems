import pandas as pd 
import numpy as np
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
import matplotlib.pyplot as plt

# Устанавливаем параметры отображения чисел
np.set_printoptions(precision=2, suppress=True)

# Загрузка данных
data1 = np.genfromtxt('wine.data', delimiter=",", dtype=float)

# Оценка статистических характеристик
means = np.mean(data1, axis=0)
min_values = np.min(data1, axis=0)
max_values = np.max(data1, axis=0)
missing_values = np.sum(np.isnan(data1), axis=0)

# Вывод статистических характеристик
print("Средние значения для каждого признака:", means)
print("Минимальные значения для каждого признака:", min_values)
print("Максимальные значения для каждого признака:", max_values)
print("Количество пропущенных значений для каждого признака:", missing_values)

# Загрузка данных о вине
data_source = 'wine.data'
#d = pd.read_table(data_source, delimiter = ',')
#print(d.head())

#d = pd.read_table(data_source, delimiter=',', header=None)
#print(d.head())

d = pd.read_table(data_source, delimiter=',', header=None,
                  names=['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                         'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
                         'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'])

# Просмотр первых строк
print(d.head())

# Визуализация данных
sb.pairplot(d)
plt.show()

# Визуализация данных с цветовой кодировкой по классу
sb.pairplot(d, hue='class', diag_kind='hist', palette={1: 'red', 2: 'green', 3: 'blue'})
plt.show()

# Разделение на признаки (X) и целевую переменную (y)
X_train = d[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
             'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
             'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']]

y_train = d['class']

# Установка начального значения для количества ближайших соседей K
K = 3  # Количество соседей для классификатора

# Создание и настройка классификатора
knn = KNeighborsClassifier(n_neighbors=K)

# Обучение модели классификатора
knn.fit(X_train, y_train)

# Использование классификатора для предсказания класса для нового объекта
X_test = pd.DataFrame([[1.2, 1.0, 2.8, 1.2, 1.1, 3.0, 2.0, 1.5, 0.3, 1.2, 0.6, 2.8, 1000]], 
                      columns=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
                               'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
                               'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'])

target = knn.predict(X_test)
print(f'Предсказанный класс для нового объекта: {target[0]}')

# Разделение данных на обучающую и тестовую выборки
X_train, X_holdout, y_train, y_holdout = train_test_split(
    d[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']], 
    d['class'], test_size=0.3, random_state=17)

# Обучение модели на обучающей выборке
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Оценка точности на тестовой выборке
knn_pred = knn.predict(X_holdout)
accur = accuracy_score(y_holdout, knn_pred)
print(f'Точность классификации (hold-out): {accur}')

# Подзадача 2.2: Оценка точности для различных значений K с использованием кросс-валидации
k_list = list(range(1, 50))
cv_scores = []

for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, d.iloc[:, 1:], d['class'], cv=10, scoring='accuracy')  # Признаки с 1 по 13
    cv_scores.append(scores.mean())

# Вычисляем ошибку классификации (misclassification error)
MSE = [1 - x for x in cv_scores]

# Строим график ошибки классификации в зависимости от K
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Зависимость ошибки классификации от K')
plt.show()

# Ищем минимальное значение ошибки и оптимальное K
k_min = min(MSE)

# Ищем все минимальные K, если их несколько
all_k_min = []
for i in range(len(MSE)): 
   if MSE[i] <= k_min:
      all_k_min.append(k_list[i])

# Печать оптимальных значений K
print(f'Оптимальные значения K: {all_k_min}')

# Подзадача 2.3: Оценка hold-out для различных долей обучающей и тестовой выборок
holdout_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
holdout_accuracies = []

for size in holdout_sizes:
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        d[['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
            'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
            'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']], 
        d['class'], test_size=size, random_state=17)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    knn_pred = knn.predict(X_holdout)
    accur = accuracy_score(y_holdout, knn_pred)
    holdout_accuracies.append(accur)

# Строим график зависимости точности от размера тестовой выборки
plt.plot(holdout_sizes, holdout_accuracies)
plt.xlabel('Размер тестовой выборки')
plt.ylabel('Точность классификации')
plt.title('Зависимость точности классификации от размера тестовой выборки')
plt.show()
