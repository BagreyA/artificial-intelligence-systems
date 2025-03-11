import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

np.set_printoptions(precision=2, suppress=True)

dataset1 = pd.read_csv('diabetes.csv')  # Замените на свой путь к файлу
print("Первые строки набора данных:")
print(dataset1.head())

# Encode the 'Outcome' column (yes/no to 1/0)
labelencoder_y = LabelEncoder()
dataset1['Outcome'] = labelencoder_y.fit_transform(dataset1['Outcome'])

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

dataset = pd.read_csv('diabetes.csv')
print(dataset.head())

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values
print("Матрица признаков"); print(X)
print("Зависимая переменная"); print(y)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:8])
X[:, 1:8] = imputer.transform(X[:, 1:8])
print(X)

labelencoder_y = LabelEncoder()
print("Зависимая переменная до обработки")
print(y)
y = labelencoder_y.fit_transform(y)
print("Зависимая переменная после обработки")
print(y)

# 5. Кодирование категориального признака
ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'), [0])  # Кодируем 1-й столбец
], remainder='passthrough')

X = ct.fit_transform(X)
print("Перекодировка категориального признака:\n", X)

X_dirty = X.copy()
print("Сopy:\n",X_dirty)
      
transformers = [
    ('onehot', OneHotEncoder(), [0]),  # Кодируем 1-й столбец
    ('imp', StandardScaler(), [1, 2])  # Масштабируем 2-й и 3-й столбцы
]

ct = ColumnTransformer(transformers)

X_transformed = ct.fit_transform(X_dirty)
print(X_transformed.shape)  # Проверяем форму массива

# Генерируем правильные названия столбцов
num_features = X_transformed.shape[1]  # Количество столбцов после трансформации
columns = [f'Feature_{i}' for i in range(num_features)]  # Создаем нужное количество имен

# Создаем DataFrame с правильным количеством столбцов
X_data = pd.DataFrame(X_transformed, columns=columns)
print(X_data.head())  # Проверяем результат

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Making predictions
y_pred = knn.predict(X_test)
print(f'Точность классификации на тестовых данных: {accuracy_score(y_test, y_pred)}')

# Cross-validation for different K values
k_list = list(range(1, 50))
cv_scores = []

for K in k_list:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # Use 10-fold cross-validation
    cv_scores.append(scores.mean())

# Misclassification error
MSE = [1 - x for x in cv_scores]

# Plotting misclassification error vs K
plt.plot(k_list, MSE)
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Зависимость ошибки классификации от K')
plt.show()

# Finding the optimal K (the minimum MSE)
k_min = min(MSE)
all_k_min = [k_list[i] for i in range(len(MSE)) if MSE[i] == k_min]

# Print optimal K values
print(f'Оптимальные значения K: {all_k_min}')

# Hold-out validation for different test set sizes
holdout_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
holdout_accuracies = []

for size in holdout_sizes:
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        dataset1.iloc[:, :-1], dataset1.iloc[:, -1], test_size=size, random_state=17)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    knn_pred = knn.predict(X_holdout)
    accur = accuracy_score(y_holdout, knn_pred)
    holdout_accuracies.append(accur)

# Plotting accuracy vs holdout size
plt.plot(holdout_sizes, holdout_accuracies)
plt.xlabel('Размер тестовой выборки')
plt.ylabel('Точность классификации')
plt.title('Зависимость точности классификации от размера тестовой выборки')
plt.show()

# 1. Гистограмма для каждого признака
dataset1.hist(figsize=(10, 8), bins=20)
plt.suptitle('Гистограммы распределения признаков')
plt.show()

# 2. Столбчатая диаграмма для зависимой переменной (Outcome)
plt.figure(figsize=(6, 4))
plt.bar([0, 1], np.bincount(y), tick_label=['0 (Negative)', '1 (Positive)'])
plt.xlabel('Outcome')
plt.ylabel('Частота')
plt.title('Распределение классов (Outcome)')
plt.show()

# 3. Гистограмма для признаков после обработки (масштабирования)
X_df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
X_df.hist(figsize=(10, 8), bins=20)
plt.suptitle('Гистограммы после обработки признаков')
plt.show()

# 4. Столбчатая диаграмма для ошибки классификации по K (уже есть график с MSE)
# Но можно добавить еще одну диаграмму для лучшего понимания.
plt.bar(k_list, MSE, color='skyblue')
plt.xlabel('Количество соседей (K)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Столбчатая диаграмма зависимости ошибки классификации от K')
plt.show()

# 5. Столбчатая диаграмма для точности классификации на разных размерах тестовых выборок
plt.bar(holdout_sizes, holdout_accuracies, color='lightgreen')
plt.xlabel('Размер тестовой выборки')
plt.ylabel('Точность классификации')
plt.title('Столбчатая диаграмма зависимости точности от размера тестовой выборки')
plt.show()