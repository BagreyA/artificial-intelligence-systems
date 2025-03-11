import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz  # Для визуализации дерева

# Загрузка данных о вине
data_source = 'wine.data'
d = pd.read_table(data_source, delimiter=',',
                  header=None,
                  names=['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                         'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 
                         'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'])

# Просмотр первых строк
print(d.head())

# Подготовка данных
dX = d.iloc[:, 1:]  # Все признаки
dy = d['class']  # Классы

# Подмножества для hold-out
X_train, X_holdout, y_train, y_holdout = \
train_test_split(dX, dy, test_size=0.3, random_state=12)

# 2.1 Построение логического классификатора с заданием max_depth и max_features
tree = DecisionTreeClassifier(max_depth=5,
                              random_state=21,
                              max_features=2)
tree.fit(X_train, y_train)

# Получение оценки hold-out
tree_pred = tree.predict(X_holdout)
accur = accuracy_score(y_holdout, tree_pred)
print("Accuracy on holdout set: ", accur)

# 2.2 Вычисление оценки cross-validation (MSE) для различных значений max_depth
d_list = list(range(1, 20))
CV_scores = []
for d in d_list:
    tree = DecisionTreeClassifier(max_depth=d, random_state=21, max_features=2)
    scores = cross_val_score(tree, dX, dy, cv=10, scoring='accuracy')
    CV_scores.append(scores.mean())

# Вычисляем ошибку классификации (MSE)
MSE = [1 - x for x in CV_scores]

# Строим график зависимости ошибки от max_depth
plt.plot(d_list, MSE)
plt.xlabel('Макс. глубина дерева (max_depth)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Зависимость ошибки от max_depth')
plt.show()

# 2.3 Вычисление оценки cross-validation (MSE) для различных значений max_features
f_list = list(range(1, 5))
CV_scores_features = []
for f in f_list:
    tree = DecisionTreeClassifier(max_depth=5, random_state=21, max_features=f)
    scores = cross_val_score(tree, dX, dy, cv=10, scoring='accuracy')
    CV_scores_features.append(scores.mean())

# Вычисляем ошибку классификации (MSE) для max_features
MSE_features = [1 - x for x in CV_scores_features]

# Строим график зависимости ошибки от max_features
plt.plot(f_list, MSE_features)
plt.xlabel('Макс. количество признаков (max_features)')
plt.ylabel('Ошибка классификации (MSE)')
plt.title('Зависимость ошибки от max_features')
plt.show()

# 2.4 Оптимальные значения max_depth и max_features
d_min = min(MSE)
all_d_min = [d_list[i] for i in range(len(MSE)) if MSE[i] <= d_min]
f_min = min(MSE_features)
all_f_min = [f_list[i] for i in range(len(MSE_features)) if MSE_features[i] <= f_min]

print('Оптимальные значения max_depth: ', all_d_min)
print('Оптимальные значения max_features: ', all_f_min)

# 2.5 Поиск оптимальных параметров с помощью GridSearchCV
dtc = DecisionTreeClassifier(random_state=21)
tree_params = { 'max_depth': range(1, 20), 'max_features': range(1, 6) }
tree_grid = GridSearchCV(dtc, tree_params, cv=10, verbose=True, n_jobs=-1)
tree_grid.fit(dX, dy)

print('\n')
print('Лучшее сочетание параметров: ', tree_grid.best_params_)
print('Лучшие баллы cross validation: ', tree_grid.best_score_)

# Генерация дерева с лучшими параметрами
export_graphviz(tree_grid.best_estimator_,
                feature_names=dX.columns,
                class_names=[str(i) for i in tree_grid.best_estimator_.classes_],  
                out_file='wine_tree.dot',
                filled=True, rounded=True)

# Визуализация дерева в формате .png
with open("wine_tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph).render("wine_tree.png", format="png")

# 2.6 Построим области решения для оптимального дерева
dtc = DecisionTreeClassifier(max_depth=3, random_state=21, max_features=2)
dtc.fit(dX, dy)

plot_markers = ['r*', 'g^', 'bo', 'cs']
answers = dy.unique()

# Пример предсказания для новых данных
res = dtc.predict([[13.2, 2.3, 2.4, 2.5, 21.0, 107.0, 3.5, 0.7, 0.2, 1.8, 3.0, 0.1, 970.0]])
print('Результат классификации для нового примера: ', res)

# Кодируем метки классов в числовые значения
label_encoder = LabelEncoder()
dy_encoded = label_encoder.fit_transform(dy)

# Создаем подграфики для каждой пары признаков
f, places = plt.subplots(4, 4, figsize=(16,16))

fmin = dX.min() - 0.5
fmax = dX.max() + 0.5
plot_step = 0.02

# Генерируем решающие границы для каждой пары признаков
for i in range(0, 4):
    for j in range(0, 4):
        if i != j:
            xx, yy = np.meshgrid(np.arange(fmin[i], fmax[i], plot_step),
                                 np.arange(fmin[j], fmax[j], plot_step))
            model = DecisionTreeClassifier(max_depth=3, random_state=21, max_features=2)
            model.fit(dX.iloc[:, [i, j]].values, dy)
            p = model.predict(np.c_[xx.ravel(), yy.ravel()])
            p = p.reshape(xx.shape)

            # Преобразуем классы в числа с помощью LabelEncoder
            p = label_encoder.transform(p.ravel())  # Преобразуем одномерный массив предсказаний
            places[i, j].contourf(xx, yy, p.reshape(xx.shape), cmap='Pastel1')

        # Обход всех классов для графиков
        for id_answer in range(len(dy.unique())):
            idx = np.where(dy == dy.unique()[id_answer])  # Получаем индексы
            if i == j:
                places[i, j].hist(dX.iloc[idx[0], i], color=plot_markers[id_answer][0], histtype='step')
            else:
                places[i, j].plot(dX.iloc[idx[0], i], dX.iloc[idx[0], j], plot_markers[id_answer],
                                   label=dy.unique()[id_answer], markersize=6)

        if j == 0:
            places[i, j].set_ylabel(dX.columns[j])
        if i == 3:
            places[i, j].set_xlabel(dX.columns[i])

plt.show()
