import numpy as np
import matplotlib.pyplot as plt

# 1. Проверка центральной предельной теоремы
# Создаём временную ось: от 0 до 3, 1000 точек
t = np.linspace(0, 3, 1000)

# 1.a) Генерация равномерно распределённой СВ на интервале (0,1)
xn = np.random.uniform(0, 1, len(t))

# Гистограмма распределения xn
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(xn, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Гистограмма равномерного распределения xn')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')

# 1.b) Формирование СВ в виде суммы равномерно распределённых величин
# Выбираем число слагаемых (например, 12)
n_summands = 12
# Для каждой из 1000 точек суммируем n_summands независимых равномерных СВ
Yn = np.sum(np.random.uniform(0, 1, (n_summands, len(t))), axis=0)

# Гистограмма распределения суммы Yn
plt.subplot(1, 2, 2)
plt.hist(Yn, bins=30, density=True, alpha=0.7, color='green')
plt.title('Гистограмма суммы (Yn) из 12 равномерных величин')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')

plt.tight_layout()
plt.show()

# 2. Вычисление АКФ по множеству реализаций

# Параметры нормального распределения
m = 0      # матожидание
s1 = 1     # стандартное отклонение
t2 = np.linspace(0, 3, 100)  # временная ось для каждой реализации (100 точек)

# Число реализаций
N_realizations = 500

# Фильтр (ядро свёртки) для формирования коррелированного процесса
h = np.array([1, 0.7, 0.3, 0.1, 0.05])

# Генерируем множество реализаций
realizations = []
for i in range(N_realizations):
    # Генерируем нормальное распределение
    x = np.random.normal(m, s1, len(t2))
    # Применяем свёртку с фильтром (используем mode='same' для сохранения длины)
    x_corr = np.convolve(x, h, mode='same')
    realizations.append(x_corr)
realizations = np.array(realizations)  # размер: (N_realizations, len(t2))

# Формирование сечения: выбираем произвольный индекс, например, 50
index_section = 50
section_values = realizations[:, index_section]

# Временная диаграмма (отрезок) одной реализации (например, первой)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t2, realizations[0], marker='o', linestyle='-', color='purple')
plt.title('Временная диаграмма одной реализации')
plt.xlabel('Время')
plt.ylabel('Значения СВ')

# Гистограмма распределения значений сечения (индекс 50)
plt.subplot(1, 2, 2)
plt.hist(section_values, bins=30, density=True, alpha=0.7, color='orange')
plt.title('Гистограмма сечения (индекс 50)')
plt.xlabel('Значения')
plt.ylabel('Плотность вероятности')
plt.tight_layout()
plt.show()

# Вычисление АКФ по множеству реализаций
# Для выбранного базового индекса (например, index_ref = 50) и разных лагов tau
taus_spec = [0, 3, 5, 7]
acf_spec = []
index_ref = 50  # базовый индекс для вычисления АКФ

for tau in taus_spec:
    if index_ref + tau < realizations.shape[1]:
        # Для каждой реализации вычисляем произведение значений в индексах index_ref и index_ref+tau
        products = realizations[:, index_ref] * realizations[:, index_ref + tau]
        acf_tau = np.mean(products)
        acf_spec.append(acf_tau)
    else:
        acf_spec.append(np.nan)

print("АКФ для tau =", taus_spec, ":", acf_spec)

# Для построения графика АКФ по tau вычислим значения для tau от 0 до max_tau
max_tau = 20
taus_full = np.arange(0, max_tau + 1)
acf_full = []
for tau in taus_full:
    if index_ref + tau < realizations.shape[1]:
        products = realizations[:, index_ref] * realizations[:, index_ref + tau]
        acf_tau = np.mean(products)
        acf_full.append(acf_tau)
    else:
        acf_full.append(np.nan)

plt.figure(figsize=(8, 5))
plt.stem(taus_full, acf_full)
plt.title(f'АКФ по множеству реализаций (базовый индекс = {index_ref})')
plt.xlabel('tau (смещение)')
plt.ylabel('АКФ')
plt.grid(True)
plt.show()

# Определение интервала корреляции: tau, при котором АКФ падает до 1/e от значения при tau=0
acf0 = acf_full[0]
tau_corr = None

for tau, acf_val in zip(taus_full, acf_full):
    if acf_val <= acf0 / np.e:
        tau_corr = tau
        break

if tau_corr is not None:
    print("Интервал корреляции (tau, когда АКФ <= 1/e от АCF(0)) =", tau_corr)
else:
    print("Интервал корреляции не найден в диапазоне tau.")

# 3. Вычисление АКФ по одной реализации

# Генерируем одну реализацию скоррелированного нормального процесса
x_single = np.convolve(np.random.normal(m, s1, len(t2)), h, mode='same')

# Вычисляем АКФ методом временного усреднения по формуле:
# АКФ(tau) = (1/(N-tau)) * sum_{i=0}^{N-tau-1} x[i] * x[i+tau]
N = len(x_single)
max_lag = 20  # максимальное значение tau
acf_time = np.zeros(max_lag + 1)
for lag in range(max_lag + 1):
    products = x_single[:N - lag] * x_single[lag:]
    acf_time[lag] = np.mean(products)

plt.figure(figsize=(8, 5))
plt.stem(np.arange(max_lag + 1), acf_time)
plt.title('АКФ по одной реализации (усреднение по времени)')
plt.xlabel('tau (смещение)')
plt.ylabel('АКФ')
plt.grid(True)
plt.show()