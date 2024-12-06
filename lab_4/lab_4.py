from keras import layers, models, optimizers
from matplotlib import pyplot as plt
import numpy as np



def make_dataset(lo, hi, step):
    """
    Генерація датасету
    :param lo: інфінум
    :param hi: супремум
    :param step: крок
    :return: список кортеджів з 3 значень функції та список відповідних наступних значень
    """
    x = np.arange(lo, hi, step)
    y = 0.5 * np.cos(x / 2) + 1

    # створення кортеджів з 3 послідовних елементів за допомогую утіліти sliding_window_view
    yp = np.lib.stride_tricks.sliding_window_view(y[:len(y) - 1], 3)
    yn = y[3:]

    return yp, yn


# створення датасетів
x, y = make_dataset(0, 3, 2e-2)
xt, yt = make_dataset(3, 6, 2e-2)
print(x, y)


# декларація мережі
model = models.Sequential([
    layers.Input([3]),
    layers.Dense(16, activation=None),
    layers.Dense(16, activation=None),
    layers.Dense(1, activation=None)
])


# запуск навчання
model.compile(loss='MSE', optimizer=optimizers.Adam(1e-3))
model.fit(x=x, y=y, validation_data=[xt, yt], epochs=300, batch_size=32)

# отримання прогнозу по тестовій виборці
yp = model.predict(xt)

# візуалізація прогнозу
for i in range(50):
    j = i * 10
    plt.plot([0, 1, 2], xt[j], color='blue')
    plt.scatter([3], yt[j], color='blue')
    plt.scatter([3], yp[j], color='red')
    plt.ylim(0.5, 1.5)
    plt.show()

