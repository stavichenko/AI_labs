from keras import layers, models, optimizers
from matplotlib import pyplot as plt
import numpy as np


class Normalizer:
    """
    Класс для нормалізації данних перед навчанням. Приймае в конструктор тренувальний датасет,
    та зберігае порогові значення, які викростовуються для нормалізації, та відновлення данних
    """
    def __init__(self, data):
        self.data = data
        self.min = data.min()
        self.range = data.max() - self.min

    def norm(self, data):
        return (data - self.min) / self.range

    def unnorm(self, data):
        return data * self.range + self.min


def make_dataset(lo, hi, n):
    """
    Створує равномірний ряд x та y в межах lo, hi з n відліками
    :param lo:
    :param hi:
    :param n:
    :return:
    """
    x = np.linspace(lo, hi, n)
    y = np.sin(x)
    return x, y


# генерація навчальних та тестових даних
x, y = make_dataset(0, 1, 11)
xt, yt = make_dataset(0.05, 1, 23)
print(x, y)


# Задання структури нейронної мережі
model = models.Sequential([
    layers.Input([1]),
    layers.Dense(16, activation='gelu'),
    layers.Dense(16, activation='gelu'),
    layers.Dense(16, activation='gelu'),
    layers.Dense(1, activation=None)
])

normalizer = Normalizer(x)


# запуск процесса навчання
model.compile(loss='MSE', optimizer=optimizers.Adam(1e-3))
model.fit(x=x, y=normalizer.norm(y), epochs=1000, batch_size=32)


# трансляція результату в оригінальний простір
yp = normalizer.unnorm(model.predict(xt))


# візуалізація результатів
plt.plot(xt, yt, color='green')
plt.plot(xt, yp, color='red')
plt.scatter(x, y, color='black')
plt.show()

