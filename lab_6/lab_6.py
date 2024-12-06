import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

PRECISION = 1e-3
SIGMA = 0.5

# Определение входных переменных
x1 = ctrl.Antecedent(np.arange(0, 3.1, PRECISION), 'X1')
x2 = ctrl.Antecedent(np.arange(0, 3.1, PRECISION), 'X2')
x3 = ctrl.Antecedent(np.arange(0, 3.1, PRECISION), 'X3')

# Определение выходной переменной
y = ctrl.Consequent(np.arange(0, 3.1, PRECISION), 'Y')

# Определение функций принадлежности
for var in [x1, x2, x3, y]:
    var['0'] = fuzz.gaussmf(var.universe, 0, SIGMA)
    var['1'] = fuzz.gaussmf(var.universe, 3, SIGMA)

# Определение правил
rules = [
    ctrl.Rule(x1['0'] & x2['0'] & x3['0'], y['0']),
    ctrl.Rule(~(x1['0'] & x2['0'] & x3['0']), y['1']),  # во всех прочих случаях возвращаем 1 так у нас "или"
]

# Создание системы управления
system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)


def print_sample(X):
    """
    Печатает результат логического вывода для заданого вектора параметров X
    :param X: входной вектор
    :return: None
    """
    sim.input['X1'] = X[0]
    sim.input['X2'] = X[1]
    sim.input['X3'] = X[2]
    sim.compute()
    Y = sim.output['Y']

    print(f"Вход: ( {X[0]:>5.2f}, {X[1]:>5.2f}, {X[2]:>5.2f} )  Выход: {Y:>5.2f}")
    return


print_sample([2.5, 1, 0])
print_sample([1, 1, 0])
print_sample([0.5, 1, 0.5])
print_sample([3, 3, 3])
print_sample([0, 0.5, 0])
print_sample([1, 1, 1])
print_sample([0.1, 0.1, 0.1])