import numpy as np
import pathlib
import cv2


class HammingNetwork:
    def __init__(self, prototypes):
        """
        Инициализация сети Хемминга.
        :param prototypes: список эталонных изображений в виде биполярных векторов
        """
        self.prototypes = np.array(prototypes)
        self.num_neurons = len(prototypes)
        self.input_dim = len(prototypes[0])
        self.weights = 0.5 * (self.prototypes + 1)
        self.bias = 0.5 * self.input_dim

    def compute_similarity(self, input_vector):
        """
        Вычислить меры сходства между входным вектором и эталонными изображениями.
        :param input_vector: входное изображение в виде биполярного вектора
        :return: вектор сходств
        """
        input_vector = np.array(input_vector)
        return np.dot(self.weights, input_vector) + self.bias

    def maxnet(self, similarities, epsilon=0.1, max_iter=100):
        """
        Функционирование подсети Maxnet для определения победителя.
        :param similarities: вектор сходств
        :param epsilon: параметр подавления
        :param max_iter: максимальное количество итераций
        :return: индекс победителя
        """
        outputs = similarities.copy()
        for _ in range(max_iter):
            outputs = np.maximum(0, outputs - epsilon * (np.sum(outputs) - outputs))
            if np.sum(outputs > 0) == 1:
                break
        winner = np.argmax(outputs)
        return winner if outputs[winner] > 0 else None

    def classify(self, input_vector):
        """
        Классифицировать входное изображение.
        :param input_vector: входное изображение в виде биполярного вектора
        :return: индекс наиболее похожего эталонного изображения или None
        """
        similarities = self.compute_similarity(input_vector)
        return self.maxnet(similarities)


def norm(img):
    """
    Конвертация изображения в биполярный вектор {-1, 1}
    :param img: ч/б изображение размерности [25, 25]
    :return: биполярный вектор
    """
    return (img / 255 * 2 - 1).reshape([25])


def unnorm(img):
    """
    Конвертация изображения представленного биполярным вектором в ч/б формат opencv
    :param img: биполярный вектор {-1, 1}
    :return: ч/б формат opencv
    """
    return ((img + 1) / 2 * 255).reshape([5, 5]).astype('uint8')


def load_data(pth):
    """
    Загрузка датасеты из указанной папки, первый симвло файла в папке отвечает за имя класса
    :param pth: путь к папке с изображениями
    :return: список numpy массивов представляющий биполярные вектора изображений
    """
    data = []
    labels = []
    for img_pth in pathlib.Path(pth).iterdir():
        lab = img_pth.stem[0]
        img = cv2.imread(str(img_pth), cv2.IMREAD_GRAYSCALE)

        labels.append(lab)
        data.append(norm(img))

    return data, labels



# загрузка тренировочного и тестового датасетов
proto, proto_lab = load_data('train')
test, test_lab = load_data('test')

# инициализация и обучение сети
network = HammingNetwork(proto)

# тестирование и визуализация результата
for i in range(len(test)):
    proto_i = network.classify(test[i])
    lab_true = test_lab[i]
    lab_pred = proto_lab[proto_i]
    print(f'true: {lab_true}  pred: {lab_pred}')
    img_true = unnorm(test[i])
    img_pred = unnorm(proto[proto_i])
    img = cv2.hconcat([img_true, img_pred])
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 255, 0] if lab_true == lab_pred else [0, 0, 255])
    cv2.imshow('img', img)
    cv2.waitKey()