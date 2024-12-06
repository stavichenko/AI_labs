import numpy as np


def similarity_russell_rao(a, b, g, h, n):
    return a / n


def similarity_jaccard(a, b, g, h, n):
    return a / (n - b)


def similarity_dice(a, b, g, h, n):
    return (2 * a) / (2 * a + g + h)


def similarity_sokal_sneath(a, b, g, h, n):
    return a / (a + 2 * (g + h))


def similarity_sokal_michener(a, b, g, h, n):
    return (a + b) / n


def similarity_kulczynski(a, b, g, h, n):
    return a / (g + h)


def similarity_yule(a, b, g, h, n):
    return (a * b - g * h) / (a * b + g * h)


def hamming_distance(x1, x2):
    return np.sum(x1 != x2)


def iou(a, b, g, h, n):
    return a / (g + h + a)


def calculate_abghn(x1, x2):
    a = np.sum(x1 & x2)
    b = np.sum(~x1 & ~x2)
    g = np.sum(~x1 & x2)
    h = np.sum(x1 & ~x2)
    return a, b, g, h, len(x1)


SIMILARITIES = {
    "Russell-Rao": similarity_russell_rao,
    "Jaccard": similarity_jaccard,
    "Dice": similarity_dice,
    "Sokal-Sneath": similarity_sokal_sneath,
    "Sokal-Michener": similarity_sokal_michener,
    "Kulczynski": similarity_kulczynski,
    "Yule": similarity_yule,
    "IOU": iou
}


def recognize_object(obj, features):
    """
    a: 1 1
    b: 0 0
    g: 0 1
    h: 1 0
    """
    scores = {key: [] for key in SIMILARITIES}

    for i in range(len(features)):
        a, b, g, h, n = calculate_abghn(obj, features[i])
        for key in SIMILARITIES:
            scores[key].append(SIMILARITIES[key](a, b, g, h, n))

    matches = {key: (np.argmax(scores[key]), max(scores[key])) for key in SIMILARITIES}

    return matches


def unpack_data_dict(data):
    return [key for key in data], np.array([data[key] for key in data]).astype(bool)


feature_names = ['eat_meat', 'eat_vegetables', 'move_fast', 'is_smart', 'form_society', 'can_fly', 'can_roar', 'climbing_trees']

etalon = {
    'Human':         [1, 1, 0, 1, 1, 0, 0, 1],
    'Tiger':         [1, 0, 1, 1, 0, 0, 1, 1],
    'Wolf':          [1, 0, 1, 1, 1, 0, 1, 0],
    'Cow':           [0, 1, 0, 0, 1, 0, 0, 0],
    'Butterfly':     [0, 1, 1, 0, 0, 1, 0, 0]
}

test = {
    'Keto-diet-man': [1, 0, 0, 1, 1, 0, 0, 1],
    'Vegan-man':     [0, 1, 0, 1, 1, 0, 0, 1],
    'Sirko':         [1, 1, 1, 1, 1, 0, 1, 0],
    'Kesha':         [1, 1, 1, 1, 0, 1, 0, 0]
}

etalon_names, etalon_features = unpack_data_dict(etalon)
test_names, test_features = unpack_data_dict(test)

for i in range(len(test)):
    print(f'{test_names[i]}')
    matches = recognize_object(test_features[i], etalon_features)
    for sim in matches:
        match_i = matches[sim][0]
        match_score = matches[sim][1]
        match_name = etalon_names[match_i]
        print(f'{sim}: {match_name} ({match_score:0.2f})')

    print()


# 'eat_meat', 'eat_vegetables', 'move_fast', 'is_smart', 'form_society', 'can_fly', 'can_roar', 'climbing_trees'
max_pair = np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1, 1, 1, 1])
min_pair = np.array([1, 1, 1, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 1, 1, 1, 0, 0])
max_p = calculate_abghn(*max_pair)
min_p = calculate_abghn(*min_pair)
print('max', similarity_russell_rao(*max_p))
print('min', similarity_russell_rao(*min_p))
print()


## Hamming
# Just sample
print('hamming 1st sample:', hamming_distance(etalon_features[0], test_features[0]))

# Serial number in group
print(
    'hamming 2nd sample (assume I have 7th number):',
    hamming_distance(np.array([1, 0, 1, 0, 1, 0, 1, 0]), np.array([0, 1, 0, 1, 0, 1, 0, 0]))
)

# same as other metric
pair = np.array([1, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])
p = calculate_abghn(*pair)
print(
    'hamming same for kulczynski',
    similarity_kulczynski(*p),
    hamming_distance(*pair),
)