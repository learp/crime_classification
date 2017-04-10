from scipy.spatial import distance

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

import string

path_to_reuters = 'reuters21578'
path_to_guardian = 'data_set'
articles_learn_count = 100
articles_to_classify = 10000


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def prepare_words_from(text):
    stop_words = stopwords.words('english')
    words = [word for word in word_tokenize(text.lower()) if
             word not in stop_words and
             word not in string.punctuation and
             not is_number(word) and
             len(word) > 2]

    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]


def center_of_mass(points):
    if len(points) == 0:
        return []

    center = [0] * len(points[0])

    for point in points:
        i = 0
        while i < len(point):
            center[i] += point[i]
            i += 1

    i = 0
    while i < len(center):
        center[i] /= len(points)
        i += 1

    return center


def dist_between(point_1, point_2):
    return distance.euclidean(point_1, point_2)