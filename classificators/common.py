from scipy.spatial import distance

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from os import listdir
from os.path import isfile, join

import string

path_to_reuters = 'reuters21578'
path_to_guardian_crimes = join('..', 'the_guardian_com', 'crimes')
path_to_guardian_not_crimes = join('..', 'the_guardian_com', 'not_crimes')
articles_learn_count = 10000
articles_classify_count = 10000


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


def get_files_from(directory):
    return [file for file in listdir(directory) if isfile(join(directory, file))]


def is_less(vector1, vector2):
    if len(vector1) != len(vector2):
        return False

    i = 0
    while i < len(vector1):
        if vector1[i] >= vector2[i]:
            return False
        i += 1

    return True
