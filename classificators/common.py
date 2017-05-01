from scipy.spatial import distance

import numpy as np

from collections import Counter

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from os import listdir
from os.path import isfile, join

import string

print("preparing to classify...")
path_to_reuters = 'reuters21578'
path_to_guardian_crimes = join('..', 'the_guardian_com', 'crimes')
path_to_guardian_not_crimes = join('..', 'the_guardian_com', 'not_crimes')

articles_learn_count = 100
articles_classify_count = 1000


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
    return distance.hamming(point_1, point_2)#distance.euclidean(point_1, point_2)


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


def tf(feature_space, document):
    words = prepare_words_from(document)
    vector = [0] * len(feature_space)

    for word in words:
        if word in feature_space:
            vector[feature_space[word]] += 1

    return vector


def tf_all(feature_space, documents):
    vectors = []
    for document in documents:
        vectors.append(tf(feature_space, document))

    return vectors


def tf_normalize(feature_space, document):
    words = prepare_words_from(document)
    vector = [0] * len(feature_space)

    for word in words:
        if word in feature_space:
            vector[feature_space[word]] += 1

    return [x / len(words) for x in vector]


def tf_normalize_all(feature_space, documents):
    vectors = []
    for document in documents:
        vectors.append(tf_normalize(feature_space, document))

    return vectors


def idf(term, documents):
    count_where_term_found = 0
    for document in documents:
        if term in prepare_words_from(document):
            count_where_term_found += 1

    return np.log2(len(documents) / count_where_term_found)


crime_articles = [open(
    join(path_to_guardian_crimes, file_name),
    'r',
    encoding='utf8').read()
    for file_name in get_files_from(path_to_guardian_crimes)[:(articles_learn_count + articles_classify_count)]]

not_crime_articles = [open(
    join(path_to_guardian_not_crimes, file_name),
    'r',
    encoding='utf8').read()
    for file_name in get_files_from(path_to_guardian_not_crimes)[:(articles_learn_count + articles_classify_count)]]


def form_idf_vector(feature_space, documents):
    idf_vector = [0] * len(feature_space)
    for word, index in feature_space.items():
        idf_vector[index] = idf(word, documents)

    return idf_vector


def tf_idf_all(feature_space, documents):
    idf_vector = form_idf_vector(feature_space, documents)

    res = []
    for document in documents:
        tf_vector = tf_normalize(feature_space, document)

        res.append(map(lambda x, y: x * y, idf_vector, tf_vector))

    return res


def tf_idf_all_with_idf_vector(feature_space, documents, idf_vector):
    res = []
    for document in documents:
        tf_vector = tf_normalize(feature_space, document)

        res.append(map(lambda x, y: x * y, idf_vector, tf_vector))

    return res


def tf_idf(feature_space, document, idf_vector):
    tf_vector = tf_normalize(feature_space, document)

    return map(lambda x, y: x * y, idf_vector, tf_vector)


def get_words_from(documents):
    counter = Counter()

    for document in documents:
        counter += Counter(prepare_words_from(document))

    return counter
