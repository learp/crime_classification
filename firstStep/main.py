# for reading all files
from os import listdir
from os.path import isfile, join
from collections import Counter
from scipy.spatial import distance

# for parsing reuters
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

import string


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


def main(path_to_articles, learn_count, classify_count):
    articles = [file for file in listdir(path_to_articles) if isfile(join(path_to_articles, file))]

    # for file in reuters:
    #     file = open(join(path_to_reuters, file))
    #     soup = BeautifulSoup(file, 'html.parser')
    #     articles_body = soup.find_all('body')
    #
    #     set()
    #     for article_body in articles_body:
    #         prepare_words_from(article_body.text)

    counter = Counter()
    article_num = 0
    for article_file in articles:
        article_num += 1

        if article_num == learn_count:
            break

        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        counter += Counter(prepare_words_from(text))

    space = 400
    i = 0
    feature_space = dict()
    for (word, count) in counter.most_common(space):
        feature_space[word] = i
        i += 1

    print(feature_space)

    article_vectors = []
    article_num = 0
    for article_file in articles:
        article_num += 1

        if article_num == learn_count:
            break

        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')
        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        article_vectors.append(article_vector)

    center = center_of_mass(article_vectors)

    success_count = 0
    max_dist = -9999999999999999
    for article_vector in article_vectors:
        cur_dist = dist_between(article_vector, center)
        if cur_dist > max_dist:
            max_dist = cur_dist

    article_num = 0
    for article_file in articles:
        article_num += 1

        if article_num < learn_count:
            continue

        if article_num == learn_count + classify_count:
            break

        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        if dist_between(article_vector, center) < max_dist:
            success_count += 1

    print(success_count / classify_count * 100)

path_to_reuters = 'reuters21578'
path_to_guardian = 'data_set'
articles_learn_count = 100
articles_to_classify = 10000

main(path_to_guardian, articles_learn_count, articles_to_classify)
