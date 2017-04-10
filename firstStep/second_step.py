# for reading all files
from os import listdir
from os.path import isfile, join
from collections import Counter

from common import *


def get_words_from(articles, path_to_articles):
    counter = Counter()
    article_num = 0

    for article_file in articles:
        article_num += 1

        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        counter += Counter(prepare_words_from(text))

    return counter


def main(path_to_articles, learn_count=100, classify_count=1000, space=400):
    articles = [file for file in listdir(path_to_articles) if isfile(join(path_to_articles, file))]

    articles_to_learn = articles[:learn_count]
    articles_to_classify = articles[learn_count:classify_count]

    # get words and their freq
    counter = get_words_from(articles_to_learn, path_to_articles)

    # make features
    i = 0
    feature_space = dict()
    for (word, count) in counter.most_common(space):
        feature_space[word] = i
        i += 1

    print(feature_space)

    # transform articles to vectors in our feature space
    article_vectors = []
    for article_file in articles_to_learn:
        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')
        text = article_file.read()

        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        article_vectors.append(article_vector)

    # find center mass of learning articles
    center = center_of_mass(article_vectors)

    # metrics
    success_count = 0
    max_dist = -9999999999999999
    for article_vector in article_vectors:
        cur_dist = dist_between(article_vector, center)
        if cur_dist > max_dist:
            max_dist = cur_dist

    for article_file in articles_to_classify:
        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        if dist_between(article_vector, center) < max_dist:
            success_count += 1

main(path_to_guardian, articles_learn_count, articles_to_classify)
