# for reading all files
from os import listdir
from os.path import isfile, join
from collections import Counter

from firstStep.common import *


def main(path_to_articles, learn_count, classify_count):
    articles = get_files_from(path_to_articles)

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

main(path_to_guardian_crimes, articles_learn_count, articles_classify_count)
