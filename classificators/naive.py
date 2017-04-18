from collections import Counter
import numpy

from classificators.common import *


def main(path_to_crime_articles, path_to_not_crime_articles, learn_count, classify_count):
    articles = get_files_from(path_to_crime_articles)

    articles_to_learn = articles[:learn_count]

    counter = Counter()
    for article_file in articles_to_learn:
        article_file = open(join(path_to_crime_articles, article_file), 'r', encoding='utf8')

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
    for article_file in articles_to_learn:
        article_file = open(join(path_to_crime_articles, article_file), 'r', encoding='utf8')
        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        article_vectors.append(article_vector)

    center = center_of_mass(article_vectors)
    success_count = 0

    # orb
    # max_dist = -9999999999999999
    # for article_vector in article_vectors:
    #     cur_dist = dist_between(article_vector, center)
    #     if cur_dist > max_dist:
    #         max_dist = cur_dist

    # like ellipse (should be better)
    max_remoteness = [0] * space
    for article_vector in article_vectors:
        diff = [abs(i - j) for i, j in zip(article_vector, center)]
        print (diff)
        max_remoteness = list(map(lambda x, y: x if x > y else y, diff, max_remoteness))

    print(max_remoteness)

    articles_to_classify = articles[learn_count:(learn_count + int(classify_count/2))]
    for article_file in articles_to_classify:
        article_file = open(join(path_to_crime_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        #if dist_between(article_vector, center) < max_dist:
        if is_less(article_vector, max_remoteness):
            success_count += 1

    print(success_count / int(classify_count/2) * 100)
    success_count = 0

    articles_to_classify = get_files_from(path_to_not_crime_articles)[:int(classify_count/2)]
    for article_file in articles_to_classify:
        article_file = open(join(path_to_not_crime_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        #if dist_between(article_vector, center) >= max_dist:
        if not is_less(article_vector, max_remoteness):
            success_count += 1

    print(success_count / int(classify_count/2) * 100)

main(path_to_guardian_crimes, path_to_guardian_not_crimes, articles_learn_count, articles_classify_count)

# Получился интересный результат (обучение - 1000, классификация - 10000):
#   если шар, то отлично определяются криминальные статьи (99%), но этот алгоритм считает и некриминальные статьи криминальными (около 0.32%)
#   если элдипс, то хорошо определяются криминальные статьи (79%), некриминальные - 20%
# переделать на процент слов, а не кол-во
