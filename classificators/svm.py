from collections import Counter
from sklearn import svm

from classificators.common import *


def get_words_from(articles, path_to_articles):
    counter = Counter()
    article_num = 0

    for article_file in articles:
        article_num += 1

        article_file = open(join(path_to_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        counter += Counter(prepare_words_from(text))

    return counter


def main(path_to_crime_articles, path_to_not_crime_articles, learn_count=100, classify_count=1000, space=400):
    articles = get_files_from(path_to_crime_articles)

    articles_to_learn = articles[:learn_count]
    articles_to_classify = articles[learn_count:(learn_count + classify_count)]

    # get words and their freq
    counter = get_words_from(articles_to_learn, path_to_crime_articles)

    # make features
    i = 0
    feature_space = dict()
    for (word, count) in counter.most_common(space):
        feature_space[word] = i
        i += 1

    print(feature_space)

    # transform articles to vectors in our feature space
    article_crimes_vectors = []
    for article_crime_file in articles_to_learn:
        article_file = open(join(path_to_crime_articles, article_crime_file), 'r', encoding='utf8')
        text = article_file.read()

        article_crime_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_crime_vector[feature_space[word]] += 1

        article_crimes_vectors.append(article_crime_vector)

    articles = get_files_from(path_to_not_crime_articles)
    articles_to_learn = articles[:learn_count]

    article_not_crimes_vectors = []
    for article_not_crime_file in articles_to_learn:
        article_file = open(join(path_to_not_crime_articles, article_not_crime_file), 'r', encoding='utf8')
        text = article_file.read()

        article_not_crime_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_not_crime_vector[feature_space[word]] += 1

        article_not_crimes_vectors.append(article_not_crime_vector)

    classifier = svm.SVC()
    X = []
    X.extend(article_crimes_vectors)
    X.extend(article_not_crimes_vectors)

    Y = [0] * (len(article_crimes_vectors))
    Y.extend([1] * len(article_not_crimes_vectors))
    print(X)
    print(Y)
    classifier.fit(X, Y)

    success_count = 0
    for article_file in articles_to_classify:
        article_file = open(join(path_to_guardian_crimes, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        print(classifier.predict(article_vector))
        if classifier.predict(article_vector)[0] == 0:
            success_count += 1

    print(success_count / classify_count * 100)

main(path_to_guardian_crimes, path_to_guardian_not_crimes, articles_learn_count, articles_classify_count)
