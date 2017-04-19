from sklearn.naive_bayes import GaussianNB
from classificators.svm import get_words_from

from classificators.common import *


def main(path_to_crime_articles, path_to_not_crime_articles, learn_count, classify_count, space=400):
    articles = get_files_from(path_to_crime_articles)

    articles_to_learn = articles[:learn_count]

    counter = get_words_from(articles_to_learn, path_to_crime_articles)

    i = 0
    feature_space = dict()
    for (word, count) in counter.most_common(space):
        feature_space[word] = i
        i += 1

    print(feature_space)

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

    classifier = GaussianNB()

    X = []
    X.extend(article_crimes_vectors)
    X.extend(article_not_crimes_vectors)

    Y = [0] * (len(article_crimes_vectors))
    Y.extend([1] * len(article_not_crimes_vectors))

    classifier.fit(X, Y)

    success_count = 0
    articles_to_classify = get_files_from(path_to_crime_articles)[learn_count:(learn_count + int(classify_count/2))]
    for article_file in articles_to_classify:
        article_file = open(join(path_to_crime_articles, article_file), 'r', encoding='utf8')

        text = article_file.read()
        article_vector = [0] * space

        for word in prepare_words_from(text):
            if word in feature_space:
                article_vector[feature_space[word]] += 1

        print(classifier.predict(article_vector))
        if classifier.predict(article_vector)[0] == 0:
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

        print(classifier.predict(article_vector))
        if classifier.predict(article_vector)[0] != 0:
            success_count += 1

    print(success_count / int(classify_count/2) * 100)

main(path_to_guardian_crimes, path_to_guardian_not_crimes, articles_learn_count, articles_classify_count)