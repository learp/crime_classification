from classificators.common import *


def naive(crime_documents, not_crime_documents, learn_count, classify_count, space=400):
    print("naive started")
    crime_documents_to_learn = crime_documents[:learn_count]
    not_crime_documents_to_learn = not_crime_documents[:learn_count]

    # get words and their freq
    counter = get_words_from(crime_documents_to_learn)

    print("make feature space...")
    i = 0
    feature_space = dict()
    for (word, count) in counter.most_common(space):
        feature_space[word] = i
        i += 1

    print(feature_space)

    print("transforming documents into feature space...")
    crime_vectors = tf_all(feature_space, crime_documents_to_learn)

    print("documents transformed successfully!")

    print("naive learning...")
    center = center_of_mass(crime_vectors)

    # orb
    max_dist = -9999999999999999
    distances = []
    for crime_vector in crime_vectors:
        cur_dist = dist_between(crime_vector, center)
        distances.append(cur_dist)
        if cur_dist > max_dist:
            max_dist = cur_dist

    print(max_dist)
    max_dist = np.percentile(distances, 90)
    print(max_dist)

    print("naive ready!")
    # like ellipse (should be better)
    # max_remoteness = [0] * space
    # for article_vector in article_vectors:
    #     diff = [abs(i - j) for i, j in zip(article_vector, center)]
    #     print(diff)
    #     max_remoteness = list(map(lambda x, y: x if x > y else y, diff, max_remoteness))

    #print(max_remoteness)

    success_count = 0
    documents_to_classify = crime_documents[learn_count:(learn_count + classify_count)]

    print("classifying crime documents...")
    for document in documents_to_classify:
        article_vector = tf(feature_space, document)

        if dist_between(article_vector, center) < max_dist:
        # if is_less(article_vector, max_remoteness):
            success_count += 1

    print(success_count / classify_count * 100)

    success_count = 0
    documents_to_classify = not_crime_documents[learn_count:(learn_count + classify_count)]
    print("classifying not crime documents...")
    for document in documents_to_classify:
        article_vector = tf(feature_space, document)

        if dist_between(article_vector, center) >= max_dist:
        #if not is_less(article_vector, max_remoteness):
            success_count += 1

    print(success_count / classify_count * 100)

naive(crime_articles, not_crime_articles, articles_learn_count, articles_classify_count)

# Получился интересный результат (обучение - 1000, классификация - 10000):
#   если шар, то отлично определяются криминальные статьи (99%), но этот алгоритм считает и некриминальные статьи криминальными (около 0.32%)
#   если элдипс, то хорошо определяются криминальные статьи (79%), некриминальные - 20%
# переделать на процент слов, а не кол-во
