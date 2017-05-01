from sklearn.naive_bayes import GaussianNB

from classificators.common import *


def bayes(crime_documents, not_crime_documents, learn_count, classify_count, space=400):
    print("bayes started")
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
    idf_vector = form_idf_vector(feature_space, crime_documents_to_learn + not_crime_documents_to_learn)

    crime_vectors = tf_idf_all_with_idf_vector(feature_space, crime_documents_to_learn, idf_vector)
    not_crime_vectors = tf_idf_all_with_idf_vector(feature_space, not_crime_documents_to_learn, idf_vector)
    #crime_vectors = tf_normalize_all(feature_space, crime_documents_to_learn)
    #not_crime_vectors = tf_normalize_all(feature_space, not_crime_documents_to_learn)

    print("documents transformed successfully!")

    classifier = GaussianNB()

    X = []
    X.extend(crime_vectors)
    X.extend(not_crime_vectors)

    Y = [0] * (len(crime_vectors))
    Y.extend([1] * len(not_crime_vectors))

    print("bayes learning...")
    classifier.fit(X, Y)
    print("bayes ready to classify!")

    success_count = 0
    documents_to_classify = crime_documents[learn_count:(learn_count + classify_count)]

    print("classifying crime documents...")
    for document in documents_to_classify:
        #article_vector = tf_normalize(feature_space, document)
        article_vector = tf_idf(feature_space, document, idf_vector)

        if classifier.predict(np.array(article_vector).reshape(1, -1))[0] == 0:
            success_count += 1

    print(success_count / classify_count * 100)

    print("classifying not crime documents...")
    success_count = 0
    documents_to_classify = not_crime_documents[learn_count:(learn_count + classify_count)]
    for document in documents_to_classify:
        #article_vector = tf_normalize(feature_space, document)
        article_vector = tf_idf(feature_space, document, idf_vector)

        if classifier.predict(np.array(article_vector).reshape(1, -1))[0] != 0:
            success_count += 1

    print(success_count / classify_count * 100)

bayes(crime_articles, not_crime_articles, articles_learn_count, articles_classify_count)

# 99.3
# 76.0