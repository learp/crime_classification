import numpy as np
import matplotlib.pyplot as plt
import random
import classificators.common as cm
import classificators.bayes as bayes
import classificators.svm as svm
import classificators.k_nearest as k_near
import classificators.naive as nv

learn = 100
#svm.support_vector_machine(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
# bayes.bayes(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=3)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=5)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=7)
# nv.naive(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)

learn = 1000
#svm.support_vector_machine(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
# bayes.bayes(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=3)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=5)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=7)
# nv.naive(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)

learn = 5000
#svm.support_vector_machine(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
# bayes.bayes(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=3)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=5)
k_near.k_nearest(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000, k=7)
# nv.naive(cm.crime_articles, cm.not_crime_articles, learn, cm.articles_classify_count, space=3000)

# import some data to play with
# iris = datasets.load_iris()
#
# X = iris.data[:(1 * len(iris.target)/3), :2]
#
# y = iris.target[:(1 * len(iris.target)/3)]
#
# center = 5
#
# r = 1
#
# X = list(np.arange(4, 6.03, 0.1))
# Y = []
#
#
# for x in X:
#     Y.append(center + np.sqrt(r ** 2 - (x-center) ** 2))
# for x in X:
#     Y.append(center - np.sqrt(r ** 2 - (x-center) ** 2))
#
# X = X + X
#
# h = .02  # step size in the mesh
#
# # create a mesh to plot in
# x_min, x_max = min(X) - 1, max(X) + 1
# y_min, y_max = min(Y) - 1, max(Y) + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
#
# c = [0] * len(X)
#
#
# tmp = []
#
# i = 0
#
# while (i < 30):
#     tmp.append(random.random() + 4.5)
#     i += 1
#
# for x in tmp:
#     y = random.random() + 4.5
#     Y.append(y)
#
# X = X + tmp
#
# c = c + [1]*len(tmp)
#
# plt.scatter(X, Y, c=c, cmap=plt.cm.coolwarm)
#
# plt.show()


