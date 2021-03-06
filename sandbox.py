from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import samples_generator
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
import numpy as np
from utils.sax import SAX
from utils.SAXTransformer import SAXTransformer

#X, y = samples_generator.make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=0)


X = []
y = []
with open('data/synthetic_control_TRAIN.txt') as infile:
    for i in infile:
        instance = []
        for j in i.split():
            instance.append(float(j))
        X.append(np.array(instance[1:]))
        y.append(instance[0])
X = np.array(X)
y = np.array(y)


base_pipe = Pipeline([('saxizer', SAXTransformer(points_per_symbol=1)),
                     ('features', FeatureUnion([('countvect', CountVectorizer(min_df=1, analyzer='char', ngram_range=(1, 10))),
                                                ('tfidfvect', TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1, 2)))])),
                     ('svc', svm.LinearSVC())])

bop_pipe = Pipeline([('saxizer', SAXTransformer(points_per_symbol=1)),
                     ('features', FeatureUnion([('countvect', CountVectorizer(min_df=1, analyzer='char', ngram_range=(1, 10))),
                                                ('tfidfvect', TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1, 2)))])),
                     ('svc', svm.LinearSVC())])


for i in [bop_pipe, base_pipe]:
    score, permutation_scores, pvalue = permutation_test_score(
        i, X, y, scoring="accuracy", cv=StratifiedKFold(y, 2), n_permutations=5, n_jobs=4)
    print("Score %s (pvalue : %s)" % (score, pvalue))

# svm_pipe = Pipeline([('svc', clf)])
# score, permutation_scores, pvalue = permutation_test_score(
#     svm_pipe, X, y, scoring="accuracy", cv=StratifiedKFold(y, 2), n_permutations=100, n_jobs=4)
# print("Baseline Classification score %s (pvalue : %s)" % (score, pvalue))

# X2 = np.array([SAX(i).sax() for i in X])
# svm_pipe = Pipeline([('svc', clf)])
# score, permutation_scores, pvalue = permutation_test_score(
#     svm_pipe, X2, y, scoring="accuracy", cv=StratifiedKFold(y, 2), n_permutations=100, n_jobs=1)
# print("Baseline Classification score %s (pvalue : %s)" % (score, pvalue))
