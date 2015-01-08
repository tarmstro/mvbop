from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import samples_generator
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
import numpy as np
from utils.sax import SAX

# vectorizer = CountVectorizer(min_df=1)
# corpus = ['This is the first document.',
#           'This is the second second document.',
#           'And the third one.',
#           'Is this the first document']
# X = vectorizer.fit_transform(corpus)
# #print X
# analyze = vectorizer.build_analyzer()
# # print analyze("This is a text document to analyze.")
# # print vectorizer.get_feature_names()
# # print X.toarray()
# # print vectorizer.transform(['Something completely new.']).toarray()

# n = 100
# t = 5
# corpus_a = [SAX(np.random.normal(0, 1, t)).stringify() for i in xrange(n)]
# corpus_b = [SAX(np.random.normal(0, 1.5, t)).stringify() for i in xrange(n)]
# vectorizer = CountVectorizer(min_df=1, analyzer='char', ngram_range=(2, 2))
# X = vectorizer.fit_transform(corpus_a)
# analyze = vectorizer.build_analyzer()
# print X.toarray()
# print vectorizer.get_feature_names()


# corpus_sax = ['accbccabbacbabaccbaaaaacccabcacbcbcabbcaaacbacacac',
# 'acacbccacabccacbcacbbccbcbcababcbbabbcabacaaaacbca',
# 'abacbcbccbabbcbabccccabacabccbaccaaaccacbabbcababa',
# 'cbababbcccbcaaaaacaccaabcabccacbbccbccaccbacbaabcc',
# 'aaaacbaaaabcccaacabcccbaaccccbacaabbcacbcabcccbabb',
# 'abcaacbabbcacbbccccaaabbaabaccacabccacbccaacaacaac',
# 'ccaaaccaaaccabaabbaccbbabccbacbacabbabcabbacbbbacc',
# 'cccaacabaacccacccabacbacbcbcaacbabaaabaabbcacbccba',
# 'cacacaabbccccabaacccaaacabccabcbaabaaacacccccaaabc',
# 'bccbcaaacabbaaccaaacaacaccaaaaacabcbcccaccbbcccbcc']


X, y = samples_generator.make_classification(n_samples=50, n_features=20, n_informative=2, n_redundant=0)
X1 = np.array([SAX(i).stringify() for i in X])
# X = vectorizer.fit_transform(X)
# analyze = vectorizer.build_analyzer()
# print X.toarray()
# print vectorizer.get_feature_names()

clf = svm.SVC(kernel='linear')
vectorizer = CountVectorizer(min_df=1, analyzer='char', ngram_range=(1, 2))
svm_pipe = Pipeline([('countvect', vectorizer),
                     ('svc', clf)])
# svm_pipe.fit(X1, y)
# prediction = svm_pipe.predict(X1)
# print svm_pipe.score(X1, y)
score, permutation_scores, pvalue = permutation_test_score(
    svm_pipe, X1, y, scoring="accuracy", cv=StratifiedKFold(y, 2), n_permutations=100, n_jobs=1)

print("Classification score %s (pvalue : %s)" % (score, pvalue))

X2 = np.array([SAX(i).sax() for i in X])
clf = svm.SVC(kernel='linear')
svm_pipe = Pipeline([('svc', clf)])
# svm_pipe.fit(X, y)
# prediction = svm_pipe.predict(X)
# print svm_pipe.score(X, y)

score, permutation_scores, pvalue = permutation_test_score(
    svm_pipe, X2, y, scoring="accuracy", cv=StratifiedKFold(y, 2), n_permutations=100, n_jobs=1)

print("Classification score %s (pvalue : %s)" % (score, pvalue))
