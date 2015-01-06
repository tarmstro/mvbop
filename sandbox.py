from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from utils.sax import SAX

vectorizer = CountVectorizer(min_df=1)
corpus = ['This is the first document.',
          'This is the second second document.',
          'And the third one.',
          'Is this the first document']
X = vectorizer.fit_transform(corpus)
#print X
analyze = vectorizer.build_analyzer()
# print analyze("This is a text document to analyze.")
# print vectorizer.get_feature_names()
# print X.toarray()
# print vectorizer.transform(['Something completely new.']).toarray()

n = 100
t = 5
corpus_a = [SAX(np.random.normal(0, 1, t)).stringify() for i in xrange(n)]
corpus_b = [SAX(np.random.normal(0, 1.5, t)).stringify() for i in xrange(n)]
vectorizer = CountVectorizer(min_df=1, analyzer='char', ngram_range=(2, 2))
X = vectorizer.fit_transform(corpus_a)
analyze = vectorizer.build_analyzer()
print X.toarray()
print vectorizer.get_feature_names()


corpus_sax = ['accbccabbacbabaccbaaaaacccabcacbcbcabbcaaacbacacac',
'acacbccacabccacbcacbbccbcbcababcbbabbcabacaaaacbca',
'abacbcbccbabbcbabccccabacabccbaccaaaccacbabbcababa',
'cbababbcccbcaaaaacaccaabcabccacbbccbccaccbacbaabcc',
'aaaacbaaaabcccaacabcccbaaccccbacaabbcacbcabcccbabb',
'abcaacbabbcacbbccccaaabbaabaccacabccacbccaacaacaac',
'ccaaaccaaaccabaabbaccbbabccbacbacabbabcabbacbbbacc',
'cccaacabaacccacccabacbacbcbcaacbabaaabaabbcacbccba',
'cacacaabbccccabaacccaaacabccabcbaabaaacacccccaaabc',
'bccbcaaacabbaaccaaacaacaccaaaaacabcbcccaccbbcccbcc']

vectorizer = CountVectorizer(min_df=1, analyzer='char', ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus_sax)
analyze = vectorizer.build_analyzer()
print X.toarray()
print vectorizer.get_feature_names()




