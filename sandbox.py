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
print analyze("This is a text document to analyze.")
print vectorizer.get_feature_names()
print X.toarray()
print vectorizer.transform(['Something completely new.']).toarray()

n = 1000
ts1 = np.random.normal(0, 1, n)
ts2 = np.random.normal(0, 1.5, n)

print SAX(ts1).stringify()
print SAX(ts2).stringify()
