from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)
#print vectorizer
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
