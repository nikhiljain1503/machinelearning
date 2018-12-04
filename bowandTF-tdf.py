
from sklearn.feature_extraction.text import  CountVectorizer

import pandas as pd

sent1 = "India is a republic country. we are proud Indians"
sent2 = "the current Prime Minister of India is Shri. Narendra Modi."
s1="my name is nikhil"
s2="my name is sunny"

count_vectorizer = CountVectorizer()
dtm = count_vectorizer.fit_transform([s1,s2])

print(pd.DataFrame(data = dtm.toarray(),columns=count_vectorizer.get_feature_names()))

#cosine distance
from sklearn.metrics.pairwise import pairwise_distances
a=pairwise_distances(dtm[0].toarray(),dtm[1].toarray(),metric='cosine')
print(dtm[0].toarray())
print(a)

from scipy.spatial.distance import cosine
print(cosine(dtm[0].toarray(),dtm[1].toarray()))


from sklearn.feature_extraction.text import TfidfVectorizer

tfid_vectors = TfidfVectorizer()
tfid_vectors = tfid_vectors.fit_transform([sent1,sent2])
print(pd.DataFrame(data = tfid_vectors.toarray()))
a1=pairwise_distances(tfid_vectors[0].toarray(),tfid_vectors[1].toarray(),metric='cosine')
print(a1)

print("________________Tf-idf corpus reader__________________________")

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
path="./text_docs/"

president_corpus = PlaintextCorpusReader(path,".*",encoding="utf-8")
tfid_vectors_corpus = TfidfVectorizer(input='filename')
files= [path+filename for filename in list(president_corpus.fileids())]
tf_idf_matrix = tfid_vectors_corpus.fit_transform(raw_documents=files)
barack = tf_idf_matrix.toarray()[0]
bush = tf_idf_matrix.toarray()[1]
trump = tf_idf_matrix.toarray()[2]

print(cosine(barack,bush))
print(cosine(bush,trump))
print(cosine(trump,barack))
