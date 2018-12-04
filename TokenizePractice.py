import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


from string import punctuation

sent = "India is a republic nation. We are proud Indians"

print(len(sent))
print(sent[0:5])
print(sent[11:19])
word_token = nltk.word_tokenize(sent)
print(word_token)

vocab = set(word_token)
print(sorted(vocab))

vocab_wo_punctuation=[]

#punctuation
for word in vocab:
    if word not  in punctuation:
        vocab_wo_punctuation.append(word)
print(vocab_wo_punctuation)

#OR
print("****")
vocab_wo_punctuation_or = vocab - set(punctuation)
print(vocab_wo_punctuation_or)

print("****")

#parts of speech
pos_list = nltk.pos_tag(vocab_wo_punctuation)
print(pos_list)

#Stemming

stemmer = PorterStemmer()
stemmer2 = SnowballStemmer('english',ignore_stopwords=True)
for ww in vocab_wo_punctuation:
    print(stemmer.stem(ww))
    print(stemmer2.stem(ww))

print(stemmer.stem("having"))
print(stemmer2.stem("having"))

print(SnowballStemmer("porter").stem("generously"))
print(stemmer2.stem("generously"))

print(" ".join(stemmer2.languages))

print("______________________________________________________________________________")
 #Lemmatizer

lemma = nltk.stem.WordNetLemmatizer()
lemma1 = nltk.stem.wordnet.WordNetLemmatizer()

#not working uppercase letters
print(lemma.lemmatize("Dogs"))
print(lemma1.lemmatize("Dogs",pos='v'))

print(lemma.lemmatize("dogs"))
print(lemma1.lemmatize("dogs",pos='v'))

print(lemma.lemmatize("are",pos='v'))
print(lemma1.lemmatize("are",pos='v'))

for ww in vocab_wo_punctuation:
    print(lemma.lemmatize(ww))
    print(lemma1.lemmatize(ww))


print("______________________________________________________________________________")

#stopwords
#there are 179 words in stopwords

stopwordslist=[]
st_words = set(stopwords.words('english'))
print(vocab_wo_punctuation)
for i in vocab_wo_punctuation:
    if i not in st_words:
        stopwordslist.append(i)

print(stopwordslist)

print("_____________________________Frequency Distribution_________________________________________________")

#frequency distribution
text_sample="The Natural Language Toolkit exists thanks to the efforts of dozens of voluntary developers who have contributed functionality" \
            " and bugfixes since the project began in 2000" \
            "Thanks to a hands-on guide introducing programming fundamentals alongside topics in computational linguistics, plus" \
            " comprehensive API documentation, NLTK is suitable for linguists, " \
            "engineers, students, educators, researchers, and industry users alike. NLTK is available for Windows, Mac OS X, and " \
            "Linux. Best of all, NLTK is a free, open source, community-driven project. "
print(dict(nltk.FreqDist(nltk.word_tokenize(text_sample))))

#nltk.FreqDist(text_sample.split()).plot()



print("___________________________N Grams_________________________________________________")

#ngrams
from nltk import ngrams

bigrams = ngrams(vocab_wo_punctuation,2)
print(list(bigrams))


print("_____________________Regex tokenizer____________________________________")
 # different tokenize form

from nltk import regexp_tokenize
s2 = ("Alas, it has not rained today. When, do you think, will it rain again?")
print(regexp_tokenize(s2, r'[,\.\?!"]\s*', gaps=False))
print(regexp_tokenize(s2, r'[,\.\?!"]\s*', gaps=True))
print(nltk.word_tokenize(s2))


s3 = ("<p>Although this is <b>not</b> the case here, we must not relax our vigilance!</p>")
print(regexp_tokenize(s3, r'</?(b|p)>', gaps=False))
print(regexp_tokenize(s3, r'</?(b|p)>', gaps=True))

s4 = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize

print(regexp_tokenize(s4, pattern='\w+|\$[\d\.]+|\S+'))
print(wordpunct_tokenize(s4))
print(blankline_tokenize(s4))
