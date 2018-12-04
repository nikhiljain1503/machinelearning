#detect unusual words

import nltk
def unusual_words(text):
    unusual_vocab=[]
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower for w in nltk.corpus.words.words())
    print(list(text_vocab))
    print(list(english_vocab))
    unusual_words = text_vocab - english_vocab
    for ww in text_vocab:
        if ww not in english_vocab:
            unusual_vocab.append(ww)
    return sorted(unusual_vocab)

sent1 = """Good news! Your order has been delivered, we hope you like them – and why not team them up with our beautiful summer dresses www.link.com”"""
sent2 = "Watching a telugu movie.. wat abt u?"
print(unusual_words(nltk.wordpunct_tokenize(sent2)))
print(sorted(nltk.wordpunct_tokenize(sent2)))

if 'abt' in nltk.corpus.words.words():
    print("yes")
else:
    print("no")

print("_____________________minimum edit distance_______________________________________")

#mininmum edit distance
unusual_words_found = ['knows','lol','nw','sms','urgnt','abt']
from nltk.metrics import edit_distance

print(len(nltk.corpus.words.words()))
possible_suggestion={}
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
for unusual_word in unusual_words_found:
    for word in english_vocab:
        dist = edit_distance(unusual_word,word)
        if dist < len(unusual_word)/2:
            if unusual_word not in possible_suggestion.keys():
                possible_suggestion[unusual_word]= [word]
            else:
                possible_suggestion[unusual_word].append(word)

print(possible_suggestion['lol'])




