#Default Tagger

#import predefined corpus
import nltk
from nltk.corpus import brown
tags=[tag for(word, tag) in brown.tagged_words()]
most_common_tag = nltk.FreqDist(tags).max()
print(most_common_tag)

from nltk import DefaultTagger
barack = "Barack Hussein Obama II (born August 4, 1961) is an American politician who served as the 44th President of the" \
         " United States from 2009 to 2017. A member of the Democratic Party, he was the first African American to be elected " \
         "to the presidency and previously served as a United States Senator from Illinois (2005–2008)"

tokenised_barack = nltk.word_tokenize(barack)

default_tagger = DefaultTagger(most_common_tag)
def_tagged_barack = default_tagger.tag(tokenised_barack)
print(def_tagged_barack)


print("____________________Lookup Taggers_____________________________________")
#lookup taggers
#Ngarm Taggers Context dependent taggers

sent1= "the quick brown fox jumps over the lazy dog"
training_tags = nltk.pos_tag(nltk.word_tokenize(sent1))
print(training_tags)
print(list(nltk.ngrams(nltk.word_tokenize(sent1),2)))
#now use these tags to train Ngarms tagger
ngarm_tagger = nltk.NgramTagger(n=2, train=[training_tags])
print(ngarm_tagger)

sent2= "the lazy dog was jumped over by the quick brown fox"
training_tags_sent2 = nltk.pos_tag(nltk.word_tokenize(sent2))
print(list(nltk.ngrams(nltk.word_tokenize(sent2),2)))
#print(training_tags_sent2)
sent2_taggers = ngarm_tagger.tag(nltk.word_tokenize(sent2))
print(sent2_taggers)

print("________________unigrams Taggers_____________________________________")
#unigrams tagger

bush ="George Walker Bush (born July 6, 1946) is an American politician who served as the 43rd President of the United States" \
      " from 2001 to 2009. He had previously served as the 46th Governor of Texas from 1995 to 2000.Bush was born in New Haven, " \
      "Connecticut, and grew up in Texas. After graduating from Yale University in 1968 and Harvard Business School in 1975, he" \
      " worked in the oil industry.[3] Bush married Laura Welch in 1977 and unsuccessfully ran for the U.S. House of Representatives" \
      " shortly thereafter. He later co-owned the Texas Rangers baseball team before defeating Ann Richards in the 1994 Texas " \
      "gubernatorial election. Bush was elected President of the United States in 2000 when he defeated Democratic incumbent Vice" \
      " President Al Gore after a close and controversial win that involved a stopped recount in Florida. He became the fourth" \
      " person to be elected president while receiving fewer popular votes than his opponent.[4] Bush is a member of a prominent" \
      " political family and is the eldest son of Barbara and George H. W. Bush, the 41st President of the United States. He is " \
      "only the second president to assume the nation's highest office after his father, following the footsteps of John Adams " \
      "and his son, John Quincy Adams.[5] His brother, Jeb Bush, a former Governor of Florida, was a candidate for the Republican " \
      "presidential nomination in the 2016 presidential election. His paternal grandfather, Prescott Bush, was a U.S. Senator from" \
      " Connecticut."

trump = "Donald John Trump (born June 14, 1946) is the 45th and current President of the United States. Before entering politics," \
        " he was a businessman and television personality.Trump was born and raised in the New York City borough of Queens. He" \
        " received an economics degree from the Wharton School of the University of Pennsylvania and was appointed president of his" \
        " family's real estate business in 1971, renamed it The Trump Organization, and expanded it from Queens and Brooklyn into " \
        "Manhattan. The company built or renovated skyscrapers, hotels, casinos, and golf courses. Trump later started various side" \
        " ventures, including licensing his name for real estate and consumer products. He managed the company until his 2017" \
        " inauguration. He co-authored several books, including The Art of the Deal. He owned the Miss Universe and Miss USA beauty " \
        "pageants from 1996 to 2015, and he produced and hosted the reality television show, The Apprentice, from 2003 to 2015. " \
        "Forbes estimates his net worth to be $3.1 billion."


pos_tag_barack = nltk.pos_tag(barack)
pos_tag_bush = nltk.pos_tag(bush)

unigram_tag = nltk.UnigramTagger(train=[pos_tag_barack,pos_tag_bush])
trump_tag = unigram_tag.tag(nltk.word_tokenize(trump))
print(trump_tag)


print("________________Backoff taggers_____________________________________")

default_tagger_new = DefaultTagger('NN')
patterns=[
    (r'.*ing$','VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*es$', 'VBZ'),
    (r'.*ould$', 'MD'),
    (r'.*\'s$', 'NN$'),
    (r'.*s$', 'NNS'),
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'[Aa][Nn][Dd]', 'CC'),
    (r',', ','),
]

regex_tagger = nltk.RegexpTagger(patterns,backoff=default_tagger)
unigram_tag  = nltk.UnigramTagger(train=[pos_tag_bush,pos_tag_barack],backoff=regex_tagger)
trump_tag = unigram_tag.tag(nltk.word_tokenize(trump))
print(trump_tag)

