#chunking

from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

barack = "Barack Hussein Obama II (born August 4, 1961) is an American politician who served as the 44th President of the" \
         " United States from 2009 to 2017. A member of the Democratic Party, he was the first African American to be elected " \
         "to the presidency and previously served as a United States Senator from Illinois (2005–2008)"

tokenised_barack = word_tokenize(barack)
pos_list = pos_tag(tokenised_barack)
print(ne_chunk(pos_list))

print("________________________RegexParser__________________________")
#regex Parser much better than default ne_chunk

from nltk import RegexpParser

grammar = r"""Place:{<NNP><NNPS>+}
Date:{<NNP><CD><,><CD>}
Person:{<NNP>}
"""

regParser = RegexpParser(grammar)
reg_lines = regParser.parse(pos_list)
print(reg_lines)
